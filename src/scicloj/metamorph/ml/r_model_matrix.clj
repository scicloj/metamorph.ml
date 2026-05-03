(ns scicloj.metamorph.ml.r-model-matrix
  (:require
   [cemerick.pomegranate :as pom]
   [cemerick.pomegranate.aether :as aether]
   [cheshire.core :as json]
   [clojure.string :as str]
   [opencpu-clj.ocpu :as ocpu]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]))

(defn- add-clojisr-dependency []
  (pom/add-dependencies
   :classloader (clojure.lang.RT/baseLoader)
   :coordinates '[[scicloj/clojisr "1.1.0"]]
   :repositories (merge cemerick.pomegranate.aether/maven-central
                        {"clojars" "https://clojars.org/repo"})))

(defn- add-renjin-deps []

  (pom/add-dependencies
   :classloader (clojure.lang.RT/baseLoader)
   :coordinates '[[org.renjin/renjin-script-engine "3.5-beta76"]]
   :repositories (merge aether/maven-central
                        {"bedatadriven-public" "https://nexus.bedatadriven.com/content/groups/public/"})))


(defn- object-id [base-url library object-name params]
  (-> (ocpu/object base-url
                   :library library
                   :R object-name
                   params)
      :result
      first (str/split #"/") (nth 3)))

(defn- model-matrix--ocpu [ds r-formula]
  (let [base-url "https://cloud.opencpu.org"
        
        ds--json
        (apply merge
               (map
                (fn [col]
                  {col
                   (json/encode (vec (get ds col)))})
                (keys ds)))
        df-object  (-> (object-id base-url :tibble :tibble ds--json))
        formula-object  (-> (object-id base-url :stats :formula {:x r-formula}))


        model-matrix-result
        (->
         (ocpu/object base-url :library :stats :R "model.matrix"
                      {:object formula-object
                       :data df-object})
         :result)

        
        model-matrix-object
        (-> model-matrix-result
            first
            (str/split #"/")
            (nth 3))
        

        attributes (ocpu/object base-url :library :base :R "attributes"
                       {:x model-matrix-object} :json)
        
        col-names
        (->
         (ocpu/object base-url :library :base :R "colnames"
                      {:x model-matrix-object} :json)
         :result)

        model-matrix
        (ocpu/session base-url (first model-matrix-result) :json)
        
        dataset
        (->
         (tc/dataset (:result model-matrix))
         (ds/rename-columns col-names))
        ]

    
    {:attributes attributes
     :model-matrix-dataset dataset}
    ))


(defn- construct [class-name args]
  (let [object-array
        (if (empty? args) 
          (object-array [])
          (object-array [args]))
        ]
    (clojure.lang.Reflector/invokeConstructor
     (Class/forName class-name)
     object-array)))


(defn- model-matrix--renjine [ds r-formula]
  (add-renjin-deps)


  (let [factory  (construct "org.renjin.script.RenjinScriptEngineFactory" (to-array []))
        engine (.getScriptEngine factory)
        _ (run!
           (fn [[k v]]
             (case (dt/elemwise-datatype v)
               :keyword (.put engine (name k) (construct "org.renjin.sexp.StringArrayVector" (into-array String v)))
               :string (.put engine (name k) (construct "org.renjin.sexp.StringArrayVector" (into-array String v)))
               :float64 (.put engine (name k) (construct "org.renjin.sexp.DoubleArrayVector" (double-array v)))
               :int16 (.put engine (name k) (construct "org.renjin.sexp.IntArrayVector" (int-array v)))))
           ds)

        _ (.eval engine
                 (format "mydata=data.frame(%s)"
                         (->>
                          ds
                          keys
                          (map name)
                          (str/join ","))))


        _
        (.eval
         engine
         (format "mm = model.matrix(%s, mydata)" r-formula))

        design-matrix
        (.eval
         engine
         (format "data.frame(mm)"))

        col-names (.. design-matrix
                      getNames
                      toArray)

        attributes
        (.. engine
            (eval "mm")
            (getAttributes)
            toMap
            )

        dataset
        (->>
         (mapv
          (fn [col]
            (let [element-vector (.getElementAsVector design-matrix col)
                  values
                  (case (.getName (class element-vector))
                    "org.renjin.sexp.DoubleArrayVector"
                    (.toDoubleArray element-vector))]
              {col values}))
          col-names)
         (apply merge)
         tc/dataset)]


    {:attributes attributes
     :model-matrix-dataset dataset}))


(defn- model-matrix--clojisr [ds r-formula]
  (add-clojisr-dependency)
  (let [clj->r (requiring-resolve 'clojisr.v1.r/clj->r)
        r->clj (requiring-resolve 'clojisr.v1.r/r->clj)
        r (requiring-resolve 'clojisr.v1.r/r)
        ds--r (clj->r ds)
        model-matrix--r
        (->
         (r (format "model.matrix(as.formula(%s),%s)" r-formula (:object-name ds--r))))]
    {:attributes (r->clj (r (format "attributes(%s)" (:object-name model-matrix--r))))
     :model-matrix-dataset (r->clj model-matrix--r)}))
 
 
 
 (defn r-model-matrix
   "Compute a model matrix from a dataset and an R-style formula.

   Parameters:
    
   - `ds`         A tech.ml.dataset dataset representing the input data.
   - `r-formula`  A string containing the R formula to use for model matrix construction. The formua is interpreted by R itself, so should be full compatible
   - `impl`       An implementation keyword, either 
     - `:ocpu`    Uses an online service https://www.opencpu.org/api.html (server: cloud.opencpu.org)
     - `:renjine` Uses https://renjin.org/   
     - `:clojisr` Uses https://github.com/scicloj/clojisr, which requires a local R installation 


   Returns a dataset containing the constructed design matrix.
   Dispatches to the appropriate backend implementation.

    
   Returns a map with 
   - `:model-matrix-dataset` having the TMD containing the design matrix specified by `r-formula`
   - `:attributes` the (R) attributes of the model.matrix object
    
    "
   [ds r-formula impl]
   
   (case impl
     :ocpu (model-matrix--ocpu ds r-formula)
     :renjine (model-matrix--renjine ds r-formula)
     :clojisr (model-matrix--clojisr ds r-formula)
     )
   )
 

(defn lm
  "Train a linear model using an R-style formula.

   This function combines R formula-based feature engineering with ordinary least
   squares (OLS) regression. It creates a design matrix from the input dataset using
   the specified R formula, then trains a linear model on the resulting features.

   Parameters:
   - `ds`             A tech.ml.dataset dataset containing the input data with all
                      variables referenced in the formula and target variable.
   - `formula`        A string containing the R formula (e.g., \"y ~ x1 + x2 * x3\").
                      The formula is interpreted by the R backend.
   - `target-var`     A keyword or string naming the target variable for regression.
                      This variable must be present in the input dataset.
   - `formula-impl`   An implementation keyword for formula evaluation:

     - `:ocpu`    Uses OpenCPU (cloud.opencpu.org), no local R needed
     - `:renjine` Uses Renjin, a Java implementation of R
     - `:clojisr` Uses clojisr with local R installation

   Returns:
   A trained linear model (OLS from fastmath) ready for predictions. The model
   excludes the intercept column and row names from the design matrix by default.

   Example:
   (lm iris-data \"Sepal.Width ~ Sepal.Length + Petal.Length\" :Sepal.Width :renjine)"
  [ds formula target-var formula-impl]
  (-> ds
   (r-model-matrix formula formula-impl)
   :model-matrix-dataset
   (tc/drop-columns [:$row.names "(Intercept)" "X.Intercept."])
   (tc/add-column target-var (get ds target-var))
   (ds-mod/set-inference-target [target-var])
   (ml/train {:model-type :fastmath/ols})
   :model-data))


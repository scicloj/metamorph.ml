(ns scicloj.metamorph.ml.impl.r
  
  
  (:require
    [cheshire.core :as json]
    [clojure.string :as str]
    [tablecloth.api :as tc]
    [tech.v3.dataset :as ds]
    [tech.v3.datatype :as dt]
    [malli.dev.pretty :as pretty]))

(defn- object-id [ocpu-object base-url library object-name params]
  (let [object-result (ocpu-object base-url
                                   :library library
                                   :R object-name
                                   params)]
    (assert (contains? #{200 201}
                       (-> object-result :status)) (format "HTTP status from '%s' !=200 or 201 : %s" base-url (-> object-result :status)))
    (-> object-result
        :result
        first (str/split #"/") (nth 3))))

(defn model-matrix--ocpu [ds r-formula]
  (let [base-url "https://cloud.opencpu.org"
        ocpu-object (requiring-resolve 'opencpu-clj.ocpu/object)
        ocpu-session (requiring-resolve 'opencpu-clj.ocpu/session)

        ds--json
        (apply merge
               (map
                (fn [col]
                  {col
                   (json/encode (vec (get ds col)))})
                (keys ds)))
        df-object  (-> (object-id ocpu-object  base-url :tibble :tibble ds--json))
        formula-object  (-> (object-id ocpu-object  base-url :stats :formula {:x r-formula}))


        model-matrix-result
        (->
         (ocpu-object base-url :library :stats :R "model.matrix"
                      {:object formula-object
                       :data df-object})
         :result)

        model-matrix-object
        (-> model-matrix-result
            first
            (str/split #"/")
            (nth 3))


        attributes (ocpu-object base-url :library :base :R "attributes"
                                {:x model-matrix-object} :json)

        col-names
        (->
         (ocpu-object base-url :library :base :R "colnames"
                      {:x model-matrix-object} :json)
         :result)

        model-matrix
        (ocpu-session base-url (first model-matrix-result) :json)

        dataset
        (->
         (tc/dataset (:result model-matrix))
         (ds/rename-columns col-names))]


    {:attributes attributes
     :model-matrix-dataset dataset}))


(defn- construct [class-name args]
  (let [object-array
        (if (empty? args)
          (object-array [])
          (object-array [args]))]
    (clojure.lang.Reflector/invokeConstructor
     (Class/forName class-name)
     object-array)))


(defn model-matrix--renjine [ds r-formula]

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
                          (map #(format "`%s`" (name %)))
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
            toMap)

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


(defn model-matrix--clojisr [ds r-formula]
  (let [clj->r (requiring-resolve 'clojisr.v1.r/clj->r)
        r->clj (requiring-resolve 'clojisr.v1.r/r->clj)
        r (requiring-resolve 'clojisr.v1.r/r)
        ds--r (clj->r ds)
        model-matrix--r
        (->
         (r (format "model.matrix(as.formula(%s),%s)" r-formula (:object-name ds--r))))]
    {:attributes (r->clj (r (format "attributes(%s)" (:object-name model-matrix--r))))
     :model-matrix-dataset (r->clj model-matrix--r)}))



(defn pretty--clojisr [s opts]
  (let [clj->r (requiring-resolve 'clojisr.v1.r/clj->r)
        r->clj (requiring-resolve 'clojisr.v1.r/r->clj)
        r (requiring-resolve 'clojisr.v1.r/r)
        pretty (r "pretty")]

    (r->clj
     (apply pretty
            (apply concat
                   (merge {:x s}
                          opts))))

    ))


(defn pretty--ocpu [s opts]
  (let [base-url "https://cloud.opencpu.org"
        ocpu-object (requiring-resolve 'opencpu-clj.ocpu/object)
        encoded-opts
        (apply merge
               (map
                (fn [[k v]]
                  {k (json/encode v)})

                opts))]
    (->
     (ocpu-object base-url :library "base" :R "pretty"
                  (merge
                   {:x (json/encode s)}
                   encoded-opts) :json)
     :result
     double-array
     seq
     )))



(defn pretty--renjine [s opts]

  (let [factory (construct "org.renjin.script.RenjinScriptEngineFactory" (to-array []))
        engine (.getScriptEngine factory)]
    (.put engine "x" (construct "org.renjin.sexp.DoubleArrayVector" (double-array s)))
    (.put engine "n" (construct "org.renjin.sexp.IntArrayVector" (int-array [(get opts :n 5)])))

    (->
     (.eval engine "pretty(x=x,n=n)")
     .toDoubleArray
     seq)))

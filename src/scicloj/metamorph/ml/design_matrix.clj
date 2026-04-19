(ns scicloj.metamorph.ml.design-matrix
  (:require
   [cemerick.pomegranate :as pom]
   [cemerick.pomegranate.aether :as aether]
   [cheshire.core :as json]
   [clojure.set :as set]
   [clojure.string :as str]
   [clojure.walk :as cljwalk]
   [opencpu-clj.ocpu :as ocpu]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dt]))

(defn- combine-with-dash [arg1 arg2]
  (let [to-string (fn [x]
                    (cond
                      (string? x) x
                      (keyword? x) (name x)
                      (symbol? x) (name x)
                      :else (str x)))
        combined-str (str (to-string arg1) "-" (to-string arg2))]
    (cond
      (keyword? arg1) (keyword combined-str)
      (symbol? arg1) (symbol combined-str)
      (string? arg1) combined-str
      :else combined-str)))  

(defn map-column->columns
  "Expands a column containing maps into multiple separate columns.

  `ds` - Dataset
  `src-col` - Column name containing map values

  Returns a new dataset where the map column is replaced with individual columns
  for each map key. New column names are formed by combining the source column
  name with each map key using dashes (e.g., `:src-key1`, `:src-key2`).

  Example: Column `:stats` with `{:mean 5 :std 2}` becomes `:stats-mean` and
  `:stats-std` columns.

  Used for feature expansion in design matrix creation."
  [ds src-col]
  (let [columns-ds
        (tc/dataset (get ds src-col))

        new-col-names
        (map #(combine-with-dash src-col %)
             (tc/column-names columns-ds))

        renamed-columns-ds
        (tc/rename-columns columns-ds
                           (zipmap
                            (tc/column-names columns-ds)
                            new-col-names))]
    (->
     (ds/append-columns ds (tc/columns renamed-columns-ds))
     (ds/remove-column src-col))))



(defn- create-design-matrix-column [ds new-column spec]
  (let [new-column (if (nil? new-column)
                     (str spec)
                     new-column)

        cols (filterv
              #(tc/has-column? ds %)

              (distinct
               (concat (flatten spec)
                       (when (map? spec)
                         (flatten
                          (vals spec))))))


        syms
        (mapv
         (fn [col] {:sym (gensym)
                    :col col})
         cols)

        col-sym-m (zipmap  (mapv :col syms)
                           (mapv :sym syms))

        fnn
        (list 'fn (mapv :sym syms)
              (cljwalk/postwalk-replace col-sym-m spec))
        f (eval fnn)]

    (tc/map-columns ds new-column cols f)))


(defn create-design-matrix
  "Converts the given dataset into a full numeric dataset.
   
   `ds` Is te tech.v3 dataset to transform
   `target-specs` are the specifications how to transform the target variables
   `features-specs` are the specifications how to transform the features 

   The 'spec' can express several types of dataset transformations in a compact way:
   - add new derived columns
   - remove columns
   - rename columns
   - convert columns to categorical
   - set inference target


   Function calls need to be given as lists (quoted by '), and can refer to column names.
   They get evaluated from top->bottom, and can refer to each other.
   
   The followig aliases can be used as part of the spec.
   (Other functions need to be full qualified).

   clojure.core  can be used without full qailifying te symbols
   ds             (tech.v3.dataset)
   tc             (tablecloth.api)
   tcc            (tablecloth.column.api)
   

   Example:
 
   (dm/create-design-matrix
         ds
         [:y] 
         [         
          [:sum '(+ :a :b :c)]
         ])
   
   This will:
   - set inference target to y:
   - create a new derived variables :sum
   - remove all columns except :y and :sum
   

   See  `design_matrix_test.clj` for more examples.
   
   "
  [ds
   targets-specs
   features-specs]

  ;; be sure, they are available
  (require '[tech.v3.dataset :as ds]
           '[tablecloth.api :as tc]
           '[tablecloth.column.api :as tcc])
  
  (let [ds-columns (tc/column-names ds)

        mapping-specs-cols
        (concat targets-specs
                (->> features-specs
                     (map first)
                     (remove nil?)))

        columns-to-be-removed
        (set/difference (into #{} ds-columns)
                        (into #{} mapping-specs-cols))


        transformed-ds
        (reduce
         (fn [ds e2]
           (create-design-matrix-column ds
                                        (first e2)
                                        (second e2)))
         ds
         features-specs)

        all-cols (tc/column-names transformed-ds)


        seperated-ds
        (reduce
         (fn [ds col]
           (let [new-ds (if  (-> (get ds col) first sequential?)
                          (tc/array-column->columns ds col {:prefix col})
                          ds)
                 new-ds (if  (-> (get ds col) first map?)
                          (map-column->columns ds col)
                          new-ds)]
             new-ds))
         transformed-ds
         all-cols)]


    (-> seperated-ds
        (ds-mod/set-inference-target targets-specs)
        (tc/drop-columns columns-to-be-removed)
        (ds/categorical->number cf/categorical))))


(defn- object-id [base-url library object-name params]
  (-> (ocpu/object base-url 
                   :library library
                   :R object-name
                   params)
      :result
      first (str/split #"/") (nth 3))
  )

(defn model-matrix--ocpu [ds r-formula]
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

        col-names
        (->
         (ocpu/object base-url :library :base :R "colnames"
                      {:x model-matrix-object} :json)
         :result)

        model-matrix
        (ocpu/session base-url (first model-matrix-result) :json)]

    (->
     (tc/dataset (:result model-matrix))
     (ds/rename-columns col-names))))

(defn add-renjin-deps []

  (pom/add-dependencies :coordinates '[[org.renjin/renjin-script-engine "3.5-beta76"]]
                        :repositories (merge aether/maven-central
                                             {"bedatadriven-public" "https://nexus.bedatadriven.com/content/groups/public/"}))
  (import '[javax.script ScriptEngine]
          '[org.renjin.primitives.matrix Matrix]
          '[org.renjin.sexp ListVector]
          '[org.renjin.sexp DoubleArrayVector]
          '[org.renjin.sexp LongArrayVector]
          '[org.renjin.sexp IntArrayVector]
          '[org.renjin.sexp StringArrayVector]
          '[org.renjin.script RenjinScriptEngineFactory]);;=> {:datatype :int64, :lookup-table {1 0, 0 1}, :values [1]}
  )

(defn model-matrix--renjine [ds r-formula]
  (add-renjin-deps)
  

  (let [factory  (RenjinScriptEngineFactory.)
        engine (.getScriptEngine factory)
        _ (run!
           (fn [[k v]]
             (case (dt/elemwise-datatype v)
               :keyword (.put engine (name k) (StringArrayVector. (into-array String v)))
               :string (.put engine (name k) (StringArrayVector. (into-array String v)))
               :float64 (.put engine (name k) (DoubleArrayVector. (double-array v)))
               :int16 (.put engine (name k) (IntArrayVector. (int-array v)))))
           ds)

        _ (.eval engine
                 (format "mydata=data.frame(%s)"
                         (->>
                          ds
                          keys
                          (map name)
                          (str/join ","))))


        ^ListVector design-matrix
        (.eval
         engine
         (format "data.frame(model.matrix(%s, mydata))" r-formula))
        col-names (.. design-matrix
                      getNames
                      toArray)]

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
     tc/dataset)))
  

  

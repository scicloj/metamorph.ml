(ns scicloj.metamorph.ml.design-matrix
  (:require
   [clojure.set :as set]
   [clojure.walk :as cljwalk]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]))

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

(defn map-column->columns [ds src-col]
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
   
   `target-specs` are the specifications how to transform the target variables
   `features-specs` are the specifications how to transform the features 

   The 'spec' can express several types of dataset transformations in a compact way:
   - add new dervied columns
   - remove columns
   - rename columns
   - convert to catgorical
   - set inference target


   Function calls need to be given as lists (quoted by '), and can refer to column names.
   They get evaluated from to->bottom, so can refer to each other.
   
   The followig aliases can be used as part of the spec.
   (Other function needs to be full qualified).

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




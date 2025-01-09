(ns scicloj.metamorph.ml.design-matrix
  (:require [tablecloth.api :as tc]
            [clojure.set :as set]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.impl.column :as col-impl]))

(defn- create-design-matrix-column [ds new-column spec]

  (def ds ds)
  (def new-column new-column)
  (def spec spec)
  
  
  
  (let [cols+params 
        (flatten (vec (rest spec)))
        
        _ (def cols+params cols+params)

        cols (filterv
              #(tc/has-column? ds %)
              cols+params)
        
        params (filterv
                #(not (tc/has-column? ds %))
                cols+params)

        _ (def params params)
        _ (def cols cols)
        syms
        (mapv
         (fn [_] (gensym))
         cols)
        
        _ (def syms syms)

        fnn
        (list 'fn syms 
              (concat 
              [(first spec)]
               params
               syms))

        _ (def fnn fnn)
        f (eval fnn)

        _ (def f f)
        ]

    (tc/map-columns ds new-column cols f)))


(tc/map-columns ds new-column cols f)
(map class spec)

( (eval fnn) 1 2 3)

(defn create-design-matrix [ds
                            targets-specs
                            feature-specs]

  (let [ ds-columns (tc/column-names ds)

        mapping-specs-cols
        (concat targets-specs
                (->> feature-specs
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
         feature-specs)
        all-cols (tc/column-names transformed-ds)


        seperated-ds
        (reduce

         (fn [ds col]
           (if  (-> (get ds col) first sequential?)
             (tc/array-column->columns ds col {:prefix col})
             ds))
         transformed-ds
         all-cols)

        design-matrix
        (-> seperated-ds
            (ds-mod/set-inference-target targets-specs)
            (tc/drop-columns columns-to-be-removed)
            (ds/categorical->number cf/categorical))]


    
    design-matrix))

(ns scicloj.metamorph.ml.design-matrix
  (:require [tablecloth.api :as tc]
            [clojure.set :as set]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]))

(defn- create-design-matrix-column [ds new-colum spec]

  (let [col-set (into #{} (tc/column-names ds))
        new-colum (if (nil? new-colum)
                    (str spec)
                    new-colum)


        matching-keyword-cols
        (->> (-> spec flatten)
             (filterv
              #(contains? col-set (keyword %)))
             (into #{}))


        matching-string-cols
        (->> (-> spec flatten)
             (filterv
              #(contains? col-set (str %)))
             (into #{}))


        matching-symbol-cols
        (->> (-> spec flatten)
             (filterv
              #(contains? col-set %))
             (into #{}))


        cols
        (mapv #(cond (contains? matching-string-cols %) (str %)
                     (contains? matching-symbol-cols %) (symbol %)
                     (contains? matching-keyword-cols %) (keyword %)
                     :else (throw (Exception. "column not found: %")))


              (concat (vec matching-string-cols)
                      (vec matching-keyword-cols)
                      (vec matching-symbol-cols)))


        duplicates
        (filter #(> (val %) 1)
                (frequencies cols))



        _ (when (seq duplicates)
            (throw (Exception. (str  "ambigous column names found" (str  (mapv first duplicates))))))


        params (mapv symbol cols)
        decl (list 'fn params)

        fnn
        (list (first decl) params spec)]

    (tc/map-columns ds new-colum cols (eval fnn))))




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

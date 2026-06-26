(ns scicloj.metamorph.ml.impl.dsutils
 {:no-doc true} 
  (:require
    [tablecloth.api :as tc]
    [tech.v3.dataset :as ds]))

(defn cast-cols-to-categorical-string [dataset columns]
  (reduce
   (fn [ds col]
     (-> ds
         (ds/column-cast col :string)
         (ds/assoc-metadata [col] :categorical? true)))
   dataset
   columns))

(defn- factor-lump-seq [factor-seq n other-name]
  (let  [to-keep (into #{} (->> factor-seq frequencies (sort-by second) reverse (take n) keys))]

    (map
     #(if (contains? to-keep %)
        %
        other-name)
     factor-seq)))

(defn lump-categories
  "Lump uncommon factor levels together into 'other'
   `n`: number of different categories to be left
   `other-name`: category used for others
   "


  [dataset columns  & {:keys [other-name n] :or {other-name ":other:" n 10}}]
  (reduce
   (fn [ds col]
     (tc/add-column ds col
                    (factor-lump-seq (get ds col) n other-name)))
   dataset
   columns))

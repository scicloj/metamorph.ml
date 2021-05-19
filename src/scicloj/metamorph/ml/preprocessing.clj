(ns scicloj.metamorph.ml.preprocessing
  (:require
   [tech.v3.dataset.math :as std-math]
   [tech.v3.dataset :as ds]
   ))

(defn std-scale
  "Metamorph transfomer, which centers and scales the dataset per column.

  `col-seq` ia a seuqnce of columns names to workon
  `mean?` If true (default), the data gets shifted by the column means, so 0 centered
  `stddev?` If true (default), the data gets scaled by the standard deviation of the column
  "
  [col-seq {:keys [mean? stddev?]
            :or {mean? true stddev? true}
            :as options}]
  (fn [{:metamorph/keys [data id mode] :as ctx}]
    (case mode
      :fit
      (let [ds (ds/select-columns data col-seq)
            fit-std-xform (std-math/fit-std-scale ds options)]
        (assoc ctx
               id {:fit-std-xform fit-std-xform}
               :metamorph/data (merge data (std-math/transform-std-scale ds fit-std-xform))))
      :transform
      (assoc ctx :metamorph/data
             (merge data
                    (-> (ds/select-columns data col-seq)
                        (std-math/transform-std-scale
                         (get-in ctx [id :fit-std-xform]))))))))

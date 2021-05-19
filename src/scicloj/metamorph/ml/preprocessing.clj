(ns scicloj.metamorph.ml.preprocessing
  (:require
   [tech.v3.dataset.math :as std-math]
   [tech.v3.dataset :as ds]
   ))

(defn std-scale
  "Metamorph transfomer, which centers and scales the dataset per column.

  `col-seq` is a sequence of column names to work on

  `options` are the options for the scaler and can take:

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
(defn min-max-scale
  "Metamorph transfomer, which scales the column data into a given range.

  `col-seq` is a sequence of columns names to work on

  `options` Options for scaler, can take:

  `min` Minimal value to scale to (default -0.5)

  `max` Maximum value to scale to (default 0.5)
  "
  [col-seq {:keys [min max]
            :or {min -0.5
                 max 0.5}
            :as options} ]
  (fn [{:metamorph/keys [data id mode] :as ctx}]
    (case mode
      :fit
      (let [ds (ds/select-columns data col-seq)
            fit-minmax-xform (std-math/fit-minmax ds options)]
        (assoc ctx
               id {:fit-minmax-xform fit-minmax-xform}
               :metamorph/data (merge data (std-math/transform-minmax ds fit-minmax-xform))))
      :transform
      (assoc ctx :metamorph/data
             (merge data
                    (-> (ds/select-columns data col-seq)
                        (std-math/transform-minmax
                         (get-in ctx [id :fit-minmax-xform]))))))))

(ns scicloj.metamorph.ml.preprocessing
  (:require
   [tech.v3.dataset.math :as std-math]
   [tech.v3.dataset :as ds]
   ))


(defn preprocessor [col-seq fit-fn transform-fn context-key options]

  (fn [{:metamorph/keys [data id mode] :as ctx}]
    (case mode
      :fit
      (let [ds (ds/select-columns data col-seq)
            fit-xform (fit-fn ds options)]
        (assoc ctx
               id {context-key fit-xform}
               :metamorph/data (merge data (transform-fn ds fit-xform))))
      :transform
      (assoc ctx :metamorph/data
             (merge data
                    (-> (ds/select-columns data col-seq)
                        (transform-fn
                         (get-in ctx [id context-key]))))))))

(defn std-scale
  "Metamorph transfomer, which centers and scales the dataset per column.

  `col-seq` is a sequence of column names to work on

  `options` are the options for the scaler and can take:

  `mean?` If true (default), the data gets shifted by the column means, so 0 centered

  `stddev?` If true (default), the data gets scaled by the standard deviation of the column

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Centers and scales the dataset at key `:metamorph/data` and stores the trained model in ctx under key at `:metamorph/id`
  Behaviour in mode :transform         | Reads trained std-scale model from ctx and applies it to data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads trained model to use for from key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores trained model in key $id

  "
  [col-seq {:keys [mean? stddev?]
            :or {mean? true stddev? true}
            :as options}]
  (preprocessor col-seq std-math/fit-std-scale std-math/transform-std-scale :fit-std-xform options))

(defn min-max-scale
  "Metamorph transfomer, which scales the column data into a given range.

  `col-seq` is a sequence of columns names to work on

  `options` Options for scaler, can take:

  `min` Minimal value to scale to (default -0.5)

  `max` Maximum value to scale to (default 0.5)

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Scales the dataset at key `:metamorph/data` and stores the trained model in ctx under key at `:metamorph/id`
  Behaviour in mode :transform         | Reads trained min-max-scale model from ctx and applies it to data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads trained model to use for from key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores trained model in key $id

  "
  [col-seq {:keys [min max]
            :or {min -0.5
                 max 0.5}
            :as options} ]
  (preprocessor col-seq std-math/fit-minmax std-math/transform-minmax :fit-minmax-xform options))

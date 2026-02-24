(ns scicloj.metamorph.ml.categorical
  (:require [tech.v3.dataset.categorical :as ds-cat]
            [tech.v3.dataset :as ds]
            [clojure.set :as set]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml.malli :as malli]))
            

(defn- apply-mappings [ds one-hot-encodings]
  (reduce (fn [data one-hot-encoding]
            (ds-cat/transform-one-hot data one-hot-encoding))
          ds
          one-hot-encodings))

(defn- transform-one-hot-full [ctx data mode id col-names options]
  (case mode
    :fit
    (let [_ (assert (ctx :metamorph.ml/full-ds) "Context need to contain full dataset at key :metamorph.ml/full-ds")
          mappings
          (map (fn [col]
                 (assert (get (:metamorph.ml/full-ds ctx) col) (format  "full-ds need to have col: %s" col))
                 (ds-cat/fit-one-hot
                  (:metamorph.ml/full-ds ctx)
                  col
                  (:table-args options)
                  (:result-datatype options)))
               col-names)]
      (assoc ctx
             id mappings
             :metamorph/data (apply-mappings data mappings)))

    :transform
    (assoc ctx :metamorph/data (apply-mappings data (get ctx id)))))


(defn- validate-mappings [data mappings]
  (run!
   (fn [mapping]
     (let [
           levels-in-mapping (-> mapping :one-hot-table keys set)
           levels-in-data (->  (get data (:src-column mapping)) distinct set)
           levels-not-mapped (set/difference levels-in-data levels-in-mapping)]
       (when (pos-int? (count levels-not-mapped))
         (throw (IllegalArgumentException. (str  "Some levels of data in :transform were not in :fit for colum xxx: " levels-not-mapped))))))
   mappings))

(defn- transform-one-hot-train->test [ctx data mode id col-names options]
  (case mode
    :fit
    (let [mappings
          (map (fn [col]
                 (ds-cat/fit-one-hot data col
                                     (:table-args options)
                                     (:result-datatype options)))
               col-names)]

      (assoc ctx
             id mappings
             :metamorph/data (apply-mappings data mappings)))

    :transform
    (let [mappings (get ctx id)
          _ (validate-mappings data mappings)]
      (assoc ctx :metamorph/data (apply-mappings data mappings)))))



(defn transform-one-hot
  "Metamorph transformer that maps categorical variables to one-hot encoded columns.

  Each unique value of the categorical column becomes its own binary column in
  the one-hot encoding.

  `column-selector` - Tablecloth column selector (keyword, fn, or selector spec)
  `strategy` - Strategy for handling train/test level differences:
               * `:full` - Levels retrieved from dataset at `:metamorph.ml/full-ds` in context
               * `:independent` - One-hot columns fitted and transformed independently
               * `:fit` - Mapping from :fit mode used in :transform (assumes all levels present in fit)
  `options` - Optional map with:
              * `:table-args` - Precise mapping as sequence of [val idx] pairs or sorted values
              * `:result-datatype` - Datatype of the one-hot-mapping columns

  Returns a metamorph step function that transforms the data in both :fit and
  :transform modes.

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Fits one-hot encoding and applies it to `:metamorph/data`
  Behaviour in mode :transform         | Applies fitted encoding to `:metamorph/data`
  Reads keys from ctx                  | In `:transform`: reads fitted encoding from `:metamorph/id`
  Writes keys to ctx                   | In `:fit`: stores fitted encoding in `:metamorph/id`

  See also: `tech.v3.dataset.categorical/fit-one-hot`, `tech.v3.dataset/categorical->one-hot`"
  ([column-selector strategy] (transform-one-hot column-selector strategy nil))
  ([column-selector strategy options]
   (malli/instrument-mm
    (fn [{:metamorph/keys [id data mode] :as ctx}]
      (let [col-names (if (fn? column-selector)
                        (tc/column-names data column-selector :all)
                        (tc/column-names data column-selector))]
        (case strategy
          :full (transform-one-hot-full ctx data mode id col-names options)
          :independent (assoc ctx :metamorph/data
                              (ds/categorical->one-hot
                               data
                               (tc/select-columns data column-selector)
                               (:table-args options)
                               (:result-datatype options)))
          :fit (transform-one-hot-train->test ctx data mode id col-names options)))))))

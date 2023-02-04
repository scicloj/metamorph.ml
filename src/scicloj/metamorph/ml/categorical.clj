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

  "Transformer which mapps categorical variables to numbers. Each value of the
  column gets its won column in one-hot-encoding.

  To handle different levls of a variable between train an test data, three
  strategies are available:

  * `:full`  The levels are retrieved from a dataset at key :metamorph.ml/full-ds in the context
  * `:independent`  One-hot columns are fitted and transformed independently for train and test  data
  * `:fit` The mapping fitted in mode :fit is used in :transform, and it is assumed that all levels are present in the data during :fit

  `options` can be:
  *  `:table-args` allows to specify the precise mapping as a sequence of pairs of [val idx] or as a sorted seq of values.
  *  `:result-datatype`  Datatype of the one-hot-mapping column

  "
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

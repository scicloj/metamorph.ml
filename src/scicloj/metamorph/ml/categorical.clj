(ns scicloj.metamorph.ml.categorical
  (:require [tech.v3.dataset.categorical :as ds-cat]
            [tech.v3.dataset :as ds]
            [clojure.set :as set]
            [tablecloth.api :as tc]
            [scicloj.metamorph.core :as mm]))

(defn- apply-mappings [ds one-hot-encodings]
  (reduce (fn [data one-hot-encoding]
            (def data data)
            (def one-hot-encoding one-hot-encoding)
            (ds-cat/transform-one-hot data one-hot-encoding))
          ds
          one-hot-encodings))

(defn transform-one-hot-full [ctx data mode id col-names]
  (case mode
    :fit
    (let [_ (def data data)
          _ (def ctx ctx)

          mappings
          (map (fn [col]
                 (def col col)
                 (ds-cat/fit-one-hot (:metamorph.ml/full-ds ctx) col))
               col-names)]



      ;; (tech.v3.dataset.categorical/tr)

      (def mappings mappings)
      (assoc ctx
             id mappings
             :metamorph/data (apply-mappings data mappings)))

    :transform
    (do
      (def data data)
      (def ctx ctx)
      (def id id)
      (assoc ctx :metamorph/data (apply-mappings data (get ctx id))))))


(defn transform-one-hot-train->test [ctx data mode id col-names]
  (case mode
    :fit
    (let [_ (def data data)
          _ (def ctx ctx)

          mappings
          (map (fn [col]
                 (def col col)
                 (ds-cat/fit-one-hot data col))
               col-names)]



      ;; (tech.v3.dataset.categorical/tr)

      (def mappings mappings)
      (assoc ctx
             id mappings
             :metamorph/data (apply-mappings data mappings)))

    :transform
    (do
      (def data data)
      (def ctx ctx)
      (def id id)
      (let [mappings (get ctx id)
            levels-in-mapping (-> mappings first :one-hot-table keys set)
            levels-in-data (-> data :col-1 distinct set)
            levels-not-mapped (set/difference levels-in-data levels-in-mapping)]
        ;; (def levels-in-mapping levels-in-mapping)
        ;; (def levels-in-data levels-in-data)
        (if (pos-int? (count levels-not-mapped))
          (throw (IllegalArgumentException. (str  "Some levels of data in :transform were not in :fit for colum xxx: " levels-not-mapped))))


        (assoc ctx :metamorph/data (apply-mappings data mappings))))))

(defn transform-one-hot [column-selector strategy]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
    (let [col-names (if (fn? column-selector)
                      (tc/column-names data column-selector :all)
                      (tc/column-names data column-selector))]
      (case strategy
        :complete-ds (transform-one-hot-full ctx data mode id col-names)
        :independent (assoc ctx :metamorph/data (ds/categorical->one-hot data (tc/select-columns data column-selector)))
        :train-to-test (transform-one-hot-train->test ctx data mode id col-names)))))

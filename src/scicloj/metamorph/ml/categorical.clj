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

(defn transform-one-hot-full [ctx data mode id col-names options]
  (case mode
    :fit
    (let [mappings
          (map (fn [col]
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
    (assoc ctx :metamorph/data (apply-mappings data (get ctx id)))()))


(defn transform-one-hot-train->test [ctx data mode id col-names options]
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
          _ (run!
             (fn [mapping]
               (let [
                     levels-in-mapping (-> mapping :one-hot-table keys set)
                     levels-in-data (->  (get data (:src-column mapping)) distinct set)
                     levels-not-mapped (set/difference levels-in-data levels-in-mapping)]
                 (if (pos-int? (count levels-not-mapped))
                   (throw (IllegalArgumentException. (str  "Some levels of data in :transform were not in :fit for colum xxx: " levels-not-mapped))))))
             mappings)]
      (assoc ctx :metamorph/data (apply-mappings data mappings)))))



(defn transform-one-hot
  "Transormer which mapps categorical variables to numbers."
  ([column-selector strategy] (transform-one-hot column-selector strategy nil))
  ([column-selector strategy options]
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
         :fit (transform-one-hot-train->test ctx data mode id col-names options))))))

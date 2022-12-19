(ns scicloj.metamorph.ml.learning-curve
  (:require
   [clojure.math :as math]
   [scicloj.metamorph.ml]
   [scicloj.metamorph.ml.loss]
   [tablecloth.api :as tc]))

(defn learning-curve [ds pipe-fn train-sizes {:keys [k metric-fn loss-or-accuracy]}]
  (let [splits (tc/split->seq ds :kfold {:k k :seed 12345})]
    (->>
               (mapv (fn [{:keys [train test]}]
                       (let [train-test-seq
                             (map-indexed
                              (fn [index train-size]
                                (let [train-subset (tc/head train (math/round (* train-size (tc/row-count train))))]
                                  {:split-uid (str index)
                                   :train train-subset
                                   :test test}))
                              train-sizes)
                             eval-results
                             (scicloj.metamorph.ml/evaluate-pipelines
                              [pipe-fn]
                              train-test-seq
                              metric-fn
                              loss-or-accuracy
                              {:evaluation-handler-fn identity
                               :return-best-pipeline-only false
                               :return-best-crossvalidation-only false})]
                         (map  (fn [result]
                                 (hash-map
                                  :train-size-index (:split-uid result)
                                  :train-ds-size (-> result :fit-ctx :metamorph/data tc/row-count)
                                  :test-ds-size (-> result :test-transform :ctx :model :scicloj.metamorph.ml/target-ds tc/row-count)
                                  :metric-test (get-in result [:test-transform :metric])
                                  :metric-train (get-in result [:train-transform :metric])))
                               (flatten eval-results))))
                     splits)
               flatten
               (tc/dataset))))

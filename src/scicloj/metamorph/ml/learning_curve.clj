(ns scicloj.metamorph.ml.learning-curve
  (:require  [clojure.test :as t]
             [tech.v3.dataset]
             [scicloj.metamorph.core :as mm]
             [tech.v3.dataset.metamorph :as mds]
             [tablecloth.api :as tc]
             [tablecloth.pipeline :as tc-mm]
             [scicloj.metamorph.ml]
             [scicloj.metamorph.ml.loss]
             [scicloj.metamorph.ml.toydata]
             [tech.v3.datatype.functional :as fun]
             [scicloj.metamorph.ml.learning-curve :as lc]
             [scicloj.ml.smile.classification]))


(defn learning-curve [ds pipe-fn train-sizes k]
  (let [splits (tc/split->seq ds :kfold {:k k})
        metrices (->>
                  (mapv (fn [{:keys [train test]}]
                          (let [train-test-seq
                                (map
                                 (fn [train-size]
                                   (let [train-subset (tc/head train (Math/round (* train-size (tc/row-count train))))]
                                     {:train train-subset
                                      :test test}))
                                 train-sizes)
                                _ (def train-test-seq train-test-seq)
                                eval-results
                                (scicloj.metamorph.ml/evaluate-pipelines
                                 [pipe-fn]
                                 train-test-seq
                                 scicloj.metamorph.ml.loss/classification-loss
                                 :loss
                                 {:evaluation-handler-fn identity
                                  :return-best-pipeline-only false
                                  :return-best-crossvalidation-only false})]
                            (map  (fn [index  result]
                                     (def result result)
                                     (hash-map
                                      :index index
                                      ;; :train-size train-size
                                      :train-ds-size (-> result :fit-ctx :metamorph/data tc/row-count)
                                      :test-ds-size (-> result :test-transform :ctx :model :scicloj.metamorph.ml/target-ds tc/row-count)
                                      :metric-test (get-in result [:test-transform :metric])
                                      :metric-train (get-in result [:train-transform :metric])))
                                  (range)
                                  ;; train-sizes
                                  (flatten eval-results))))
                        splits)
                  flatten
                  (tc/dataset))]
    (def metrices metrices)
    (-> metrices
        (tc/group-by :index)
        (tc/aggregate-columns [:metric-test :metric-train :train-ds-size :test-ds-size]
                              [fun/mean fun/mean  fun/mean fun/mean])
        (tc/rename-columns {:$group-name :index})
        (tc/order-by :train-ds-size))))


(->> train-test-seq
     (map :train)
     (map tc/row-count))
;; => (81 162 243 324 405 486 567 648 729 810)

(->> train-test-seq
     (map :test)
     (map tc/row-count))

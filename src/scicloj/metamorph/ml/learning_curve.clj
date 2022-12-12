(ns scicloj.metamorph.ml.learning-curve
  (:require  [clojure.test :as t]
             [tech.v3.dataset]
             [scicloj.metamorph.core :as mm]
             [tech.v3.dataset.metamorph :as mds]
             [tablecloth.api :as tc]
             [tablecloth.pipeline :as tc-mm]
             [clojure.math :as math]
             [scicloj.metamorph.ml]
             [scicloj.metamorph.ml.loss]
             [scicloj.metamorph.ml.toydata]
             [tech.v3.datatype.functional :as fun]
             [scicloj.ml.smile.classification]))




(defn rounded-mean [coll]
  (math/round (fun/mean coll)))


(defn mean+std [col]
  (+
   (fun/mean col)
   (fun/standard-deviation col)))

(defn mean-std [col]
  (-
   (fun/mean col)
   (fun/standard-deviation col)))


(defn learning-curve [ds pipe-fn train-sizes k]
  ;; (def ds ds)
  ;; (def pipe-fn pipe-fn)
  ;; (def train-sizes train-sizes)
  ;; (def k k)

  (let [splits (tc/split->seq ds :kfold {:k k})
        _ (def splits splits)
        metrices (->>
                  (mapv (fn [{:keys [train test]}]
                          (let [train-test-seq
                                (map-indexed
                                 (fn [index train-size]
                                   (let [train-subset (tc/head train (math/round (* train-size (tc/row-count train))))]
                                     {:split-uid (str index)
                                      :train train-subset
                                      :test test}))
                                 train-sizes)
                                ;; _ (def train-test-seq train-test-seq)
                                eval-results
                                (scicloj.metamorph.ml/evaluate-pipelines
                                 [pipe-fn]
                                 train-test-seq
                                 scicloj.metamorph.ml.loss/classification-accuracy
                                 :accuracy
                                 {:evaluation-handler-fn identity
                                  :return-best-pipeline-only false
                                  :return-best-crossvalidation-only false})]
                            (map  (fn [result]
                                    ;; (def result result)
                                    (hash-map
                                     :train-size-index (:split-uid result)
                                     ;; :train-size train-size
                                     :train-ds-size (-> result :fit-ctx :metamorph/data tc/row-count)
                                     :test-ds-size (-> result :test-transform :ctx :model :scicloj.metamorph.ml/target-ds tc/row-count)
                                     :metric-test (get-in result [:test-transform :metric])
                                     :metric-train (get-in result [:train-transform :metric])))

                                  ;; train-sizes
                                  (flatten eval-results))))
                        splits)
                  flatten
                  (tc/dataset))]
    ;; (def metrices metrices)
    (-> metrices
        (tc/group-by :train-size-index)

        (tc/aggregate {:metric-test      #(fun/mean (:metric-test %))
                       :metric-test-min  #(mean-std (:metric-test %))
                       :metric-test-max  #(mean+std (:metric-test %))
                       :metric-train     #(fun/mean (:metric-train %))
                       :metric-train-min #(mean-std (:metric-train %))
                       :metric-train-max #(mean+std (:metric-train %))
                       :train-ds-size    #(rounded-mean (:train-ds-size %))
                       :test-ds-size     #(rounded-mean (:test-ds-size %))}))))

(ns scicloj.metamorph.dmatrix-test
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.metrics :as metrics]
   [scicloj.metamorph.ml.rdatasets :as datasets]
   [scicloj.ml.xgboost]
   [same.core :refer [ish? zeroish? set-comparator!]]
   [same.compare]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.core :as mm]
   [tech.v3.dataset.column-filters :as cf])
  (:import
   [ml.dmlc.xgboost4j.java DMatrix]))

(set-comparator! (same.compare/compare-ulp 1e12 2))
(def dm (DMatrix. "test/data/iris.libsvm.txt?format=libsvm"))

(deftest dmatrix
  (let [model-from-dm (ml/train dm {:model-type :xgboost/classification
                                    :num-class 4})
        prediction (ml/predict dm model-from-dm)
        ]
    
    (is (ish? 0.7866
              (metrics/accuracy
               (.getLabel dm)
               (map float (get prediction :label)))))))

(deftest dataset
  (let [ds (-> (datasets/datasets-iris)
               (tc/drop-columns [:rownames])
               (ds/categorical->number [:species])
               (ds-mod/set-inference-target :species))
        model-from-ds (ml/train ds {:model-type :xgboost/classification})]

    (is (= 1.0
           (metrics/accuracy
            (:species ds)
            (:species (ml/predict ds model-from-ds)))))))

(deftest eval-pipeline--dmatrix
  (let [pipe-fn
        (mm/pipeline
         {:metamorph/id :model}
         (ml/model  {:model-type :xgboost/classification
                     :num-class 4}))
        result
        (ml/evaluate-pipelines
         [pipe-fn]
         [{:train dm :test dm}]
         metrics/accuracy
         :accuracy)]
    (is (ish?
         0.7866
         (-> result first first :test-transform :metric)))
    (is (ish?
         0.7866
         (-> result first first :train-transform :metric)))))





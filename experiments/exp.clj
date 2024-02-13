(ns exp
  (:require
   [scicloj.ml.smile.classification]
   [konserve.filestore :refer [connect-fs-store]]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [taoensso.nippy :as nippy]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.ml.cache :as cache]))

(def iris (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

(ml/define-model! :slow-model
  (fn train
    [feature-ds label-ds options]
    (println "wait 2s")
    (dotimes [n 2]
        (println :wait  n)
        (Thread/sleep 1000)))


  (fn predict [feature-ds thawed-model model]
               

    (println "predict")
    (Thread/sleep 1000)


    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) "versicolor")
                                    {:column-type :prediction})]))
  {})

(def splits (tc/split->seq
             (-> iris
                 (ds/categorical->number [:species]))
             :kfold {:k 11
                     :seed 4753}))
(def  store (connect-fs-store "/tmp/store" :opts {:sync? true}))

(def  pipe-fn-ada (morph/pipeline
                   (morph/lift ds-mod/set-inference-target :species)
                   {:metamorph/id :model} (ml/model {:model-type :smile.classification/ada-boost
                                                     :caching-predict-fn (fn [dataset model]
                                                                           (cache/caching-predict store dataset model))
                                                     :caching-train-fn (fn [dataset options]
                                                                         (cache/caching-train store dataset options))})))

(def  pipe-fn-lg (morph/pipeline
                  (morph/lift ds-mod/set-inference-target :species)
                  {:metamorph/id :model} (ml/model {:model-type :smile.classification/logistic-regression
                                                    :caching-predict-fn (fn [dataset model]
                                                                          (cache/caching-predict store dataset model))
                                                    :caching-train-fn (fn [dataset options]
                                                                        (cache/caching-train store dataset options))})))

(defn  pipe-fn-rf [trees] (morph/pipeline
                           (morph/lift ds-mod/set-inference-target :species)
                           {:metamorph/id :model} (ml/model {:model-type :smile.classification/random-forest
                                                             :trees trees
                                                             :caching-predict-fn (fn [dataset model]
                                                                                   (cache/caching-predict store dataset model))
                                                             :caching-train-fn (fn [dataset options]
                                                                                 (cache/caching-train store dataset options))})))
(def  pipe-fn-slow (morph/pipeline
                    (morph/lift ds-mod/set-inference-target :species)
                    {:metamorph/id :model} (ml/model {:model-type :slow-model
                                                      :very-slow? true
                                                      :caching-predict-fn (fn [dataset model]
                                                                           (cache/caching-predict store dataset model))
                                                      :caching-train-fn (fn [dataset options]
                                                                          (cache/caching-train store dataset options))})))
(def  evaluation-result
 (ml/evaluate-pipelines
  (concat

   (map pipe-fn-rf
        [10 50 100 150 200 500 750 1000])
   [pipe-fn-slow pipe-fn-lg pipe-fn-ada])
  splits
  loss/classification-accuracy

  :accuracy
  {}))

(println
 (-> evaluation-result flatten first :train-transform :mean)
 (-> evaluation-result flatten first :fit-ctx :model :train-result-wrapper :options))

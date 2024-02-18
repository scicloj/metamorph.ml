(ns exp
  (:require
   [clojure.core.cache.wrapped :as wcache]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.cache :as cache]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.ml.smile.classification]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [taoensso.nippy]
   [tech.v3.dataset.modelling :as ds-mod]))



(def iris
  (->
   (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
   (ds/categorical->number [:species])
   (ds-mod/set-inference-target :species)))





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


(def splits (tc/split->seq)
  (-> iris)
                 
  :kfold {:k 11
          :seed 12345})


(def wcache (wcache/fifo-cache-factory
             (cache/fs-persisted-map-factory "/tmp/store")
             {:threshold 1000}))

(def  pipe-fn-ada (morph/pipeline

                   {:metamorph/id :model} (ml/model {:model-type :smile.classification/ada-boost
                                                     :wcache wcache})))

                                                     

(def  pipe-fn-lg (morph/pipeline

                  {:metamorph/id :model} (ml/model {:model-type :smile.classification/logistic-regression
                                                    :wcache wcache})))
                                                    

(defn  pipe-fn-rf [trees] (morph/pipeline

                           {:metamorph/id :model} (ml/model {:model-type :smile.classification/random-forest
                                                             :trees trees
                                                             :wcache wcache})))

(def  pipe-fn-slow (morph/pipeline
                    {:metamorph/id :model} (ml/model {:model-type :slow-model
                                                      :very-slow? true
                                                      :wcache wcache})))

                                                      


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
 (-> evaluation-result flatten first :test-transform :mean)
 (-> evaluation-result flatten first :fit-ctx :model :model-wrapper :options))

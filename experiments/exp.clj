(ns exp
  (:require
   [clojure.core.cache.wrapped :as wcache]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   ;; [scicloj.metamorph.ml.cache :as cache]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.ml.smile.classification]

   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [taoensso.nippy]
   [tech.v3.dataset.modelling :as ds-mod])
  (:import
   [org.mlflow.tracking MlflowClient MlflowContext]))
    
   ;; [taoensso.carmine :as car :refer [wcar]]


;; (defonce my-conn-pool (car/connection-pool {}))
                                     ; Create a new stateful pool
;; (def     my-wcar-opts {:pool my-conn-pool})

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


(def splits (tc/split->seq
             iris
             :kfold
             {:k 5
              :seed 12345}))

(def wcache {})

;; (def wcache (wcache/basic-cache-factory
;;              {}))
             ;; (cache/redis-persisted-map-factory my-wcar-opts)


(def  pipe-fn-ada (morph/pipeline

                   {:metamorph/id :model} (ml/model {:model-type :smile.classification/ada-boost})))
                                                     ;; :wcache wcache


                                                     

(defn  pipe-fn-lg [opts] (morph/pipeline

                          {:metamorph/id :model} (ml/model
                                                  (assoc opts
                                                         :model-type :smile.classification/logistic-regression))))
                                                    ;; :wcache wcache

                                                    

(defn  pipe-fn-rf [trees] (morph/pipeline

                           {:metamorph/id :model} (ml/model {:model-type :smile.classification/random-forest
                                                             :trees trees})))
                                                             ;; :wcache wcache


(def  pipe-fn-slow (morph/pipeline
                    {:metamorph/id :model} (ml/model {:model-type :slow-model
                                                      :very-slow? true})))
                                                      ;; :wcache wcache


                                                      


(def  evaluation-result
  (ml/evaluate-pipelines
   (concat

    (map pipe-fn-lg
         (take 100
               (gs/sobol-gridsearch
                (ml/hyperparameters :smile.classification/logistic-regression))))



    []) ;; pipe-fn-slow
   ;; pipe-fn-rf

   ;; pipe-fn-ada

   splits
   loss/classification-accuracy

   :accuracy
   {:map-fn :pmap
    :mlflow (MlflowContext. "http://localhost:5000")}))
             ;;


   



(println
 (-> evaluation-result flatten first :test-transform :mean)
 (-> evaluation-result flatten first :fit-ctx :model :options))

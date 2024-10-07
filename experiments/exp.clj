(ns exp
  (:require
   [clojure.core.cache.wrapped :as wcache]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.cache :as cache]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.ml.smile.classification]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [taoensso.nippy]
   [camel-snake-kebab.core :as csk]
   [tech.v3.dataset.modelling :as ds-mod]
   [taoensso.carmine :as car :refer [wcar]]))

(add-tap println)
(defonce my-conn-pool (car/connection-pool {})) ; Create a new stateful pool
(def     my-wcar-opts {:pool my-conn-pool})


(def iris
  (->
   (tc/dataset "creditcard_2023.csv" {:key-fn csk/->kebab-case-keyword})
   (ds/categorical->number [:class])
   (tc/drop-columns [:id])
   (tc/drop-missing)
   (ds-mod/set-inference-target :class)
   (tc/shuffle {:seed 1234})
   (tc/head 100000))
  )

iris
;; (def iris
;;   (->
;;    (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
;;    (ds/categorical->number [:species])
;;    (ds-mod/set-inference-target :species)))


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
             {:k 4
              :seed 12345}))

(def wcache (wcache/basic-cache-factory
             (cache/redis-persisted-map-factory my-wcar-opts)))



(def pipe-fns-ada
  (->> 
   (gs/sobol-gridsearch
    (ml/hyperparameters :smile.classification/ada-boost))
   (take 50)
   (map #(morph/pipeline
          {:metamorph/id :model}
          (ml/model
           (merge 
            {:model-type :smile.classification/ada-boost
             :wcache wcache}
            %
            ))))))



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
(println :start)
                                                      
(def start (System/currentTimeMillis))

(def pipe-fns
     (concat
   (map pipe-fn-rf
        [10
             ;50 
         100
             ;150 
         200
             ;500 
         750
             ;1000
         ]) 
      pipe-fns-ada
   [;pipe-fn-lg
       ;pipe-fn-slow
    ]))

(println :n-models-expected 
         (* (count pipe-fns ) (count splits)))

(def  evaluation-result
  (ml/evaluate-pipelines
   pipe-fns
   splits
   loss/classification-accuracy
   :accuracy
   {}))

(println :n-models
         (-> evaluation-result flatten count))

(println
 :best-test-mean (-> evaluation-result flatten first :test-transform :mean) "\n"
 :best-train-mean (-> evaluation-result flatten first :train-transform :mean) "\n"
 :best-options (-> evaluation-result flatten first :fit-ctx :model :options))

(def end (System/currentTimeMillis))

(println :duration
       (/ (- end start) 1000.0)
       "s")
(ns cache 
  (:require
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.ml.smile.classification]
   [tablecloth.api :as tc]
   [tablecloth.pipeline :as tc-mm]
   [taoensso.carmine :as car]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as mds]
   [tech.v3.dataset.modelling :as ds-mod]))

(def titanic-train
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                 {:key-fn keyword})))

(def titanic-test
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/test.csv"
                 {:key-fn keyword})
   (tc/add-column :Survived 0)
   )
  )


(defn preprocess [ds]
  (-> ds
      (tc/select-columns [:Pclass :Survived :Embarked :Sex])
      (tc/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                              1 "yes"
                                                              0 "no")
                                                           (:Survived ds))))
      (ds/categorical->number [:Survived :Sex :Embarked])
      (tc/replace-missing)
      (ds-mod/set-inference-target :Survived)
      )
  
  )

(defn pipe-fn [options]
  (mm/pipeline
   (mds/select-columns [:Pclass :Survived :Embarked :Sex])
   

   (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                           1 "yes"
                                                           0 "no")
                                                        (:Survived ds))))
   (mds/categorical->number [:Survived :Sex :Embarked])
   (tc-mm/replace-missing )
   (mds/set-inference-target :Survived)

   (fn [ctx]
     (assoc ctx :options (dissoc options
                                 :cache-opts))
     )
   {:metamorph/id :model}
   (ml/model options)))

(defonce my-conn-pool (car/connection-pool {}))
(def     my-conn-spec {:uri "redis://localhost:6379"})
(def     my-wcar-opts {:pool my-conn-pool, :spec my-conn-spec})


(reset! ml/wcar-opts my-wcar-opts)

(defn pipe-fns [model-type hyper-params n]
  (->>
   (map
    #(pipe-fn
      (assoc %
             :model-type model-type))
    (gs/sobol-gridsearch hyper-params))
   (take n))
   )
(def n 10)
(def all-piep-fns
  (concat
   (pipe-fns :smile.classification/decision-tree
             {:max-nodes (gs/linear 10 1000 30 :int32)
              :node-size (gs/linear 1 20 20 :int32)
              :max-depth (gs/linear 1 50 20 :int32)
              :split-rule (gs/categorical [:gini :entropy :classification-error])}
             n)
   (pipe-fns :smile.classification/logistic-regression
             (ml/hyperparameters :smile.classification/logistic-regression)
             n)
  
   (pipe-fns :smile.classification/ada-boost
             (ml/hyperparameters :smile.classification/ada-boost)
             n)
   (pipe-fns :smile.classification/random-forest
             {:trees (gs/linear 10 1000 100 :int32)
              :max-depth (gs/linear 10 100 100 :int32)
              :max-nodes (gs/linear 10 100 100 :int32)
              :node-size (gs/linear 1 100 100 :int32)
              :sample-rate (gs/linear 0.1 1.0 100)}
             n))
)

(time
 
 (clojure.pprint/pprint
  (->
   (ml/evaluate-pipelines
    all-piep-fns
    [{:train 
      ;(preprocess) 
      titanic-train  
      :test titanic-test
      }]
    loss/classification-accuracy
    :accuracy)
   first
   first
   (#(hash-map :options (get-in % [:fit-ctx :options])
               :train-accuracy (get-in % [:train-transform :metric])
               :test-accuracy (get-in % [:test-transform :metric])))

   )))

                                 
 
 
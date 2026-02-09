(ns cache 
  (:require
   [clojure.pprint :as pp]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.evaluation-handler :as eval-handler]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.ml.smile.classification]
   [tablecloth.api :as tc]
   [tablecloth.pipeline :as tc-mm]
   [taoensso.carmine :as car]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as mds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tablecloth.column.api :as tcc]
   [clojure.java.io :as io]
   [taoensso.nippy :as nippy]
   ))

(def titanic-train
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                 {:key-fn keyword})))

(def titanic-test
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/test.csv"
                 {:key-fn keyword})
   (tc/add-column :Survived 0)))


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

  ;;  (fn [ctx]
  ;;    (assoc ctx :options (dissoc options
  ;;                                :cache-opts))
  ;;    )
   {:metamorph/id :model}
   (ml/model options)))

(defonce my-conn-pool (car/connection-pool {}))
(def     my-conn-spec {:uri "redis://localhost:6379"})
(def     my-wcar-opts {:pool my-conn-pool, :spec my-conn-spec})


;(ns-unmap *ns* 'cache-map)
(defonce cache-map (atom {}))


;; (reset! ml/kv-cache {:use-cache true
;;                      :get-fn (fn [key] (car/wcar my-wcar-opts (car/get key)))
;;                      :set-fn (fn [key value] (car/wcar my-wcar-opts (car/set key value)))})

(reset! ml/train-predict-cache {:use-cache true
                     :get-fn (fn [key] (get @cache-map key))
                     :set-fn (fn [key value] (swap! cache-map assoc key value))})

(reset! ml/train-predict-cache {:use-cache true
                                :get-fn (fn [key]
                                          (let [f (format "/tmp/cache/%s.cache" key)]
                                            (when (.exists  (io/file f))
                                              (nippy/thaw-from-file f))))
                                :set-fn (fn [key value]
                                          (nippy/freeze-to-file
                                           (format "/tmp/cache/%s.cache" key)
                                           value))})


(defn pipe-fns [model-type hyper-params n]
  (->>
   (map
    #(pipe-fn
      (assoc %
             :model-type model-type))
    (gs/sobol-gridsearch hyper-params))
   (take n))
   )
(def n 100)
(def all-piep-fns
  (concat
   (pipe-fns :smile.classification/decision-tree
             (ml/hyperparameters :smile.classification/decision-tree)
             n)
   (pipe-fns :smile.classification/logistic-regression
             (ml/hyperparameters :smile.classification/logistic-regression)
             n)
  
   (pipe-fns :smile.classification/ada-boost
             (ml/hyperparameters :smile.classification/ada-boost)
             n)
   (pipe-fns :smile.classification/random-forest
             (ml/hyperparameters :smile.classification/random-forest)

             n))
)

(time
 (def eval-result  
   (ml/evaluate-pipelines
    all-piep-fns
    [{:train
      titanic-train
      :test titanic-test}]
    loss/classification-accuracy
    :accuracy
    {:return-best-pipeline-only false
     :return-best-crossvalidation-only false
     ;;  :evaluation-handler-fn (fn [result]
     ;;                           (eval-handler/metrics-and-options-keep-fn result))
     
     })
   ))

(pp/pprint
 (-> eval-result

     first
     first
     (#(hash-map :options (get-in % [:fit-ctx :model :options])
                 :train-accuracy (get-in % [:train-transform :metric])
                 :test-accuracy (get-in % [:test-transform :metric])))))
(def datasets
  (map
   (fn [result]
     (tc/dataset
      (merge (-> result  :test-transform (select-keys [:metric]))
             (-> result :fit-ctx :model :options)))
     )
   (-> eval-result flatten)))

(def metrices
  (apply tc/concat datasets))

(-> metrices
    (tc/group-by :model-type)
    (tc/aggregate (fn [ds]
                    (tcc/mean (:metric ds)))))
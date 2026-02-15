(ns cache
  (:require
   [clojure.pprint :as pp]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.cache :as cache]
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
   [taoensso.nippy :as nippy]))

(def titanic-train
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                 {:key-fn keyword})))

(def titanic-test
  (->
   (ds/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/test.csv"
                 {:key-fn keyword})
   (tc/add-column :Survived 0)))
(-> xxx :metamorph/data)

(defn pipe-fn [options]
  (with-meta
    (mm/pipeline
     (fn [ctx]
       (assoc ctx :the-train-ds (:metamorph/data ctx)))


    (mds/select-columns [:Pclass :Survived :Embarked :Sex])


    (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                            1 "yes"
                                                            0 "no")
                                                         (:Survived ds))))
    (mds/categorical->number [:Survived :Sex :Embarked])
    (tc-mm/replace-missing)
    (mds/set-inference-target :Survived)

    ;;  (fn [ctx]
    ;;    (assoc ctx :options (dissoc options
    ;;                                :cache-opts))
    ;;    )
    {:metamorph/id :model}
    (ml/model options))
    options))

;(ns-unmap *ns* 'cache-map)


(defonce my-conn-pool (car/connection-pool {}))

(def wcar-opts {:pool my-conn-pool,
                :spec {:uri "redis://localhost:6379"}})


(defn pipe-fns [model-type hyper-params n]
  (->>
   (map
    #(pipe-fn
      (assoc %
             :model-type model-type))
    (gs/sobol-gridsearch hyper-params))
   (take n)))
(def n 10)
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

             n)))

(defn- eval-pipe []
  (ml/evaluate-pipelines
   all-piep-fns
   [{:train
     titanic-train
     :test titanic-test}]
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false
    :return-best-crossvalidation-only false
    :skip-scoring-fn (fn [train-ds pipeline-fn]
                       (let [exp-file (io/file (format "/tmp/exp-database/%s-%s.nippy" 
                                                       (hash train-ds) 
                                                       (hash (meta pipeline-fn))))]
                         (.exists exp-file)
                         )
                       
                       )
    :evaluation-handler-fn (fn [ctx]
                             (def ctx ctx 
                             )

                             (let [train-ds (-> ctx  :fit-ctx :train-ds)
                                   train-ds-hash (hash train-ds)
                                   train-options (-> ctx :fit-ctx :model :options)
                                   train-options-hash (hash train-options)
                                   metric (-> ctx :test-transform :metric)
                                   train-run-data {:train-ds train-ds
                                                   :train-options train-options
                                                   :metric metric}
                                   train-result-file-name (format "/tmp/exp-database/%s-%s.nippy" train-ds-hash train-options-hash)]

                               (when (not (.exists (io/file train-result-file-name)))
                                 (println :write-to-db train-result-file-name)
                                 (nippy/freeze-to-file train-result-file-name train-run-data)))
                             (-> (eval-handler/metrics-and-options-keep-fn ctx)))}))



(time
 (do
   (cache/disable-cache!)
   (time (let [_ (eval-pipe)]))))

(do
  (let [cache-map (atom {})]
    (cache/enable-atom-cache! cache-map)
    (eval-pipe))
  (time (let [_ (eval-pipe)])))

(do
  (cache/enable-redis-cache! wcar-opts)
  (let [_ (eval-pipe)])
  (time (let [_ (eval-pipe)])))


(let [cache-dir "/tmp/cache"]
  (when (.exists (io/as-file cache-dir))
    (run! io/delete-file (reverse (file-seq (io/as-file cache-dir)))))
  (.mkdirs (io/as-file cache-dir))

  (cache/enable-disk-cache! cache-dir)
  (let [_ (eval-pipe)])
  (time (let [_ (eval-pipe)])))


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
             (-> result :fit-ctx :model :options))))
   (-> eval-result flatten)))

(def metrices
  (apply tc/concat datasets))

(-> metrices
    (tc/group-by :model-type)
    (tc/aggregate (fn [ds]
                    (tcc/mean (:metric ds)))))
(ns cache
  (:require
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.cache :as cache]
   [scicloj.metamorph.ml.gridsearch :as gs]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.ml.smile.classification]
   [tablecloth.api :as tc]
   [tablecloth.pipeline :as tc-mm]
   [taoensso.carmine :as car]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as mds]
   [taoensso.carmine :as car]
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

(defn enable-redis-cache!
  "Enables the caching of train/predict calls in redis/carmine.
   
   `wcar-opts`: Options for Carmine
   
   See the [Carmine](https://github.com/taoensso/carmine) documentation for the setup.

   It requires adding of `com.taoensso/carmine` to classpath
   "
  [wcar-opts]
  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key]
                                            (require 'taoensso.carmine)
                                            (taoensso.carmine/wcar
                                             wcar-opts
                                             (taoensso.carmine/get key)))

                                  :set-fn (fn [key value]
                                            (require 'taoensso.carmine)
                                            (taoensso.carmine/wcar
                                             wcar-opts
                                             (taoensso.carmine/set key value)))}))



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
    :return-best-crossvalidation-only false}))


(time
 (do
   (cache/disable-cache!)
   (time (let [_ (eval-pipe)]))))


(let [cache-map (atom {})]
  (cache/enable-atom-cache! cache-map)
  (eval-pipe)
  (time (let [_ (eval-pipe)])))

(do
  (enable-redis-cache! wcar-opts)
  (eval-pipe)
  (time (let [_ (eval-pipe)])))


(let [cache-dir "/tmp/cache"]
  (when (.exists (io/as-file cache-dir))
    (run! io/delete-file (reverse (file-seq (io/as-file cache-dir)))))
  (.mkdirs (io/as-file cache-dir))

  (cache/enable-disk-cache! cache-dir)
  (let [_ (eval-pipe)])
  (time (let [_ (eval-pipe)])))



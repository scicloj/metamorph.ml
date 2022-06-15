(ns scicloj.metamorph.ensemble-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.ensemble :as ensemble]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.metamorph :as ds-mm]))

(ml/define-model! :test-model
  (fn train
    [feature-ds label-ds options]
    {:model-data {:model-as-bytes [1 2 3]
                  :smile-df-used [:blub]}})
  (fn predict
    [feature-ds thawed-model {:keys [target-columns
                                     target-categorical-maps
                                     top-k
                                     options]}]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) 1)
                                    {:column-type :prediction})]))
  {:explain-fn (fn  [thawed-model {:keys [feature-columns]} _options]
                 {:coefficients {:petal_width [0]}})})


(ml/define-model! :test-model-2
  (fn train
    [feature-ds label-ds options]
    {:model-data {:model-as-bytes [1 2 3]
                  :smile-df-used [:blub]}})
  (fn predict
    [feature-ds thawed-model {:keys [target-columns
                                     target-categorical-maps
                                     top-k
                                     options]}]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) 0)
                                    {:column-type :prediction})]))
  {:explain-fn (fn  [thawed-model {:keys [feature-columns]} _options]
                 {:coefficients {:petal_width [0]}})})

(def iris (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))
(def pipe-1
  (morph/pipeline
   (ds-mm/set-inference-target :species)
   (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) {} :int)
   (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

   {:metamorph/id :model}
   (ml/model {:model-type :test-model})))

(def pipe-2
  (morph/pipeline
   (ds-mm/set-inference-target :species)
   (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) {} :int)
   (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

   {:metamorph/id :model}
   (ml/model {:model-type :test-model-2})))



(deftest test-ensemble
  (let [
        ensemble-pipe (ensemble/ensemble-pipe [pipe-1 pipe-1 pipe-2])

        fit-ctx
        (morph/fit-pipe iris ensemble-pipe)

        transformed-ctx
        (morph/transform-pipe iris ensemble-pipe fit-ctx)]
    (is (= {1 150}
           (-> transformed-ctx :metamorph/data :prediction frequencies)))))

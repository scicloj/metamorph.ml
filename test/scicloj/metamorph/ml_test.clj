(ns scicloj.metamorph.ml-test
  (:require  [clojure.test :refer [is deftest]]
             [scicloj.metamorph.core :as morph]
             [tech.v3.libs.smile.classification]
             [tech.v3.dataset.metamorph :as ds-mm]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.modelling :as ds-mod]
             [tech.v3.dataset.column-filters :as cf]
             [tech.v3.ml.metamorph :as ml-mm]
             [tech.v3.ml.gridsearch :as gs]
             [scicloj.metamorph.ml :as ml-eval]
             [tech.v3.ml.loss :as loss]
             [tech.v3.datatype.functional :as dtf]
             [tech.v3.ml.gridsearch :as ml-gs]
             [tablecloth.api.split :as split]
             [tablecloth.api :as tc]
             ))





(deftest evaluate-pipelines-simplest
  (let [
        ;;  the data
        ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        ;;  the (single, fixed) pipe-fn
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
         (ml-mm/model {:model-type :smile.classification/random-forest}))

        ;;  the simplest split
        train-split-seq (split/split ds :holdout)

        ;; one pipe-fn in the seq
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml-eval/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss)


        ;; we have only one result
        best-fitted-context  (-> evaluations first :fitted-ctx)
        best-pipe-fn         (-> evaluations first :pipe-fn)


        ;;  simulate new data
        new-ds (ds/sample ds 10 {:seed 1234} )

        ;;  do prediction on new data
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species))]

    (is (= ["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]
         predictions))))



(deftest grid-search
  (let [
        ds(ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        grid-search-options
        {:trees (gs/categorical [10 50 100 500])
         :split-rule (gs/categorical [:gini :entropy])
         :model-type :smile.classification/random-forest}

        create-pipe-fn (fn[options]
                         (morph/pipeline
                          (ds-mm/set-inference-target :species)
                          (ds-mm/categorical->number cf/categorical)
                          (fn [ctx]
                            (assoc ctx :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
                          (ml-mm/model options)))

        all-options (gs/sobol-gridsearch grid-search-options)
        pipe-fn-seq (map create-pipe-fn all-options)
        train-test-seq (split/split ds :kfold {:k 10})
        evaluations (ml-eval/evaluate-pipelines pipe-fn-seq train-test-seq loss/classification-loss)

        evalution-with-lowest-avg-loss
        (->>
         (group-by :pipe-fn evaluations)
         vals
         (map first)
         (sort-by :avg-loss)
         (first))

        new-ds (ds/sample ds 10 {:seed 1234} )

        predictions
        (->   ((evalution-with-lowest-avg-loss :pipe-fn)
               (merge (evalution-with-lowest-avg-loss :fitted-ctx)
                      {:metamorph/data new-ds
                       :metamorph/mode :transform}))
              (:metamorph/data)
              (ds-mod/column-values->categorical :species)
              seq)]

    (is (= ["versicolor"
            "versicolor"
            "virginica"
            "versicolor"
            "virginica"
            "setosa"
            "virginica"
            "virginica"
            "versicolor"
            "versicolor"]
           predictions))))

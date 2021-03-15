(ns scicloj.metamorph.ml-test
  (:require [clojure.test :refer [deftest is]]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml-eval]
            [tablecloth.api.split :as split]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.metamorph.ml.gridsearch :as gs]

            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.libs.smile.classification]
            [scicloj.metamorph.ml.metamorph :as ml-mm]))



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
        train-split-seq (split/split->seq ds :holdout)

        ;; one pipe-fn in the seq
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml-eval/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)


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

(deftest evaluate-pipelines-without-model
  (let [;;  the data
        ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx)))))
        train-split-seq (split/split->seq ds :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations (ml-eval/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)
        best-fitted-context  (-> evaluations first :fitted-ctx)
        best-pipe-fn         (-> evaluations first :pipe-fn)


        new-ds (-> (ds/sample ds 3 {:seed 1234} )
                            (ds/add-or-update-column (ds/new-column :species (repeat 3  "setosa" ) {:categorical? true}) )
                            (ds-mod/set-inference-target :species)
                            )
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species)
         )]

    (is (= ["setosa" "setosa" "setosa" ]
           predictions))
    )
  )



(deftest grid-search
  (let [
        ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        grid-search-options
        {:trees (gs/categorical [10 50 100 500])
         :split-rule (gs/categorical [:gini :entropy])
         :model-type :smile.classification/random-forest}

        create-pipe-fn
        (fn[options]
          (morph/pipeline
           (ds-mm/set-inference-target :species)
           (ds-mm/categorical->number cf/categorical)
           (fn [ctx]
             (assoc ctx :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
           (ml-mm/model options)))

        all-options-combinations (gs/sobol-gridsearch grid-search-options)

        pipe-fn-seq (map create-pipe-fn (take 7 all-options-combinations))

        train-test-seq (split/split->seq ds :kfold {:k 10})

        evaluations
        (ml-eval/evaluate-pipelines pipe-fn-seq train-test-seq loss/classification-loss :loss)


        new-ds (-> (ds/sample ds 10 {:seed 1234} )
                   (ds/add-or-update-column (ds/new-column :species (repeat 10  "setosa" ) {:categorical? true}) )
                   )

        predictions
        (ml-eval/predict-on-best-model evaluations new-ds :loss)]

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

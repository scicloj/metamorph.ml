(ns scicloj.metamorph.ml-test
  (:require [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.dataset :as ds-ml]
            [scicloj.metamorph.ml.dataset-metamorph :as ds-ml-mm]
            [scicloj.metamorph.ml.gridsearch :as gs]
            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.libs.smile.classification]
            [tablecloth.api :as tc]))

(deftest evaluate-pipelines-simplest
  (let [
        ;;  the data
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        ;;  the (single, fixed) pipe-fn
        pipe-fn
        (ml/pipeline
         (ds-ml-mm/set-inference-target :species)
         (ds-ml-mm/categorical->number ds-ml/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (ds-ml/target (:metamorph/data ctx))))
         (ml/model {:model-type :smile.classification/random-forest}))

        ;;  the simplest split
        train-split-seq (tc/split->seq ds :holdout)

        ;; one pipe-fn in the seq
        pipe-fn-seq [pipe-fn]


        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)

        ;; we have only one result
        best-fitted-context  (-> evaluations first :fitted-ctx)
        best-pipe-fn         (-> evaluations first :pipe-fn)


        ;;  simulate new data
        new-ds (->
                (tc/shuffle ds  {:seed 1234} )
                (tc/head 10)
                )
        ;;  do prediction on new data
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-ml/column-values->categorical :species))]

    (is (= ["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]
           predictions))))

(deftest evaluate-pipelines-without-model
  (let [;;  the data
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (ml/pipeline
         (ds-ml-mm/set-inference-target :species)
         (ds-ml-mm/categorical->number ds-ml/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (ds-ml/target (:metamorph/data ctx)))))
        train-split-seq (tc/split->seq ds :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)
        best-fitted-context  (-> evaluations first :fitted-ctx)
        best-pipe-fn         (-> evaluations first :pipe-fn)

        _ (def ds ds)
        new-ds (->
                (tc/shuffle ds  {:seed 1234} )
                (tc/head 3)
                )
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-ml/column-values->categorical :species)
         )]

    (is (= ["versicolor" "versicolor" "virginica" ]
           predictions))))



(deftest grid-search
  (let [
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        grid-search-options
        {:trees (gs/categorical [10 50 100 500])
         :split-rule (gs/categorical [:gini :entropy])
         :model-type :smile.classification/random-forest}

        create-pipe-fn
        (fn[options]
          (ml/pipeline
           (ds-ml-mm/set-inference-target :species)
           (ds-ml-mm/categorical->number ds-ml/categorical)
           (fn [ctx]
             (assoc ctx :scicloj.metamorph.ml/target-ds (ds-ml/target (:metamorph/data ctx))))
           (ml/model options)))

        all-options-combinations (gs/sobol-gridsearch grid-search-options)

        pipe-fn-seq (map create-pipe-fn (take 7 all-options-combinations))

        train-test-seq (tc/split->seq ds :kfold {:k 10})

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-test-seq loss/classification-loss :loss)

        new-ds (->
                (tc/shuffle ds  {:seed 1234} )
                (tc/head 10)
                )

        predictions
        (ml/predict-on-best-model evaluations new-ds :loss)]

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


(deftest test-model
  (let [
        src-ds (tc/dataset "test/data/iris.csv")
        ds (->  src-ds
                (ds-ml/categorical->number ds-ml/categorical)
                (ds-ml/set-inference-target "species")

                (tc/shuffle {:seed 1234}))
        feature-ds (ds-ml/feature ds)
        split-data (first (tc/split->seq ds :holdout {:seed 1234}))
        train-ds (:train split-data)
        test-ds  (:test split-data)

        pipeline (fn  [ctx]
                   ((ml/model {:model-type :smile.classification/random-forest})
                    ctx))


        fitted
        (pipeline
         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/data train-ds})


        prediction
        (pipeline (merge fitted
                         {:metamorph/mode :transform
                          :metamorph/data test-ds}))

        predicted-species (ds-ml/column-values->categorical (:metamorph/data prediction)
                                                            "species"
                                                            )]

    (is (= ["versicolor" "virginica" "versicolor"]
           (take 3 predicted-species)))))

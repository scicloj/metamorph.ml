(ns scicloj.metamorph.ml-test
  (:require [clojure.test :refer [deftest is]]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.gridsearch :as gs]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.ml.smile.classification]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tablecloth.api :as tc]))

(deftest evaluate-pipelines-simplest
  (let [

        ;;  the data
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

        ;;  the (single, fixed) pipe-fn
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
         (ml/model {:model-type :smile.classification/random-forest}))

        ;;  the simplest split
        train-split-seq (tc/split->seq ds :holdout)

        ;; one pipe-fn in the seq
        pipe-fn-seq [pipe-fn]


        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)

        ;; we have only one result
        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)


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
         (ds-mod/column-values->categorical :species))]

    (is (= ["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]
           predictions))
    (is (=  1) (count evaluations))
    (is (=  1) (count (first evaluations)))

    (is (= [:fit-ctx :transform-ctx :metric :metric-fn :pipe-fn :min :mean :max] (keys (first (first evaluations)))))
    (is (contains?   (:fit-ctx (first (first evaluations)))  :metamorph/mode))
    (is (contains?   (:transform-ctx (first (first evaluations)))  :metamorph/mode))


    
    ))



(deftest evaluate-pipelines-without-model
  (let [;;  the data
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx)))))
        train-split-seq (tc/split->seq ds :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)
        best-fitted-context  (-> evaluations first first :fitted-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)

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
         (ds-mod/column-values->categorical :species)
         )]

    (is (= ["versicolor" "versicolor" "virginica" ]
           predictions))))



(deftest grid-search
  (let [
        ds (->
            (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
            (ds-mod/set-inference-target :species)
            )

        grid-search-options
        {:trees (gs/categorical [10 50 100 500])
         :split-rule (gs/categorical [:gini :entropy])
         :model-type :smile.classification/random-forest}

        create-pipe-fn
        (fn[options]
          (morph/pipeline
           ;; (ds-mm/set-inference-target :species)
           (ds-mm/categorical->number cf/categorical)
           (fn [ctx]
             (assoc ctx :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
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
        (ml/predict-on-best-model (flatten evaluations) new-ds :loss)]

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
                (ds/categorical->number cf/categorical)
                (ds-mod/set-inference-target "species")

                (tc/shuffle {:seed 1234}))
        feature-ds (cf/feature ds)
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

        predicted-species (ds-mod/column-values->categorical (:metamorph/data prediction)
                                                            "species"
                                                            )]

    (is (= ["setosa" "versicolor" "versicolor"]
           (take 3 predicted-species)))))



(comment



  ;;  the data
  (def  ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

  ;;  the (single, fixed) pipe-fn
  (def  pipe-fn
    (morph/pipeline
     (ds-mm/set-inference-target :species)
     (ds-mm/categorical->number cf/categorical)
     ;; (fn [ctx]
     ;;   (assoc ctx
     ;;          :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
     (ml/model {:model-type :smile.classification/random-forest})))

  ;;  the simplest split
  (def  train-split-seq (tc/split->seq ds :kfold))

  ;; one pipe-fn in the seq
  (def pipe-fn-seq [pipe-fn pipe-fn])


  (def evaluations
    (ml/evaluate-pipelines
     pipe-fn-seq train-split-seq loss/classification-loss :loss
     :result-dissoc-seq []
     :return-best-crossvalidation-only
      true
      :return-best-pipeline-only true)
    )
  (first (first evaluations))
  ;; we have only one result
  (def best-fitted-context (-> evaluations first first :fit-ctx))
  (def best-pipe-fn (-> evaluations first first :pipe-fn))


  ;;  simulate new data
  (def new-ds (->
               (tc/shuffle ds  {:seed 1234} )
               (tc/head 10)
               ))
  ;;  do prediction on new data
  predictions
  (->
   (best-pipe-fn
    (merge best-fitted-context
           {:metamorph/data new-ds
            :metamorph/mode :transform}))
   (:metamorph/data)
   (ds-mod/column-values->categorical :species))

  (is (= ["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]
         predictions)))

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
            [tablecloth.api :as tc]
            [scicloj.ml.smile.classification]
            [fastmath.stats :as stats]))

(deftest evaluate-pipelines-simplest
  (let [
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
         (ml/model {:model-type :smile.classification/random-forest}))

        train-split-seq (tc/split->seq ds :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)

        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)


        new-ds (->
                (tc/shuffle ds  {:seed 1234})
                (tc/head 10))
                
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species))]

    (is (= ["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor"]
           predictions))
    (is (=  1 (count evaluations)))
    (is (=  1 (count (first evaluations))))

    (is (= #{:min :mean :max :timing :ctx :metric}
           (set (-> evaluations first first :train-transform keys))))
    ;; =>
    (is (= (set [:fit-ctx :test-transform :train-transform :pipe-fn :metric-fn]) (set (keys (first (first evaluations))))))
    (is (contains?   (:fit-ctx (first (first evaluations)))  :metamorph/mode))
    (is (contains?   (:ctx (:train-transform (first (first evaluations))))  :metamorph/mode))
    (is (contains?   (:ctx (:test-transform (first (first evaluations))))  :metamorph/mode))))



    

(deftest evaluate-pipelines-several-cross
  (let [
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
         (ml/model {:model-type :smile.classification/random-forest}))

        train-split-seq (tc/split->seq ds :kfold)
        pipe-fn-seq [pipe-fn pipe-fn]

        evaluations-1
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss
                               {:return-best-crossvalidation-only false})
        evaluations-2
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss
                               {:return-best-crossvalidation-only false
                                :return-best-pipeline-only false})
                                
        evaluations-3
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss
                               {:return-best-pipeline-only false})]

        

    (def evaluations-2 evaluations-2)

    (is (= 5 (count (first evaluations-1))))
    (is (= 1 (count evaluations-1)))

    (is (= 5 (count (first evaluations-2))))
    (is (= 2 (count evaluations-2)))

;    (distin ) (map :max (first evaluations-2))


    (is (= 1 (count (first evaluations-3))))
    (is (= 2 (count evaluations-3)))))
    

  



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
        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)

        new-ds (->
                (tc/shuffle ds  {:seed 1234})
                (tc/head 3))
                
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species))]
         

    (is (= ["versicolor" "versicolor" "virginica"]
           predictions))))



(deftest grid-search
  (let [
        ds (->
            (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
            (ds-mod/set-inference-target :species))
            

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
                (tc/shuffle ds  {:seed 1234})
                (tc/head 10))
                
        _ (def evaluations evaluations)

        best-pipe-fn         (-> evaluations first first :pipe-fn)

        best-fitted-context  (-> evaluations first first :fit-ctx)

        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species))]
         
        ;; (ml/predict-on-best-model (flatten evaluations) new-ds :loss)
        

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
                                                            "species")]
                                                            

    (is (= ["setosa" "versicolor" "versicolor"]
           (take 3 predicted-species)))))


(comment
  (def  ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))
  (def  pipe-fn
    (morph/pipeline
     (ds-mm/set-inference-target :species)
     (ds-mm/categorical->number cf/categorical)

     (ml/model {:model-type :smile.classification/random-forest})))

  (def train-split-seq (tc/split->seq ds :holdout {:repeats 1000}))
  (def pipe-fn-seq [pipe-fn])

  (def  evaluations
    (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-accuracy :accuracy
                           {:result-dissoc-in-seq []
                            :return-best-pipeline-only false
                            :return-best-crossvalidation-only false}))

 (require '[clj-memory-meter.core :as mm])

 (fastmath.stats/stats-map
  (map
   #(-> % :train-transform :metric) 
   (-> evaluations flatten)))
 :ok)


(def single-ds (tech.v3.dataset/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))
(mm/measure single-ds);; => "9.0 KiB"


(defn create-lots-of-ds [n]
  (let [ds (tech.v3.dataset/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        copies (mapv #(tech.v3.dataset/categorical->number % tech.v3.dataset.column-filters/categorical)
                     (repeat n ds))]
    (mm/measure copies)))

(count lots-of-ds);; => 1
(create-lots-of-ds 1);; => "10.2 KiB"
(create-lots-of-ds 10);; => "14.4 KiB"
(create-lots-of-ds 100);; => "56.4 KiB"
(create-lots-of-ds 1000);; => "475.8 KiB"
(create-lots-of-ds 10000);; => "4.6 MiB"
(create-lots-of-ds 100000);; => "45.5 MiB"
(create-lots-of-ds 1000000);; => "455.3 MiB"
;;
;;
;;



(measure 1 false);; => "247.1 KiB"
(measure 10 false);; => "2.4 MiB"
(measure 100 false);; => "24.2 MiB"
(measure 1000 false);; => "240.9 MiB"

(measure 1 true);; => "262.0 KiB"
(measure 10 true);; => "308.8 KiB"
(measure 100 true);; => "777.9 KiB"
(measure 1000 true);; => "5.4 MiB"
(measure 10000 true)


(for [i (range 10)]
  (do
    (System/gc)
    (print
     (mm/measure (keys (first  evaluations)))
     " : ")
    (println (count evaluations))
    (Thread/sleep 10000)))

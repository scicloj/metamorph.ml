(ns scicloj.metamorph.ml-test
  (:require [clojure.test :refer [deftest is] :as t]
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
            [fastmath.stats :as stats]
            [taoensso.nippy :as nippy]
            [confuse.multi-class-metrics :as mcm]
            [scicloj.metamorph.ml.metrics]
            [scicloj.metamorph.ml.evaluation-handler :as eval]
            [scicloj.metamorph.ml.evaluation-handler :refer [get-source-information qualify-pipelines qualify-keywords]])


  (:import (java.util UUID) (java.io File)))



(deftest evaluate-pipelines-simplest
  (let [
        ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
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

    (is (= #{:min :mean :max :timing :ctx :metric :other-metrices}
           (set (-> evaluations first first :train-transform keys))))
    ;; =>
    (is (= (set [:fit-ctx :test-transform :train-transform :pipe-fn :pipe-decl :metric-fn :timing-fit :loss-or-accuracy :source-information])
           (set (keys (first (first evaluations))))))
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

        


    (is (= 5 (count (first evaluations-1))))
    (is (= 1 (count evaluations-1)))

    (is (= 5 (count (first evaluations-2))))
    (is (= 2 (count evaluations-2)))

                                        ;    (distin ) (map :max (first evaluations-2))


    (is (= 1 (count (first evaluations-3))))
    (is (= 2 (count evaluations-3)))))
    





(deftest evaluate-pipelines-without-model
  (is (thrown? Exception
               (let [ ;;  the data
                     ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
                     pipe-fn
                     (morph/pipeline
                      (ds-mm/set-inference-target :species)
                      (ds-mm/categorical->number cf/categorical))
         
                     train-split-seq (tc/split->seq ds :holdout)
                     pipe-fn-seq [pipe-fn]]

                 (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)))))



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
           (ds-mm/categorical->number cf/categorical)
           (ml/model options)))

        all-options-combinations (gs/sobol-gridsearch grid-search-options)

        pipe-fn-seq (map create-pipe-fn (take 7 all-options-combinations))

        train-test-seq (tc/split->seq ds :kfold {:k 10})

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-test-seq loss/classification-loss :loss)

        new-ds (->
                (tc/shuffle ds  {:seed 1234})
                (tc/head 10))
                

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

(defn do-xxx [col] col)

(deftest qualify-pipelines-test
  (is (= (repeat 3 :scicloj.metamorph.ml-test/do-xxx)
         (qualify-pipelines [ ;; 'do-xxx
                             ::do-xxx
                             'scicloj.metamorph.ml-test/do-xxx
                             :scicloj.metamorph.ml-test/do-xxx]
                           (find-ns 'scicloj.metamorph.ml-test)))))


(defn fit-pipe-in-new-ns [file ds]
  (let [new-ns (create-ns (symbol (str (UUID/randomUUID))))
        _ (intern new-ns 'file file)
        _ (intern new-ns 'ds ds)
        _ (.addAlias new-ns 'morph (the-ns 'scicloj.metamorph.core))
        _ (.addAlias new-ns 'nippy (the-ns 'taoensso.nippy))
        species-freqs (binding [*ns* new-ns]  (do
                                                (eval '(def thawed-result (nippy/thaw-from-file file)))


                                                (eval '(def thawed-pipe-fn (clojure.core/->
                                                                            thawed-result
                                                                            :pipe-decl
                                                                            (morph/->pipeline))))
                                                (eval '(clojure.core/->
                                                        (morph/fit-pipe ds thawed-pipe-fn)
                                                        :metamorph/data
                                                        :species
                                                        (clojure.core/frequencies)))))]
    species-freqs))


(deftest round-trip-full-names
  (is (= {1.0 50, 0.0 50, 2.0 50}

         (let [ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
               base-pipe-declr

               [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
                [:tech.v3.dataset.metamorph/categorical->number [:species]]
                [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
                [:scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest}]]
               files (atom [])

               nippy-handler (eval/nippy-handler files
                                                 "/tmp"
                                                 "/home/carsten/Dropbox/sources/metamorph.ml/test/scicloj/metamorph/ml_test.clj"
                                                 *ns*)
               eval-result (ml/evaluate-pipelines
                            [base-pipe-declr]
                            (tc/split->seq ds)
                            loss/classification-accuracy
                            :accuracy
                            {:evaluation-handler-fn nippy-handler})]
           (fit-pipe-in-new-ns (first @files) ds)))))


(deftest round-trip-aliased-names
  (is (= {1.0 50, 0.0 50, 2.0 50}

         (let [ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})

               base-pipe-declr
               (qualify-pipelines
                [
                 [[:ds-mm/set-inference-target [:species]]
                  [:ds-mm/categorical->number [:species]]
                  [:ds-mm/update-column :species ::do-xxx]
                  [:ds-mm/update-column :species :clojure.core/identity]
                  [:ml/model {:model-type :smile.classification/random-forest}]]]
                (find-ns 'scicloj.metamorph.ml-test))


               files (atom [])
               nippy-handler (eval/nippy-handler files
                                                 "/tmp"
                                                 "/home/carsten/Dropbox/sources/metamorph.ml/test/scicloj/metamorph/ml_test.clj"
                                                 *ns*)
                              

               eval-result (ml/evaluate-pipelines
                            base-pipe-declr
                            (tc/split->seq ds)
                            loss/classification-accuracy
                            :accuracy
                            {:evaluation-handler-fn nippy-handler})]

           (fit-pipe-in-new-ns (first @files) ds)))))

(deftest remove-all
  (let [ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species]]
         [:scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq ds)
         loss/classification-accuracy
         :accuracy
         {:result-dissoc-in-seq ml/result-dissoc-in-seq--all})]

    (is (pos? (-> evaluation-result first first :train-transform :timing)))))


(deftest other-metrices
  (let [ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species]]
         [:scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq ds)
         loss/classification-accuracy
         :accuracy
         { ;; :result-dissoc-in-seq ml/result-dissoc-in-seq--all
          :other-metrices [{:name :acc-2  :metric-fn loss/classification-accuracy}
                           {:name :fscore :metric-fn (fn [truth prediction] (mcm/macro-avg-fmeasure (vec truth) (vec prediction)))}
                           {:name :fpr    :metric-fn scicloj.metamorph.ml.metrics/fnr}]})]
    (is (pos? (-> evaluation-result first first :train-transform :other-metrices first :metric)))
    (is (pos? (-> evaluation-result first first :train-transform :other-metrices second :metric)))
    (is (some? (-> evaluation-result first first :train-transform :other-metrices (nth 2) :metric)))))

    ;; evaluation-result

    ;; evaluation-result

    ;; (is (pos? (-> evaluation-result first first :train-transform :timing)))



(comment

  (def ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))


  (def pipe-fn
    (morph/pipeline
     (ds-mm/set-inference-target :species)
     (ds-mm/categorical->number cf/categorical)
     (ml/model {:model-type :smile.classification/random-forest})))


  (def train-split-seq (tc/split->seq ds :holdout))
  (def pipe-fn-seq [pipe-fn])





  (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-accuracy :accuracy
                         {:other-metrices [{:name :acc-2
                                            :metric-fn loss/classification-accuracy}

                                           {:name :fpr
                                            :metric-fn scicloj.metamorph.ml.metrics/fnr}]})

  (vec
   (float-array [1 2]))
  (require '[scicloj.metamorph.ml.metrics])

  (scicloj.metamorph.ml.metrics/fnr [:1 :2] [:1 :3])


  (morph/fit-pipe ds
                  (-> base-pipe-declr
                      morph/->pipeline))


  (def pipe-decls
    (qualify-pipelines
     base-pipe-declrss
     *ns*))

  (require '[malli.core :as m])
  (require '[malli.error :as me])





  (->


   (m/validate (-> #'ml/evaluate-pipelines meta :malli/schema (nth 2)))


   (me/humanize))



  (require '[malli.dot :as md])

  (->>
   (-> #'ml/evaluate-pipelines meta :malli/schema (nth 2))

   (md/transform)
   (spit "/tmp/ev.dot"))

  ;; me/humanize

  (defn pp-str [x]
    (with-out-str (clojure.pprint/pprint x)))


  (def res (nippy/thaw-from-file "/tmp/be5145de-4afa-4b7d-a12c-441c7f7dbef6.nippy"))




  (spit "/tmp/res.txt"
        (pp-str
         (ml/dissoc-in res [:fit-ctx #uuid "e2663a08-24d8-42b2-9c01-13f673c90456" :model-data])))




          

  :ok)

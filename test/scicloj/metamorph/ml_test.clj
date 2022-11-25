(ns scicloj.metamorph.ml-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [confuse.multi-class-metrics :as mcm]
   [malli.core :as m]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.evaluation-handler
    :as eval
    :refer [qualify-pipelines]]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.metrics]
   [tablecloth.api :as tc]
   [taoensso.nippy :as nippy]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.metamorph :as ds-mm]
   [tech.v3.dataset.modelling :as ds-mod])
  (:import
   (clojure.lang ExceptionInfo)
   (java.util UUID)))

(def iris (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))


(defn keys-in
  "Returns a sequence of all key paths in a given map using DFS walk."
  [m]
  (letfn [(children [node]
            (let [v (get-in m node)]
              (if (map? v)
                (map (fn [x] (conj node x)) (keys v))
                [])))
          (branch? [node] (-> (children node) seq boolean))]
    (->> (keys m)
         (map vector)
         (mapcat #(tree-seq branch? children %)))))
(defn do-define-model []
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
                                      (repeat (tc/row-count feature-ds) "setosa")
                                      {:column-type :prediction})]))
    {:explain-fn (fn  [thawed-model {:keys [feature-columns]} _options]
                   {:coefficients {:petal_width [0]}})}))




(deftest evaluate-pipelines-simplest
  (do-define-model)
  (let [

        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) {} :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss {:evaluation-handler-fn identity})


        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)

        new-ds (->
                (tc/shuffle iris  {:seed 1234})
                (tc/head 10))

        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data))]
    ;; (ds-mod/column-values->categorical :species)



    (is (= ["setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa" "setosa"]
           (:species predictions)))
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




(deftest test-explain
  (let [

        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) {} :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss {:evaluation-handler-fn identity})
        _ (def evaluations evaluations)

        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)]


    (is (= :petal_width (-> best-fitted-context :model (ml/explain) :coefficients first first)))))



(deftest test-data-removed
  (do-define-model)
  (let [

        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) {} :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)]


    (is (nil? (-> evaluations first first :train-transform :ctx :model :model-data :model-as-bytes)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :model-data :smile-df-used)))
    (is (nil? (-> evaluations first first :test-transform :ctx :model :model-data :model-as-bytes)))
    (is (nil? (-> evaluations first first :test-transform :ctx :model :model-data :smile-df-used)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :scicloj.metamorph.ml/target-ds)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :scicloj.metamorph.ml/feature-ds)))))




    

(deftest evaluate-pipelines-several-cross
  (do-define-model)
  (let [

        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)

         {:metamorph/id :model}(ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :kfold)
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

                     pipe-fn
                     (morph/pipeline
                      (ds-mm/set-inference-target :species)
                      (ds-mm/categorical->number cf/categorical))
         
                     train-split-seq (tc/split->seq iris :holdout)
                     pipe-fn-seq [pipe-fn]]

                 (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)))))









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
  (do-define-model)
  (is (= {1.0 50, 0.0 50, 2.0 50}

         (let [files (atom [])

               base-pipe-declr

               [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
                [:tech.v3.dataset.metamorph/categorical->number [:species]]
                [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
                {:metamorph/id :model}[:scicloj.metamorph.ml/model {:model-type :test-model}]]
               files (atom [])

               nippy-handler (eval/example-nippy-handler files "/tmp" identity)


                              
               eval-result (ml/evaluate-pipelines
                            [base-pipe-declr]
                            (tc/split->seq iris)
                            loss/classification-accuracy
                            :accuracy
                            {:evaluation-handler-fn nippy-handler})]
           (fit-pipe-in-new-ns (first @files) iris)))))




(deftest dissoc--all-fn
  (do-define-model)
  (let [
        base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species]]
         {:metamorph/id :model}[:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris)
         loss/classification-accuracy
         :accuracy
         {:evaluation-handler-fn ml/result-dissoc-in-seq--all-fn})]

    (def evaluation-result evaluation-result)
    (is (= (->>
            (flatten evaluation-result)
            (apply merge)
            keys-in)
           [
            [:train-transform]
            [:train-transform :metric]
            [:train-transform :min]
            [:train-transform :mean]
            [:train-transform :max]
            [:test-transform]
            [:test-transform :metric]
            [:test-transform :min]
            [:test-transform :mean]
            [:test-transform :max]]))
    (is (pos? (-> evaluation-result first first :train-transform :metric)))))


(deftest remove-all
  (do-define-model)
  (let [
        base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species]]
         {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris)
         loss/classification-accuracy
         :accuracy
         {:evaluation-handler-fn (fn [result]
                                   {:train-transform {:metric 1}
                                    :test-transform {:metric 1}})})]

                                   

    (is (pos? (-> evaluation-result first first :train-transform :metric)))))



(deftest other-metrices
  (do-define-model)
  (let [base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species]]
         {:metamorph/id :model}[:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris)
         loss/classification-accuracy
         :accuracy
         {
          :other-metrices [{:name :acc-2  :metric-fn loss/classification-accuracy}
                           {:name :fscore :metric-fn (fn [truth prediction] 0)}
                           {:name :acc    :metric-fn scicloj.metamorph.ml.metrics/accuracy}]})]

    (is (pos? (-> evaluation-result first first :train-transform :other-metrices first :metric)))
    (is (zero? (-> evaluation-result first first :train-transform :other-metrices second :metric)))
    (is (some? (-> evaluation-result first first :train-transform :other-metrices (nth 2) :metric)))))


(deftest validate-schema
 (do-define-model)
 (let [

       create-base-pipe-decl
       (fn  [node-size]
         [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
          [:tech.v3.dataset.metamorph/categorical->number [:species]]
          {:metamorph/id :model}[:scicloj.metamorph.ml/model {:model-type :test-model
                                                              :node-size node-size}]])

       pipes (map create-base-pipe-decl [1 5 10 20 50 100])

       split (tc/split->seq iris :holdout)

       result-schema (-> #'ml/evaluate-pipelines meta :malli/schema second :registry :scicloj.metamorph.ml/evaluation-result)

       evaluation-result
       (ml/evaluate-pipelines
        pipes split
        loss/classification-accuracy
        :accuracy
        {:result-dissoc-in-seq []
         :return-best-crossvalidation-only false
         :return-best-pipeline-only false
         :attach-fn-sources {:ns (find-ns 'clojure.core)
                             :pipe-fns-clj-file "test/scicloj/metamorph/ml_test.clj"}})]


   (is (true?
        (m/validate
         result-schema
         evaluation-result)))))





(deftest call-without-ds
  (do-define-model)
  (is  (thrown? ExceptionInfo
                (ml/train ""
                          {:model-type :test-model}))))


(ml/define-model! :test-model-float-predictions
  (fn train
    [feature-ds label-ds options])

  (fn predict
    [feature-ds thawed-model {:keys [target-columns
                                     target-categorical-maps
                                     top-k
                                     options]}]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) 1.0)
                                    {:column-type :prediction})]))
  {})


(ml/define-model! :test-model-string-predictions
  (fn train
    [feature-ds label-ds options])

  (fn predict [feature-ds thawed-model {:keys [target-columns
                                               target-categorical-maps
                                               top-k
                                               options]}]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) "pred")
                                    {:column-type :prediction})]))
  {})

(deftest test-preditc-float
  (let [model
        (->
         (ds/->dataset {:x [0 1 ] :target ["x" "y"]})
         (ds-mod/set-inference-target :target)
         (ml/train {:model-type :test-model-float-predictions}))]


    (is (= [1.0]
         (-> (ml/predict (ds/->dataset {:x [0]}) model) :species)))))


(deftest test-predict-striong
  (let [model
        (->
         (ds/->dataset {:x [0 1 ] :target ["x" "y"]})
         (ds-mod/set-inference-target :target)
         (ml/train {:model-type :test-model-string-predictions}))]


    (is (= ["pred"]
           (-> (ml/predict (ds/->dataset {:x [0]}) model) :species)))))



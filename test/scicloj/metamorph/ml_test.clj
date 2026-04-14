(ns scicloj.metamorph.ml-test
  (:require
   [scicloj.metamorph.ml.classification]
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.common]
   [malli.core :as m]
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.evaluation-handler
    :as eval
    :refer [qualify-pipelines]]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.metrics]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.io.nippy]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.metamorph :as ds-mm]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.ml.tools :refer [keys-in]]
   [scicloj.metamorph.ml.cache :as cache]
   [scicloj.metamorph.ml.column-metric :as metric]
   [libpython-clj2.python :as py])
  (:import
   (clojure.lang ExceptionInfo)
   (java.util UUID)))


(def iris
  (rdatasets/datasets-iris))

(def iris-train
  (-> iris
      (ds-mod/set-inference-target :species)))

(def iris-target-values (-> iris :species distinct sort))


(defn do-define-model []
  (ml/define-model! :test-model
    (fn train
      [_ _ _]
      {:model-data {:model-as-bytes [1 2 3]
                    :smile-df-used [:blub]}})
    (fn predict
      [feature-ds _ {:keys [target-columns
                                       target-categorical-maps]}]


      (let [predic-col (ds/new-column :species (repeat (tc/row-count feature-ds) 1)
                                      {:categorical-map (get  target-categorical-maps (first target-columns))
                                       :column-type :prediction})
            predict-ds (ds/new-dataset [predic-col])]

        predict-ds))

    {:explain-fn (fn  [thawed-model {:keys [feature-columns]} _options]
                   {:coefficients {:petal_width [0]}})}))

(defn- validate-simple-pipeline []
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) iris-target-values :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout {:seed 123})
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq 
                               {:model-type :classification
                                :metric :accuracy
                                :averaging :micro
                                :loss-or-accuracy :loss
                                :options {}}
                               {:evaluation-handler-fn identity})


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


    (is (= (repeat 10 "versicolor")
           (-> predictions ds-cat/reverse-map-categorical-xforms :species seq)))
    (is (=  1 (count evaluations)))
    (is (=  1 (count (first evaluations))))

    (is (= #{:min :mean :max :timing :ctx :metric :other-metrics  :probability-distribution}
           (set (-> evaluations first first :train-transform keys))))
    ;; =>
    (is (= (set [:fit-ctx :test-transform :train-transform :pipe-fn :pipe-decl :metric-fn :timing-fit :source-information :split-uid])
           (set (keys (first (first evaluations))))))
    (is (contains?   (:fit-ctx (first (first evaluations)))  :metamorph/mode))
    (is (contains?   (:ctx (:train-transform (first (first evaluations))))  :metamorph/mode))
    (is (contains?   (:ctx (:test-transform (first (first evaluations))))  :metamorph/mode))))

(deftest evaluate-pipelines-simplest
  (do-define-model)
  (validate-simple-pipeline))


(deftest evaluate-pipelines-simplest-with-atom-cache
  (do-define-model)
  (let [cache-map (atom {})]
    (cache/enable-atom-cache! cache-map)
    (validate-simple-pipeline)
    (validate-simple-pipeline)
    (cache/disable-cache!)))

(deftest evaluate-pipelines-simplest-with-disk-cache
  (do-define-model)
  (cache/enable-disk-cache! "/tmp/")
  (validate-simple-pipeline)
  (validate-simple-pipeline)
  (cache/disable-cache!))





(deftest test-explain
  (do-define-model)
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) iris-target-values :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq 
                               {:model-type :classification
                                :metric :accuracy
                                :averaging :micro
                                :loss-or-accuracy :loss 
                                :options {}} 
                               
                               {:evaluation-handler-fn identity})

        best-fitted-context  (-> evaluations first first :fit-ctx)
        best-pipe-fn         (-> evaluations first first :pipe-fn)]


    (is (= :petal_width (-> best-fitted-context :model (ml/explain) :coefficients first first)))))



(deftest test-data-removed
  (do-define-model)
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/target ds))) iris-target-values :int)
         (ds-mm/categorical->number (fn [ds] (cf/intersection (cf/categorical ds) (cf/feature ds))) {} :float)

         {:metamorph/id :model}
         (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq {:model-type :classification
                                                            :metric :accuracy
                                                            :averaging :micro
                                                            :loss-or-accuracy :loss

                                                            :options {}})]


    (is (nil? (-> evaluations first first :train-transform :ctx :model :model-data :model-as-bytes)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :model-data :smile-df-used)))
    (is (nil? (-> evaluations first first :test-transform :ctx :model :model-data :model-as-bytes)))
    (is (nil? (-> evaluations first first :test-transform :ctx :model :model-data :smile-df-used)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :scicloj.metamorph.ml/target-ds)))
    (is (nil? (-> evaluations first first :train-transform :ctx :model :scicloj.metamorph.ml/feature-ds)))))






(deftest evaluate-pipelines-several-cross
  (do-define-model)
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical iris-target-values)

         {:metamorph/id :model} (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :kfold)
        pipe-fn-seq [pipe-fn pipe-fn]

        evaluations-1
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq {:model-type :classification
                                                            :metric :accuracy
                                                            :averaging :micro
                                                            :loss-or-accuracy :loss
                                                            :options {}} 
                               {:return-best-crossvalidation-only false})
        evaluations-2
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq {:model-type :classification
                                                            :metric :accuracy
                                                            :averaging :micro
                                                            :loss-or-accuracy :loss
                                                            :options {}} 
                               {:return-best-crossvalidation-only false
                                :return-best-pipeline-only false})

        evaluations-3
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq {:model-type :classification
                                                            :metric :accuracy
                                                            :averaging :micro
                                                            :loss-or-accuracy :loss
                                                            :options {}} 
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
               (let [;;  the data

                     pipe-fn
                     (morph/pipeline
                      (ds-mm/set-inference-target :species)
                      (ds-mm/categorical->number cf/categorical))

                     train-split-seq (tc/split->seq iris :holdout)
                     pipe-fn-seq [pipe-fn]]

                 (ml/evaluate-pipelines pipe-fn-seq train-split-seq {:model-type :classification
                                                                     :metric :accuracy
                                                                     :averaging :micro
                                                                     :options {}} :loss)))))



(deftest evaluate-pipelines--metric-keep-fn
  (do-define-model)
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         {:metamorph/id :model} (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        result
        (ml/evaluate-pipelines
         pipe-fn-seq
         train-split-seq
         {:model-type :classification
          :metric :accuracy
          :averaging :micro
          :loss-or-accuracy :loss
          :options {}}
         {:evaluation-handler-fn
          ;(fn [result] (def result result) result)
          eval/metrics-keep-fn})]
    (is (=
         [[:train-transform]
          [:train-transform :metric]
          [:train-transform :min]
          [:train-transform :max]
          [:train-transform :mean]
          [:test-transform]
          [:test-transform :metric]
          [:test-transform :min]
          [:test-transform :max]
          [:test-transform :mean]]
         (keys-in (-> result first first))))))

(deftest evaluate-pipelines--metric-and-options-keep-fn
  (do-define-model)
  (let [pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         {:metamorph/id :model} (ml/model {:model-type :test-model}))

        train-split-seq (tc/split->seq iris :holdout)
        pipe-fn-seq [pipe-fn]

        result
        (ml/evaluate-pipelines
         pipe-fn-seq
         train-split-seq
         {:model-type :classification
          :metric :accuracy
          :averaging :micro
          :loss-or-accuracy :loss
          :options {}}
         {:evaluation-handler-fn
          eval/metrics-and-options-keep-fn})]
    (is (=
         [[:train-transform]
          [:train-transform :metric]
          [:train-transform :min]
          [:train-transform :max]
          [:train-transform :mean]
          [:test-transform]
          [:test-transform :metric]
          [:test-transform :min]
          [:test-transform :max]
          [:test-transform :mean]
          [:fit-ctx]
          [:fit-ctx :model]
          [:fit-ctx :model :options]
          [:fit-ctx :model :options :model-type]]
         (keys-in (-> result first first))))))






(defn do-xxx [col] col)

(deftest qualify-pipelines-test
  (is (= (repeat 3 :scicloj.metamorph.ml-test/do-xxx)
         (qualify-pipelines [;; 'do-xxx
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
        species-freqs (binding [*ns* new-ns]
                        (eval '(def thawed-result (nippy/thaw-from-file file)))


                        (eval '(def thawed-pipe-fn (clojure.core/->
                                                    thawed-result
                                                    :pipe-decl
                                                    (morph/->pipeline))))
                        (eval '(clojure.core/->
                                (morph/fit-pipe ds thawed-pipe-fn)
                                :metamorph/data
                                :species
                                (clojure.core/frequencies))))]
    species-freqs))


(deftest round-trip-full-names
  (do-define-model)
  (is (= {1 50, 0 50, 2 50}

         (let [files (atom [])

               base-pipe-declr

               [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
                [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
                [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
                {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model}]]
               files (atom [])

               nippy-handler (eval/example-nippy-handler files "/tmp" identity)



               _ (ml/evaluate-pipelines
                            [base-pipe-declr]
                            (tc/split->seq iris)
                            {:model-type :classification
                             :metric :accuracy
                             :averaging :micro
                             :loss-or-accuracy :accuracy
                             :options {}}
                            {:evaluation-handler-fn nippy-handler})]
           (fit-pipe-in-new-ns (first @files) iris)))))




(deftest dissoc--all-fn
  (do-define-model)
  (let [base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
         {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris)
         {:model-type :classification
          :metric :accuracy
          :averaging :micro
          :loss-or-accuracy :accuracy
          :options {}}
         {:evaluation-handler-fn eval/result-dissoc-in-seq--all-fn})]

    (is (=
         [[:train-transform]
          [:train-transform :metric]
          [:train-transform :min]
          [:train-transform :max]
          [:train-transform :mean]
          [:test-transform]
          [:test-transform :metric]
          [:test-transform :min]
          [:test-transform :max]
          [:test-transform :mean]
          [:split-uid]]

         (->>
          (flatten evaluation-result)
          (apply merge)
          keys-in
          vec)))

    (is (pos? (-> evaluation-result first first :train-transform :metric)))))


(deftest remove-all
  (do-define-model)
  (let [base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
         {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris)
         {:model-type :classification
          :metric :accuracy
          :averaging :micro
          :loss-or-accuracy :accuracy
          :options {}}
         {:evaluation-handler-fn (fn [_]
                                   {:train-transform {:metric 1}
                                    :test-transform {:metric 1}})})]



    (is (pos? (-> evaluation-result first first :train-transform :metric)))))



(deftest other-metrics
  (do-define-model)
  (let [base-pipe-declrss
        [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
         [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
         {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model}]]

        evaluation-result
        (ml/evaluate-pipelines
         [base-pipe-declrss]
         (tc/split->seq iris :holdout {:seed 123})
         {:model-type :classification
          :metric :accuracy
          :averaging :micro
          :loss-or-accuracy :accuracy
          :options {}}
         {:other-metrics [{:name :acc-2  :metric-fn {:model-type :classification
                                                     :metric :accuracy
                                                     :averaging :micro
                                                     :options {}}}
                          {:name :f-measure :metric-fn {:model-type :classification
                                                     :metric :accuracy
                                                     :averaging :micro
                                                     :options {}}}
                          {:name :acc    :metric-fn {:model-type :classification
                                                     :metric :accuracy
                                                     :averaging :micro
                                                     :options {}}}]})]

    (is (pos? (-> evaluation-result first first :train-transform :other-metrics first :metric)))
    (is (= 0.3 (-> evaluation-result first first :train-transform :other-metrics second :metric)))
    (is (some? (-> evaluation-result first first :train-transform :other-metrics (nth 2) :metric)))))


;; (deftest validate-schema
;;   (do-define-model)
;;   (let [create-base-pipe-decl
;;         (fn  [node-size]
;;           [[:tech.v3.dataset.metamorph/set-inference-target [:species]]
;;            [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
;;            {:metamorph/id :model} [:scicloj.metamorph.ml/model {:model-type :test-model
;;                                                                 :node-size node-size}]])

;;         pipes (map create-base-pipe-decl [1 5 10 20 50 100])

;;         split (tc/split->seq iris :holdout)

;;         result-schema (-> #'ml/evaluate-pipelines meta :malli/schema second :registry :scicloj.metamorph.ml/evaluation-result)

;;         evaluation-result
;;         (ml/evaluate-pipelines
;;          pipes split
;;          {:model-type :classification
;;           :metric :accuracy
;;           :averaging :micro
;;           :loss-or-accuracy :accuracy
;;           :options {}}
;;          {:result-dissoc-in-seq []
;;           :return-best-crossvalidation-only false
;;           :return-best-pipeline-only false
;;           :attach-fn-sources {:ns (find-ns 'clojure.core)
;;                               :pipe-fns-clj-file "test/scicloj/metamorph/ml_test.clj"}})]


;;     (is (true?
;;          (m/validate
;;           result-schema
;;           evaluation-result)))))



(deftest call-without-ds
  (do-define-model)
  (is  (thrown? ExceptionInfo
                (ml/train ""
                          {:model-type :test-model}))))


(ml/define-model! :test-model-float-predictions
  (fn train
    [_ _ _])

  (fn predict
    [feature-ds _ _]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) 1.0)
                                    {:column-type :prediction})]))
  {})


(ml/define-model! :test-model-string-predictions
  (fn train
    [_ _ _])

  (fn predict [feature-ds _ _]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) "pred")
                                    {:column-type :prediction})]))
  {})

(deftest test-preditc-float
  (let [model
        (->
         (ds/->dataset {:x [0 1] :target ["x" "y"]})
         (ds-mod/set-inference-target :target)
         (ml/train {:model-type :test-model-float-predictions}))]


    (is (= [1.0]
           (-> (ml/predict (ds/->dataset {:x [0]}) model) :species)))))


(deftest test-predict-striong
  (let [model
        (->
         (ds/->dataset {:x [0 1] :target ["x" "y"]})
         (ds-mod/set-inference-target :target)
         (ml/train {:model-type :test-model-string-predictions}))]


    (is (= ["pred"]
           (-> (ml/predict (ds/->dataset {:x [0]}) model) :species)))))

(defn- do-score [predict-col trueth-col metric-fn]
  (#'ml/score
   (ds/new-dataset  [predict-col])
   (ds/new-dataset  [trueth-col])
   metric-fn
   {}))


(defn is-accuracy [predict-col trueth-col metric-fn expected-acc]
  (is (= {:metric expected-acc, :other-metrics-result []}
         (do-score predict-col trueth-col metric-fn))))



(defn- score-categorical [predict-col-seq predict-a-b-table
                          trueth-col-seq trueth-a-b-table
                          metric-fn]
  (do-score
   (ds/new-column  :species predict-col-seq
                   (merge {:column-type :prediction}
                          (when predict-a-b-table
                            {:categorical-map
                             {:lookup-table predict-a-b-table
                              :src-column :species}})))
   (ds/new-column  :species trueth-col-seq
                   (merge {:inference-target? true}
                          (when trueth-a-b-table
                            {:categorical-map
                             {:lookup-table trueth-a-b-table
                              :src-column :species}})))
   metric-fn))

(defn is-mapped-columns-accuracy [predict-col-seq predict-a-b-table
                                  trueth-col-seq trueth-a-b-table
                                  metric-fn
                                  expected-accuracy]

  (is (= {:metric expected-accuracy, :other-metrics-result []}

         (score-categorical
          predict-col-seq predict-a-b-table
          trueth-col-seq trueth-a-b-table
          metric-fn))))


(deftest test-score

  (is-accuracy
   (ds/new-column  :species [1 1 1 1 1 1] {:column-type :prediction})
   (ds/new-column  :species [1 1 1 0 0 0] {:inference-target? true})
   {:model-type :classification
    :metric :accuracy
    :averaging :micro
    :options {}}
   0.5)


  (is-accuracy
   (ds/new-column  :species [:a :a] {:column-type :prediction})
   (ds/new-column  :species [:a :b] {:inference-target? true})
   {:model-type :classification
    :metric :accuracy
    :averaging :micro
    :options {}}
   0.5)

  (is-mapped-columns-accuracy [0 1] {:a 0 :b 1}
                              [0 1] {:a 0 :b 1}
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              1.0)

  (is-mapped-columns-accuracy [0 1] {:a 0 :b 1}
                              [1 0] {:a 1 :b 0}
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              1.0)

  (is-mapped-columns-accuracy [:a :b] nil
                              [1 0] {:a 1 :b 0}
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              1.0)

  (is-mapped-columns-accuracy [:a :b] nil
                              [:a :b] nil
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              1.0)

  (is-mapped-columns-accuracy [0.0 1.0] {:a 0 :b 1}
                              [0.0 1.0] {:a 0 :b 1}
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              1.0)



  (is-mapped-columns-accuracy [0 1] {:a 0 :b 1}
                              [1 0] {:a 0 :b 1}
                              {:model-type :classification
                               :metric :accuracy
                               :averaging :micro
                               :options {}}
                              0.0))

(deftest score-failing
  (is (thrown? Exception
               (score-categorical [0.0 1.0] {:a 0.0 :b 1.0}
                                  [0 1]     {:a 0 :b 1}
                                  loss/classification-accuracy)))
  (is (thrown? Exception
               (score-categorical   [0 1] {:a 0.0 :b 1.0}
                                    [0 1] {:a 0.0 :b 1.0}
                                    loss/classification-accuracy))))



(deftest score-other-metrics
  (is (=
       {:metric 0.6666666666666666,
        :other-metrics-result
        [{:name :m-1, :metric-fn {:model-type :classification
                                  :metric :accuracy
                                  :averaging :micro
                                  :options {}} :metric 0.6666666666666666}
         {:name :m-2, :metric-fn {:model-type :classification
                                  :metric :precision
                                  :averaging :macro
                                  :options {}} :metric 0.3333333333333333}]}

       (#'ml/score
        (ds/new-dataset [(ds/new-column :x [:a :a :a] {:column-type :prediction})])
        (ds/new-dataset [(ds/new-column :x [:a :b :a] {:inference-target? true})])
        {:model-type :classification
         :metric :accuracy
         :averaging :micro
         :options {}}
        [{:name :m-1
          :metric-fn {:model-type :classification
                      :metric :accuracy
                      :averaging :micro
                      :options {}}}
         {:name :m-2
          :metric-fn {:model-type :classification
                      :metric :precision
                      :averaging :macro
                      :options {}}}]))))

(deftest define-model-schema
  (ml/define-model! :test-model--options
    (fn train
      [_ _ _])
    (fn predict
      [_ _])
    {:options [:map {:closed true}]})


  (try
    (ml/train iris-train {:model-type :test-model--options
                          :a 1})
    (throw (Exception.))
    (catch ExceptionInfo e
      (is (=
           {:a ["disallowed key"]}
           (ex-data e)))))
  (is (nil?
       (-> 
        (ml/train iris-train {:model-type :test-model--options})
        :model-data))))

(deftest non-consistent-cat-map []
  (is ( thrown?  Exception
       (ml/train
        (->
         (ds/new-dataset [(ds/new-column :x [0 1 2 3]
                                         {:categorical-map (ds-cat/create-categorical-map {:x-0 0
                                                                                           :x-1 1}
                                                                                          :x
                                                                                          :int16)})
                          (ds/new-column  :y [0 1 0 1 2]
                                          {:categorical-map (ds-cat/create-categorical-map {:y-0 0
                                                                                            :y-1 1}
                                                                                           :y
                                                                                           :int16)})]))
        {:model-type :metamorph.ml/dummy-classifier
         :dummy-strategy :fixed-class
         :fixed-class 3}))))


(deftest train-predict-validate 
  (require '[scicloj.ml.smile.classification])
  
  (let [data
        (-> iris
            (ds/categorical->number [:species])
            (ds-mod/set-inference-target :species))]
    


    (is (= [0.9802469135802468 0.9703630507978335 1.0]
           (let [{:keys [train test]} (first (tc/split->seq data :holdout {:seed 123
                                                                           :ratio [0.1 0.9]}))
                 logreg (ml/train train {:model-type :smile.classification/logistic-regression})
                 prediction (ml/predict test logreg)
                 ]
             [
              (metric/classification-metric--fastmath test prediction :accuracy :macro)
              (metric/classification-metric--fastmath test prediction :f1-score :macro)
              (metric/roc_auc-score test prediction :ovr :macro)]

             ))))
  
  ;;=> [0.9802469135802468 0.9703630507978335 1.0]
  


  
  )
 

(comment 
  (require '[libpython-clj2.python :as py])
  
  (py/from-import sklearn.metrics roc_auc_score)
  

  (roc_auc_score
   (py/->py-list
    (map
     (fn [_] (rand-int 3))
     (-> data-split :test cf/target ds/columns first vec)))
   
   (-> prediction cf/probability-distribution ds/rowvecs py/->python)
   :multi_class :ovr
   :average nil)
  )






(comment 
  (require '[scicloj.ml.smile.classification])
  
  (->>
   (ml/train
    (->  (ds/->dataset {:x [1 0] :y [1 0]})
         (ds-mod/set-inference-target [:y])
         (ds/assoc-metadata [:y] 
                            :categorical-map nil
                            :categorical? true)
         )
    {:model-type :smile.classification/ada-boost})
   (ml/predict (ds/->dataset {:x [1 0] :y [1 0]}))
   :y
   meta
   :categorical-map
   )
  

  (defn call->lable-column [m]
    (->
     (ds-mod/probability-distributions->label-column
      (ds/->dataset
       m)
      :x
      :int64)
     (get :x)
     ( (fn [col]
         {:datatype (-> col meta :datatype)
          :lookup-table (-> col meta :categorical-map :lookup-table)
          :values (vec col)}
         ))
     ))
  

  (call->lable-column {:x-1 [0.4] :x-2 [0.6]})
  
  ;;=> {:datatype :int64, :lookup-table {:x-1 0, :x-2 1}, :values [1]}
  
  (call->lable-column {0 [0.4] 1 [0.6]})
  
  ;;=> {:datatype :int64, :lookup-table {0 0, 1 1}, :values [1]}
  
  (call->lable-column {1 [0.4] 0 [0.6]})
  ;;=> {:datatype :int64, :lookup-table {1 0, 0 1}, :values [1]}
  )

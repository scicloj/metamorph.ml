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
            [scicloj.metamorph.persistence-tools :refer [qualify-pipelines get-source-information qualify-keywords]])
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

    (is (= #{:min :mean :max :timing :ctx :metric}
           (set (-> evaluations first first :train-transform keys))))
    ;; =>
    (is (= (set [:fit-ctx :test-transform :train-transform :pipe-fn :pipe-decl :metric-fn :timing-fit :loss-or-accuracy])
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

               nippy-handler (fn [result]

                               (let [freezable-result
                                     (-> result
                                         (ml/multi-dissoc-in  [
                                                               [:pipe-fn]
                                                               [:metric-fn]])
                                         (assoc :source-information
                                                (-> result :pipe-decl get-source-information)))

                                     temp-file (.getPath (File/createTempFile "test" ".nippy"))
                                     _ (swap! files #(conj % temp-file))]
                                 (nippy/freeze-to-file temp-file freezable-result)))


               eval-result (ml/evaluate-pipelines
                            [base-pipe-declr]
                            (tc/split->seq ds)
                            loss/classification-accuracy
                            :accuracy
                            {:evaluation-handler-fn nippy-handler})]
           (fit-pipe-in-new-ns (first @files) ds)))))



(deftest dummy
  (println "qualify: "))


(get-source-information (qualify-keywords [[:ds-mm/set-inference-target [:species]]] (find-ns 'scicloj.metamorph.ml-test)))

(->
 (ns-publics 'tech.v3.dataset)
 vals
 (first)
 symbol
 (clojure.repl/source-fn))

(meta
 #'tech.v3.dataset.metamorph/add-column)


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

               nippy-handler (fn [result]

                               (let [freezable-result
                                     (-> result
                                         (ml/multi-dissoc-in  [
                                                               [:pipe-fn]
                                                               [:metric-fn]])
                                         (assoc :source-information
                                                (-> result :pipe-decl get-source-information)))

                                     temp-file (.getPath (File/createTempFile "test" ".nippy"))
                                     _ (swap! files #(conj % temp-file))]
                                 (nippy/freeze-to-file temp-file freezable-result)))

               eval-result (ml/evaluate-pipelines
                            base-pipe-declr
                            (tc/split->seq ds)
                            loss/classification-accuracy
                            :accuracy
                            {:evaluation-handler-fn nippy-handler})]

           (fit-pipe-in-new-ns (first @files) ds)))))







(comment

  (def ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))



  (def base-pipe-declrss

    [[:tech.v3.dataset.metamorph/set-inference-target [:species]
      [:tech.v3.dataset.metamorph/categorical->number [:species]]]
     [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
     [:scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest}]])






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
   (-> #'ml/evaluate-pipelines meta :malli/schema (nth 2))
   (m/explain
    (ml/evaluate-pipelines
     [base-pipe-declrss]
     (tc/split->seq ds)
     loss/classification-accuracy
     :accuracy))
   (me/humanize))



  (require '[malli.dot :as md])

  (->>
   (-> #'ml/evaluate-pipelines meta :malli/schema (nth 2))

   (md/transform)
   (spit "/tmp/ev.dot"))

   ;; me/humanize



  

  :ok)

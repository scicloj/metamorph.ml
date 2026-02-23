;; # Hyperparameter Tuning with Gridsearch and evaluate-pipelines

;; An interactive tutorial demonstrating hyperparameter optimization in metamorph.ml

^{:kindly/hide-code true}
(ns notebooks.gridsearch-tutorial
  (:require [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.gridsearch :as gs]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.metamorph.ml.metrics :as metrics]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [scicloj.ml.smile.classification]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.metamorph :as ds-mm]
            
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.kindly.v4.kind :as kind]
            [tech.v3.dataset.categorical :as ds-cat]))

;; ## Introduction

;; This tutorial demonstrates how to use **metamorph.ml**'s powerful tools for
;; hyperparameter optimization:

;; - **`evaluate-pipelines`**: Evaluates ML pipelines across train/test splits
;; - **`sobol-gridsearch`**: Generates hyperparameter combinations using efficient Sobol sequences

;; Together, they enable systematic exploration of hyperparameter spaces to find
;; optimal model configurations.


;; Let's start with the classic Iris dataset:

(def iris-ds
  (-> (rdatasets/datasets-iris)
      (ds-mod/set-inference-target :species)))

;; ### Dataset Overview

^{:kindly/hide-code false}
(kind/table
 (tc/head iris-ds 10))

;; ### Dataset Statistics

^{:kindly/hide-code false}
(kind/table
 (tc/info iris-ds))

;; ### Target Distribution

(def species-counts
  (-> iris-ds
      (tc/group-by [:species])
      (tc/aggregate {:count tc/row-count})))

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values (tc/rows species-counts :as-maps)}
  :mark :bar
  :encoding {:x {:field :species :type :nominal}
             :y {:field :count :type :quantitative}
             :color {:field :species :type :nominal}}
  :width 400
  :height 300})
;; ## Part 2: Basic Pipeline Evaluation

;; First, let's evaluate a single pipeline to understand the workflow.
;;
;; ### Create Train/Test Splits
;; 
;; We'll use 5-fold cross-validation:

(def iris-splits
  (tc/split->seq iris-ds :kfold {:k 5 :seed 42}))

;; Verify we have 5 splits:
(count iris-splits)

;; Each split contains `:train` and `:test` datasets:
(keys (first iris-splits))

;; ### Define a Simple Pipeline
;; 
;; A metamorph pipeline chains data transformations and model training:

(def simple-pipeline
  (morph/pipeline
   ;; Convert species to numeric (required for classification)
   (ds-mm/categorical->number [:species])
   ;; Add model step
   {:metamorph/id :model}
   (ml/model {:model-type :smile.classification/random-forest
              :max-depth 10
              :trees 50})))

;; ### Evaluate the Pipeline

(def simple-results
  (ml/evaluate-pipelines
   [simple-pipeline]                      ; Vector of pipelines
   iris-splits                            ; Train/test splits
   loss/classification-accuracy           ; Metric function
   :accuracy                              ; Higher is better
   {:return-best-pipeline-only false      ; Keep all results
    :return-best-crossvalidation-only false}))

;; ### Extract and Visualize Results

(def cv-scores
  (for [[fold-idx result] (map-indexed vector (first simple-results))]
    {:fold (inc fold-idx)
     :train-accuracy (-> result :train-transform :metric)
     :test-accuracy (-> result :test-transform :metric)}))

^{:kindly/hide-code false}
(kind/table cv-scores)

;; ### Cross-Validation Performance Visualization

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values cv-scores}
  :mark :bar
  :encoding {:x {:field :fold :type :ordinal :title "Fold"}
             :y {:field :test-accuracy :type :quantitative :title "Accuracy"}
             :color {:value "steelblue"}}
  :width 500
  :height 300
  :title "Test Accuracy Across Folds"})

;; ### Calculate Summary Statistics

(def avg-train-acc
  (/ (reduce + (map :train-accuracy cv-scores))
     (count cv-scores)))

(def avg-test-acc
  (/ (reduce + (map :test-accuracy cv-scores))
     (count cv-scores)))

^{:kindly/hide-code false}
(kind/table
 [{:metric "Average Train Accuracy" :value (format "%.4f" avg-train-acc)}
  {:metric "Average Test Accuracy" :value (format "%.4f" avg-test-acc)}
  {:metric "Overfitting Gap" :value (format "%.4f" (- avg-train-acc avg-test-acc))}])

;; ## Part 3: Introduction to Gridsearch
;;
;; Instead of manually trying different hyperparameters, gridsearch
;; systematically explores the parameter space.

;; ### Defining Search Spaces
;;
;; Use `gs/linear` for numeric ranges and `gs/categorical` for discrete choices:

(def example-search-space
  {:model-type :smile.classification/random-forest
   :max-depth (gs/linear 5 20 10 :int16)          ; 10 values from 5 to 20
   :trees (gs/linear 10 200 10 :int16)        ; 10 values from 10 to 200
   :split-rule (gs/categorical [:gini :entropy])})

;; ### Generate Parameter Combinations
;;
;; `sobol-gridsearch` uses Sobol sequences for efficient space exploration:

(def param-combos
  (take 20 (gs/sobol-gridsearch example-search-space)))

;; Let's visualize how Sobol sequences distribute samples:

(def combo-data
  (map-indexed
   (fn [idx combo]
     {:sample idx
      :max-depth (:max-depth combo)
      :num-trees (:num-trees combo)
      :split-rule (name (:split-rule combo))})
   param-combos))

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values combo-data}
  :mark {:type :point :size 100}
  :encoding {:x {:field :max-depth :type :quantitative :title "Max Depth"}
             :y {:field :num-trees :type :quantitative :title "Number of Trees"}
             :color {:field :split-rule :type :nominal :title "Split Rule"}
             :shape {:field :split-rule :type :nominal}}
  :width 500
  :height 400
  :title "Sobol Sequence Coverage of Hyperparameter Space"})

;; Notice how Sobol sequences spread points evenly across the space,
;; unlike random sampling which can cluster.

;; ### View First 5 Parameter Combinations

^{:kindly/hide-code false}
(kind/table
 (take 5 combo-data))

;; ## Part 4: Hyperparameter Optimization
;;
;; Now let's combine gridsearch with evaluate-pipelines for
;; automated hyperparameter tuning.

;; ### Define Comprehensive Search Space

(def rf-search-space
  {:model-type :smile.classification/random-forest
   :max-depth (gs/linear 5 30 8 :int16)           ; 8 depth values
   :trees (gs/linear 50 300 8 :int16)         ; 8 tree count values
   :mtry (gs/linear 1 4 4 :int16)                 ; 4 values for features per split
   :split-rule (gs/categorical [:gini :entropy])  ; 2 options
   :sample-rate (gs/linear 0.6 1.0 5)})    ; 5 subsample rates

;; Total unique combinations: 8 × 8 × 4 × 2 × 5 = 2,560
;; We'll sample 40 of them:

(def rf-params
  (take 40 (gs/sobol-gridsearch rf-search-space)))

;; ### Create Pipeline for Each Parameter Set

(defn make-rf-pipeline [params]
  (morph/pipeline
   (ds-mm/categorical->number [:species])
   {:metamorph/id :model}
   (ml/model params)))

(def rf-pipelines
  (map make-rf-pipeline rf-params))

(count rf-pipelines)

;; ### Run Grid Search with Cross-Validation
;;
;; This evaluates all 40 pipelines across 5 folds = 200 model trainings!

(println "Starting grid search (this may take a minute)...")

(def grid-results
  (ml/evaluate-pipelines
   rf-pipelines
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false       ; Keep all for analysis
    :return-best-crossvalidation-only true ; One result per pipeline
    :map-fn :map}))                        ; Sequential (use :pmap for parallel)

(println "Grid search complete!")

;; ### Analyze Grid Search Results

(def grid-analysis
  (map-indexed
   (fn [idx results]
     (let [result (first results)
           params (-> result :fit-ctx :model :options)]
       {:pipeline-idx idx
        :test-accuracy (-> result :test-transform :metric)
        :train-accuracy (-> result :train-transform :metric)
        :max-depth (:max-depth params)
        :num-trees (:num-trees params)
        :mtry (:mtry params)
        :split-rule (name (:split-rule params))
        :sample-rate (:sample-rate params)}))
   grid-results))

;; ### Top 10 Performing Configurations

(def top-configs
  (take 10 (reverse (sort-by :test-accuracy grid-analysis))))

^{:kindly/hide-code false}
(kind/table
 (map #(update % :test-accuracy (fn [v] (format "%.4f" v))) top-configs))

;; ### Visualize Performance Distribution

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values grid-analysis}
  :mark :bar
  :encoding {:x {:field :test-accuracy
                 :type :quantitative
                 :bin {:maxbins 20}
                 :title "Test Accuracy"}
             :y {:aggregate :count
                 :title "Number of Configurations"}}
  :width 600
  :height 300
  :title "Distribution of Model Performance"})

;; ### Hyperparameter Impact Analysis
;;
;; How does each hyperparameter affect performance?

;; #### Max Depth vs. Accuracy

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values grid-analysis}
  :mark {:type :point :size 80 :opacity 0.6}
  :encoding {:x {:field :max-depth :type :quantitative :title "Max Depth"}
             :y {:field :test-accuracy :type :quantitative :title "Test Accuracy"}
             :color {:field :split-rule :type :nominal}}
  :width 500
  :height 350
  :title "Impact of Max Depth on Accuracy"})

;; #### Number of Trees vs. Accuracy

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values grid-analysis}
  :mark {:type :point :size 80 :opacity 0.6}
  :encoding {:x {:field :num-trees :type :quantitative :title "Number of Trees"}
             :y {:field :test-accuracy :type :quantitative :title "Test Accuracy"}
             :color {:field :split-rule :type :nominal}}
  :width 500
  :height 350
  :title "Impact of Number of Trees on Accuracy"})

;; #### Features per Split (mtry) vs. Accuracy

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values grid-analysis}
  :mark :boxplot
  :encoding {:x {:field :mtry :type :ordinal :title "Features per Split (mtry)"}
             :y {:field :test-accuracy :type :quantitative :title "Test Accuracy"}}
  :width 500
  :height 350
  :title "Impact of mtry on Accuracy"})

;; ### Best Model Details

(def best-config (first top-configs))

^{:kindly/hide-code false}
(kind/table
 [{:metric "Test Accuracy" :value (format "%.4f" (:test-accuracy best-config))}
  {:metric "Max Depth" :value (:max-depth best-config)}
  {:metric "Number of Trees" :value (:num-trees best-config)}
  {:metric "Features per Split" :value (:mtry best-config)}
  {:metric "Split Rule" :value (:split-rule best-config)}
  {:metric "Sample Rate" :value (format "%.2f" (:sample-rate best-config))}])

;; ## Part 5: Using the Best Model
;;
;; Extract the best model and use it for predictions.

(def best-idx (:pipeline-idx best-config))
(def best-result (first (nth grid-results best-idx)))
(def best-pipeline (:pipe-fn best-result))
(def best-fitted-ctx (:fit-ctx best-result))

;; ### Make Predictions on New Data

(def test-samples
  (tc/shuffle iris-ds {:seed 999}))

(def predictions
  (-> (best-pipeline
       (merge best-fitted-ctx
              {:metamorph/data test-samples
               :metamorph/mode :transform}))
      :metamorph/data))

;; ### Compare Predictions with True Labels

(def prediction-comparison
  (let [prediction-reversed-cat (-> predictions ds-cat/reverse-map-categorical-xforms)]
    (tc/dataset
     {:true-species (vec (:species test-samples))
      :predicted-species (vec (:species prediction-reversed-cat))
      :correct? (map = (:species test-samples) 
                     (:species prediction-reversed-cat))})))


^{:kindly/hide-code false}
(kind/table
 (tc/head prediction-comparison 15))

;; ### Prediction Accuracy

(def prediction-accuracy
  (/ (count (filter true? (:correct? prediction-comparison)))
     (tc/row-count prediction-comparison)))

^{:kindly/hide-code false}
(kind/hiccup
 [:div
  [:h3 "Prediction Accuracy on Shuffled Data"]
  [:p {:style "font-size: 24px; font-weight: bold; color: #2c5aa0;"}
   (format "%.2f%%" (* 100 prediction-accuracy))]])

;; ## Part 6: Comparing Multiple Model Types
;;
;; Let's compare Random Forest with other classifiers.

;; ### Define Search Spaces for Different Models

(def svm-search
  {:model-type :smile.classification/svm
   :C (gs/linear 0.1 10.0 8)
   :tol (gs/linear 0.01 0.5 5)})

(def logistic-search
  {:model-type :smile.classification/logistic-regression
   :lambda (gs/linear 0.0 1.0 10)})

;; ### Generate Parameter Combinations

(def all-model-params
  (concat
   (take 15 (gs/sobol-gridsearch rf-search-space))
   ;(take 15 (gs/sobol-gridsearch svm-search))
   (take 10 (gs/sobol-gridsearch logistic-search))))

(count all-model-params)

;; ### Create and Evaluate All Pipelines

(defn make-general-pipeline [params]
  (morph/pipeline
   (ds-mm/categorical->number [:species])
   {:metamorph/id :model}
   (ml/model params)))

(def all-pipelines
  (map make-general-pipeline all-model-params))

(println "Evaluating all model types...")

(def multi-model-results
  (ml/evaluate-pipelines
   all-pipelines
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false
    :return-best-crossvalidation-only true
    :map-fn :map}))

;; ### Compare Model Types

(def model-comparison
  (map-indexed
   (fn [idx results]
     (let [result (first results)
           params (-> result :fit-ctx :model :options)
           model-type (name (:model-type params))]
       {:model-type (last (clojure.string/split model-type #"/"))
        :test-accuracy (-> result :test-transform :metric)
        :train-accuracy (-> result :train-transform :metric)}))
   multi-model-results))

^{:kindly/hide-code false}
(kind/vega-lite
 {:data {:values model-comparison}
  :mark :boxplot
  :encoding {:x {:field :model-type :type :nominal :title "Model Type"}
             :y {:field :test-accuracy :type :quantitative :title "Test Accuracy"}
             :color {:field :model-type :type :nominal}}
  :width 600
  :height 400
  :title "Performance Comparison Across Model Types"})

;; ### Best Model from Each Type

(def best-by-type
  (->> model-comparison
       (group-by :model-type)
       (map (fn [[model-type configs]]
              {:model-type model-type
               :best-accuracy (apply max (map :test-accuracy configs))
               :avg-accuracy (/ (reduce + (map :test-accuracy configs))
                               (count configs))}))
       (sort-by :best-accuracy)
       reverse))

^{:kindly/hide-code false}
(kind/table
 (map #(-> %
           (update :best-accuracy (fn [v] (format "%.4f" v)))
           (update :avg-accuracy (fn [v] (format "%.4f" v))))
      best-by-type))

;; ## Part 7: Advanced Techniques
;;
;; ### Memory-Efficient Grid Search
;;
;; For large searches, use custom evaluation handlers to reduce memory:

(require '[scicloj.metamorph.ml.evaluation-handler :as eval-handler])

(def memory-efficient-results
  (ml/evaluate-pipelines
   (take 10 all-pipelines)
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only true        ; Only keep best
    :return-best-crossvalidation-only true
    :evaluation-handler-fn eval-handler/result-dissoc-in-seq--all-fn}))

;; ### Multiple Metrics
;;
;; Evaluate additional metrics beyond the main one:

(def multi-metric-results
  (ml/evaluate-pipelines
   [(first rf-pipelines)]
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false
    :return-best-crossvalidation-only false
    :other-metrics
    [
     {
      :name :precision 
      :metric-fn
      (fn [y-true y-pred]
        (metrics/precision y-true y-pred))}
     {:name :recall 
      :metric-fn
      (fn [y-true y-pred]
                (metrics/recall y-true y-pred))}]
    }))

;; Extract metrics from first fold:
(def fold-metrics
  (let [result (-> multi-metric-results first first :test-transform)]
    (def result result)
    {:accuracy (:metric result)
     :f1-macro (get-in result [:other-metrics :f1-macro])
     :precision (get-in result [:other-metrics :precision])
     :recall (get-in result [:other-metrics :recall])}))

(-> result :other-metrics)
^{:kindly/hide-code false}
(kind/table
 (map (fn [[k v]] {:metric (name k) :value (format "%.4f" v)})
      fold-metrics))

;; ## Summary
;;
;; In this tutorial, we've covered:
;;
;; 1. **Basic Pipeline Evaluation** - Using `evaluate-pipelines` for cross-validation
;; 2. **Gridsearch Fundamentals** - Creating search spaces with `sobol-gridsearch`
;; 3. **Hyperparameter Optimization** - Combining both tools for automated tuning
;; 4. **Result Analysis** - Visualizing and interpreting grid search results
;; 5. **Model Selection** - Comparing different model types
;; 6. **Advanced Techniques** - Memory management and multiple metrics
;;
;; ### Key Takeaways
;;
;; - Sobol sequences efficiently explore hyperparameter spaces
;; - `evaluate-pipelines` handles cross-validation automatically
;; - Parallel evaluation speeds up grid search (`:map-fn :pmap`)
;; - Custom handlers manage memory for large searches
;; - Visualization helps understand hyperparameter impacts
;;
;; ### Next Steps
;;
;; - Try different model types (gradient boosting, neural networks)
;; - Experiment with preprocessing pipelines
;; - Use nested cross-validation for unbiased estimates
;; - Implement custom metric functions for specific use cases

^{:kindly/hide-code true}
(kind/hiccup
 [:div {:style "background-color: #e8f4f8; padding: 20px; border-radius: 8px; margin-top: 20px;"}
  [:h3 "📚 Additional Resources"]
  [:ul
   [:li [:a {:href "https://github.com/scicloj/metamorph"} "metamorph documentation"]]
   [:li [:a {:href "https://github.com/scicloj/metamorph.ml"} "metamorph.ml repository"]]
   [:li [:a {:href "https://scicloj.github.io/clay/"} "Clay documentation"]]
   [:li "API docs: " [:code "(doc ml/evaluate-pipelines)"] " and " [:code "(doc gs/sobol-gridsearch)"]]]])

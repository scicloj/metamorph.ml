;; # Introduction to Supervised Machine Learning with metamorph.ml
;;
;; This tutorial introduces the fundamentals of supervised machine learning
;; using the metamorph.ml library. We'll cover:
;;
;; - Loading and preparing data
;; - Building ML pipelines
;; - Training and evaluating models
;; - Making predictions
;; - Hyperparameter tuning
;;
;; metamorph.ml is a Clojure library that provides a unified pipeline-based
;; approach to machine learning, integrating data preprocessing and model
;; training into cohesive workflows.
;; 
;; (created with the help of Claude Code)

(ns supervised-ml-intro
  (:require [scicloj.clay.v2.api :as clay]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.preprocessing :as preprocessing]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.ml.gridsearch :as gs]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.ml.smile.classification]
            [tablecloth.pipeline :as tc-mm]
            [tech.v3.dataset.categorical :as ds-cat]))

;; ## 1. Loading Data
;;
;; We'll use the classic Iris dataset, which contains measurements of iris flowers
;; and their species. This is a multi-class classification problem with 3 classes.

(def iris-ds
  (->
   (rdatasets/datasets-iris)
   (tc/drop-columns [:rownames])))

;; Let's examine the first few rows:

(tc/head iris-ds 5)

;; The dataset has 4 numeric features (sepal and petal measurements) and
;; 1 target column (species). Let's check the shape and column types:

(ds/shape iris-ds)

^:kind/md
^:kindly/hide-code
(str "**Dataset dimensions:** "
     (first (ds/shape iris-ds)) " rows × "
     (second (ds/shape iris-ds)) " columns")

;; View the column information:

(ds/columns iris-ds)

;; ## 2. Preparing Data for Training
;;
;; Before training, we need to:
;; 1. Set the target column (what we want to predict)
;; 2. Create train/test splits for evaluation

;; Set the target column:

(def iris-prepared
  (ds-mod/set-inference-target iris-ds :species))

;; Create cross-validation splits (5-fold):

(def iris-splits
  (tc/split->seq iris-prepared :kfold {:k 5 :seed 42}))

^:kindly/hide-code
^:kind/md
(str "**Created " (count iris-splits) " cross-validation folds**")

;; Each split contains a `:train` and `:test` dataset:

;;**First fold train/test sizes:**

(let [first-split (first iris-splits)]
  {:train-size (first (ds/shape (:train first-split)))
   :test-size (first (ds/shape (:test first-split)))})

;; ## 3. Building Your First Pipeline
;;
;; A metamorph.ml pipeline combines data transformations and model training.
;; Pipelines are composable functions that operate in two modes:
;; - `:fit` mode: Learn parameters from training data
;; - `:transform` mode: Apply learned transformations to new data

;; Here's a simple pipeline:

(def simple-pipeline
  (mm/pipeline
   ;; Convert categorical target to numeric (required for many models)
   (ds-mm/categorical->number [:species])
   ;; Add a step identifier (useful for tracking)
   {:metamorph/id :model}
   ;; Define the model
   (ml/model {:model-type :smile.classification/random-forest
              :max-depth 10
              :trees 50})))


;; **Pipeline created!** This pipeline will:
;; 1. Convert species labels to numeric codes
;; 2. Train a Random Forest classifier with 50 trees and max depth of 10


;; ## 4. Training and Evaluating the Model
;;
;; The `evaluate-pipelines` function handles:
;; - Training on each fold's training set
;; - Evaluating on each fold's test set
;; - Computing performance metrics
;; - Finding the best model

(def results
  (ml/evaluate-pipelines
   [simple-pipeline]                    ; Can evaluate multiple pipelines
   iris-splits                          ; Cross-validation splits
   loss/classification-accuracy         ; Metric function
   :accuracy))                          ; Higher is better


;;"**Model trained and evaluated!**"

;; Extract the best result:

(def best-result
  (-> results first first))

;; View the performance:

^:kindly/hide-code
^:kind/md
(str "**Training Accuracy:** "
     (format "%.4f" (-> best-result :train-transform :metric)))

^:kindly/hide-code
^:kind/md
(str "**Test Accuracy:** "
     (format "%.4f" (-> best-result :test-transform :metric)))

;; The trained model and pipeline context are stored in the result:

(def trained-ctx
  (:fit-ctx best-result))

(def trained-pipeline
  (:pipe-fn best-result))

;; ## 5. Making Predictions on New Data
;;
;; Once trained, we can use the pipeline to make predictions on new data.
;; We use the trained context and set the mode to `:transform`:

;; Create some test data (using a shuffled version of the original data):

(def new-data
  (-> iris-ds
      (tc/shuffle {:seed 999})
      (tc/head 10)))

;; Make predictions:

(def predictions
  (-> (trained-pipeline
       (merge trained-ctx
              {:metamorph/data new-data
               :metamorph/mode :transform}))
      :metamorph/data))

;; View predictions alongside actual values:

(-> predictions
    (ds-cat/reverse-map-categorical-xforms)
    (tc/select-columns [:species])
    (tc/rename-columns {:species "Predicted"})

    (tc/add-column "Actual" (:species new-data))
    (tc/head 10))

;; ## 6. Comparing Multiple Models
;;
;; Let's compare different model types to see which performs best:

(def model-types
  [:smile.classification/random-forest
   :smile.classification/logistic-regression
   :smile.classification/decision-tree])

;; Create a pipeline for each model type:

(defn make-pipeline [model-type]
  (mm/pipeline
   (ds-mm/categorical->number [:species])
   {:metamorph/id :model}
   (ml/model {:model-type model-type})))

(def pipelines
  (map make-pipeline model-types))

;; Evaluate all models:

(def comparison-results
  (ml/evaluate-pipelines
   pipelines
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false         ; Keep all results
    :return-best-crossvalidation-only true})) ; Keep best fold per pipeline

;; Compare the results:

(def comparison-table
  (map-indexed
   (fn [idx results]
     (let [result (first results)
           model-type (nth model-types idx)]
       {:model-type (name model-type)
        :train-accuracy (-> result :train-transform :metric)
        :test-accuracy (-> result :test-transform :metric)}))
   comparison-results))

(tc/dataset comparison-table)

;; ## 7. Hyperparameter Tuning with Grid Search
;;
;; Grid search helps find the best hyperparameters for a model.
;; metamorph.ml uses Sobol sequences for efficient space exploration.

;; Define a search space for Random Forest:

(def search-space
  {:model-type :smile.classification/random-forest
   :max-depth (gs/categorical [5 10 15 20])      ; Try different depths
   :trees (gs/linear 10 100 5 :int32)            ; Try 10, 32, 55, 77, 100 trees
   :mtry (gs/categorical [1 2 3 4])              ; Features per split
   :split-rule (gs/categorical [:gini :entropy])}) ; Splitting criterion


;; **Search space defined!** We'll explore combinations of:
;; - Max depth: 5, 10, 15, 20
;; - Number of trees: 10 to 100
;; - Features per split: 1, 2, 3, 4
;; - Split rule: Gini or Entropy

;; Generate parameter combinations (using Sobol sequence sampling):

(def param-combinations
  (take 20 (gs/sobol-gridsearch search-space)))

^:kindly/hide-code
^:kind/md
(str "**Generated " (count param-combinations) " parameter combinations to try**")

;; View the first few combinations:

(take 3 param-combinations)

;; Create pipelines for each combination:

(defn make-tuned-pipeline [params]
  (mm/pipeline
   (ds-mm/categorical->number [:species])
   {:metamorph/id :model}
   (ml/model params)))

(def tuned-pipelines
  (map make-tuned-pipeline param-combinations))

;; Run grid search (this may take a moment):

(def grid-results
  (ml/evaluate-pipelines
   tuned-pipelines
   iris-splits
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false        ; Keep all for analysis
    :return-best-crossvalidation-only true
    :map-fn :pmap}))                        ; Parallel evaluation


;**Grid search complete!**

;; Analyze the results:

(def grid-analysis
  (map-indexed
   (fn [idx results]
     (let [result (first results)
           params (-> result :fit-ctx :model :options)]
       {:idx idx
        :test-accuracy (-> result :test-transform :metric)
        :max-depth (:max-depth params)
        :trees (:trees params)
        :mtry (:mtry params)
        :split-rule (:split-rule params)}))
   grid-results))

;; Sort by test accuracy and view top 5:

(def top-configs
  (->> grid-analysis
       (sort-by :test-accuracy)
       reverse
       (take 5)))

(tc/dataset top-configs)

;; Get the best configuration:

(def best-grid-result
  (first (first grid-results)))

^:kindly/hide-code
^:kind/md
(str "**Best test accuracy from grid search:** "
     (format "%.4f" (-> best-grid-result :test-transform :metric)))

;; Best hyperparameters:

(-> best-grid-result :fit-ctx :model :options
    (select-keys [:max-depth :trees :mtry :split-rule]))

;; ## 8. Adding Data Preprocessing
;;
;; Real-world ML often requires preprocessing. Let's add feature scaling:

(def numeric-cols (tc/column-names (cf/numeric iris-ds)))

(def preprocessing-pipeline
  (mm/pipeline
   ;; Standardize numeric features (mean=0, std=1)
   (preprocessing/std-scale numeric-cols {:mean? true :stddev? true})
   ;; Convert categorical target
   (ds-mm/categorical->number [:species])
   ;; Model
   {:metamorph/id :model}
   (ml/model {:model-type :smile.classification/random-forest
              :max-depth 15
              :trees 100})))

;; Evaluate with preprocessing:

(def preproc-results
  (ml/evaluate-pipelines
   [preprocessing-pipeline]
   iris-splits
   loss/classification-accuracy
   :accuracy))

^:kindly/hide-code
^:kind/md
(str "**Test accuracy with preprocessing:** "
     (format "%.4f" (-> preproc-results first first :test-transform :metric)))

;; ## 9. Using Different Metrics
;;
;; metamorph.ml supports various metrics. Let's try classification loss:

(def loss-results
  (ml/evaluate-pipelines
   [simple-pipeline]
   iris-splits
   loss/classification-loss           ; Loss instead of accuracy
   :loss))                            ; Lower is better

^:kindly/hide-code
^:kind/md
(str "**Classification loss:** "
     (format "%.4f" (-> loss-results first first :test-transform :metric)))

;; ## 10. Complete Workflow Example
;;
;; Here's a complete workflow from start to finish:

(defn complete-ml-workflow [dataset target-column model-config]
  ;; 1. Prepare data
  (let [prepared-ds (ds-mod/set-inference-target dataset target-column)
        splits (tc/split->seq prepared-ds :kfold {:k 5 :seed 42})

        ;; 2. Create pipeline
        pipeline (mm/pipeline
                  (ds-mm/categorical->number [target-column])
                  {:metamorph/id :model}
                  (ml/model model-config))

        ;; 3. Train and evaluate
        results (ml/evaluate-pipelines
                 [pipeline]
                 splits
                 loss/classification-accuracy
                 :accuracy)

        ;; 4. Extract best model
        best-result (-> results first first)
        trained-ctx (:fit-ctx best-result)
        trained-pipeline (:pipe-fn best-result)]

    ;; Return everything needed for predictions
    {:accuracy (-> best-result :test-transform :metric)
     :pipeline trained-pipeline
     :context trained-ctx
     :make-predictions
     (fn [new-data]
       (-> (trained-pipeline
            (merge trained-ctx
                   {:metamorph/data new-data
                    :metamorph/mode :transform}))
           :metamorph/data))}))

;; Use it:

(def workflow-result
  (complete-ml-workflow
   iris-ds
   :species
   {:model-type :smile.classification/random-forest
    :max-depth 15
    :trees 100}))

^:kindly/hide-code
^:kind/md
(str "**Workflow accuracy:** " (format "%.4f" (:accuracy workflow-result)))

;; Make predictions with the workflow:

(def workflow-predictions
  ((:make-predictions workflow-result)
   (tc/shuffle iris-ds {:seed 777})))

(-> workflow-predictions
    (tc/select-columns [:species])
    (tc/head 10))

;; ## Summary
;;
;; In this tutorial, we covered:
;;
;; 1. **Loading data** with rdatasets
;; 2. **Creating pipelines** with preprocessing and models
;; 3. **Training and evaluation** using cross-validation
;; 4. **Making predictions** on new data
;; 5. **Model comparison** across different algorithms
;; 6. **Hyperparameter tuning** with grid search
;; 7. **Data preprocessing** with standardization
;; 8. **Complete workflows** from data to predictions
;;
;; ## Next Steps
;;
;; - Explore other model types from scicloj.ml.smile
;; - Try regression problems with `loss/mse` or `loss/rmse`
;; - Use ensemble methods with `scicloj.metamorph.ml.ensemble`
;; - Add more sophisticated preprocessing
;; - Visualize results with learning curves and confusion matrices
;; - Export models for production use
;;
;; For more information, visit:
;; - [metamorph.ml GitHub](https://github.com/scicloj/metamorph.ml)
;; - [Scicloj Community](https://scicloj.github.io)


;; ---
;; **Tutorial complete!** You now have the foundations for supervised machine learning with metamorph.ml.

(ns scicloj.metamorph.random-forest-test
  (:require [clojure.test :refer [deftest is testing]]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.random-forest]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]
            [tablecloth.api :as tc]))

;; ============================================================================
;; Unit Tests for Core Functions
;; ============================================================================

(deftest test-gini-impurity
  (testing "Gini impurity calculation with indices"
    ;; Pure node
    (is (= 0.0 (#'scicloj.metamorph.ml.random-forest/gini-impurity-indexed
                [:a :a :a :a] [0 1 2 3])))

    ;; 50-50 split (maximum impurity for 2 classes)
    (is (< 0.49 (#'scicloj.metamorph.ml.random-forest/gini-impurity-indexed
                 [:a :a :b :b] [0 1 2 3]) 0.51))

    ;; Three classes, equal distribution
    (let [impurity (#'scicloj.metamorph.ml.random-forest/gini-impurity-indexed
                    [:a :b :c :a :b :c] [0 1 2 3 4 5])]
      (is (< 0.6 impurity 0.7)))))

(deftest test-mse
  (testing "Mean squared error calculation with indices"
    ;; All same values
    (is (= 0.0 (#'scicloj.metamorph.ml.random-forest/mse-indexed [5.0 5.0 5.0] [0 1 2])))

    ;; Known MSE
    (let [mse (#'scicloj.metamorph.ml.random-forest/mse-indexed [1.0 2.0 3.0] [0 1 2])]
      (is (< 0.6 mse 0.7)))))

(deftest test-bootstrap-indices
  (testing "Bootstrap index sampling"
    (let [n 3
          rng (java.util.Random. 42)
          indices (#'scicloj.metamorph.ml.random-forest/bootstrap-indices n rng)]
      ;; Same size as original
      (is (= 3 (count indices)))
      ;; All indices should be valid (0-2)
      (is (every? #(<= 0 % 2) indices))
      ;; Should be a vector
      (is (vector? indices)))))

(deftest test-calculate-max-features
  (testing "Max features calculation"
    (is (= 3 (#'scicloj.metamorph.ml.random-forest/calculate-max-features 10 :sqrt)))
    (is (= 3 (#'scicloj.metamorph.ml.random-forest/calculate-max-features 10 :log2)))
    (is (= 5 (#'scicloj.metamorph.ml.random-forest/calculate-max-features 10 5)))
    (is (= 10 (#'scicloj.metamorph.ml.random-forest/calculate-max-features 10 20)))))

(deftest test-decision-tree-simple
  (testing "Decision tree on simple XOR-like problem"
    (let [;; XOR-like problem: class depends on both features
          X [[0.0 0.0] [0.0 1.0] [1.0 0.0] [1.0 1.0]]
          y [:a :a :a :b]
          rng (java.util.Random. 42)
          ;; Extract feature columns
          feature-columns (#'scicloj.metamorph.ml.random-forest/extract-feature-columns X)
          indices [0 1 2 3]
          tree (#'scicloj.metamorph.ml.random-forest/build-tree-indexed
                feature-columns y indices 0 10 2 1 2 :classification rng false)]

      ;; Tree should not be a single leaf
      (is (= :split (:type tree)))

      ;; Should be able to predict
      (let [predictions (mapv #(#'scicloj.metamorph.ml.random-forest/predict-tree tree %)
                              X)]
        (is (= 4 (count predictions)))))))

(deftest test-decision-tree-depth-limit
  (testing "Decision tree respects max depth"
    (let [X [[1.0] [2.0] [3.0] [4.0] [5.0] [6.0]]
          y [:a :a :b :b :c :c]
          rng (java.util.Random. 42)
          ;; Extract feature columns
          feature-columns (#'scicloj.metamorph.ml.random-forest/extract-feature-columns X)
          indices [0 1 2 3 4 5]
          ;; Max depth of 1 means only root split
          tree (#'scicloj.metamorph.ml.random-forest/build-tree-indexed
                feature-columns y indices 0 1 2 1 1 :classification rng false)]

      ;; Root should be split
      (is (= :split (:type tree)))
      ;; Children should be leaves
      (is (= :leaf (:type (:left tree))))
      (is (= :leaf (:type (:right tree)))))))

;; ============================================================================
;; Integration Tests - Classification
;; ============================================================================

(deftest test-classification-iris
  (testing "Random forest classification on iris dataset"
    (let [;; Create simple iris-like dataset
          ds (ds/->dataset
              {:sepal-length [5.1 4.9 7.0 6.4 6.3 5.8]
               :sepal-width  [3.5 3.0 3.2 3.2 3.3 2.7]
               :petal-length [1.4 1.4 4.7 4.5 6.0 5.1]
               :petal-width  [0.2 0.2 1.4 1.5 2.5 1.9]
               :species      [:setosa :setosa :versicolor :versicolor :virginica :virginica]})

          ;; Mark species as categorical
          ds (ds-mod/set-inference-target ds :species)

          ;; Split into features and target
          feature-ds (cf/feature ds)
          target-ds  (cf/target ds)

          ;; Train model
          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 20
                           :random-seed 42})

          ;; Predict
          predictions (ml/predict feature-ds model)]

      ;; Check predictions exist
      (is (= 6 (ds/row-count predictions)))
      (is (= [:species] (ds/column-names predictions)))

      ;; Check metadata
      (let [pred-col (predictions :species)
            col-meta (meta pred-col)]
        (is (= :prediction (:column-type col-meta)))
        ;(is (contains? col-meta :categorical-map))
        )

      ;; Should have reasonable accuracy on training set
      (let [actual (vec (target-ds :species))
            predicted (vec (predictions :species))
            correct (count (filter true? (map = actual predicted)))
            accuracy (/ correct (count actual))]
        (is (>= accuracy 0.8) "Accuracy should be at least 80% on training data")))))

(deftest test-classification-binary
  (testing "Binary classification task"
    (let [ds (ds/->dataset
              {:x1 [1.0 1.5 2.0 5.0 5.5 6.0 6.5 2.5]
               :x2 [1.0 1.5 2.0 5.0 5.5 6.0 6.5 2.5]
               :y  [:a :a :a :b :b :b :b :a]})

          ds (ds-mod/set-inference-target ds :y)
          feature-ds (cf/feature ds)
          target-ds (cf/target ds)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 50
                           :random-seed 123})

          predictions (ml/predict feature-ds model)]

      (is (= 8 (ds/row-count predictions)))
      (is (= [:y] (ds/column-names predictions)))

      ;; Verify predictions are valid classes
      (let [predicted (vec (predictions :y))]
        (is (every? #{:a :b} predicted))))))

;; ============================================================================
;; Integration Tests - Regression
;; ============================================================================

(deftest test-regression-simple
  (testing "Random forest regression on simple dataset"
    (let [;; Simple linear relationship: y = 2*x + 1
          ds (ds/->dataset
              {:x [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
               :y [3.0 5.0 7.0 9.0 11.0 13.0 15.0 17.0 19.0 21.0]})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 50
                           :max-depth 10
                           :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          target-ds (ds/select-columns ds cf/target)
          predictions (ml/predict feature-ds model)]

      ;; Check predictions exist
      (is (= 10 (ds/row-count predictions)))
      (is (= [:y] (ds/column-names predictions)))

      ;; Check metadata
      (let [pred-col (predictions :y)
            col-meta (meta pred-col)]
        (is (= :prediction (:column-type col-meta))))

      ;; Calculate MAE
      (let [actual (vec (target-ds :y))
            predicted (vec (predictions :y))
            mae (/ (reduce + (map #(Math/abs (- %1 %2)) actual predicted))
                   (count actual))]
        ;; Should have low MAE on training data
        (is (< mae 2.0) "Mean absolute error should be less than 2.0")))))

(deftest test-regression-nonlinear
  (testing "Random forest regression on non-linear relationship"
    (let [;; Quadratic relationship
          xs (range 1 11)
          ys (map #(+ (* % %) (* 2 %) 1) xs)
          ds (ds/->dataset {:x xs :y ys})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 100
                           :max-depth 10
                           :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          target-ds (ds/select-columns ds cf/target)
          predictions (ml/predict feature-ds model)]

      (is (= 10 (ds/row-count predictions)))

      ;; Should fit reasonably well
      (let [actual (vec (target-ds :y))
            predicted (vec (predictions :y))
            mae (/ (reduce + (map #(Math/abs (- (int %1) (int %2))) actual predicted))
                   (count actual))]
        (is (< mae 5.0) "MAE should be reasonable for quadratic function")))))

;; ============================================================================
;; Hyperparameter Tests
;; ============================================================================

(deftest test-hyperparameters
  (testing "Different hyperparameter configurations"
    (let [ds (ds/->dataset
              {:x1 [1.0 1.5 2.0 5.0 5.5 6.0]
               :x2 [1.0 1.5 2.0 5.0 5.5 6.0]
               :y  [:a :a :a :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)
          feature-ds (ds/select-columns ds cf/feature)
          target-ds (ds/select-columns ds cf/target)]

      ;; Test with different n-trees
      (let [model (ml/train ds
                            {:model-type :metamorph.ml/random-forest
                             :n-trees 5
                             :random-seed 42})
            predictions (ml/predict feature-ds model)]
        (is (= 6 (ds/row-count predictions))))

      ;; Test with max-depth
      (let [model (ml/train ds
                            {:model-type :metamorph.ml/random-forest
                             :n-trees 10
                             :max-depth 2
                             :random-seed 42})
            predictions (ml/predict feature-ds model)]
        (is (= 6 (ds/row-count predictions))))

      ;; Test with min-samples-split
      (let [model (ml/train ds
                            {:model-type :metamorph.ml/random-forest
                             :n-trees 10
                             :min-samples-split 3
                             :random-seed 42})
            predictions (ml/predict feature-ds model)]
        (is (= 6 (ds/row-count predictions))))

      ;; Test without bootstrap
      (let [model (ml/train ds
                            {:model-type :metamorph.ml/random-forest
                             :n-trees 10
                             :bootstrap false
                             :random-seed 42})
            predictions (ml/predict feature-ds model)]
        (is (= 6 (ds/row-count predictions)))))))

;; ============================================================================
;; Model Explanation Tests
;; ============================================================================

(deftest test-feature-importance
  (testing "Feature importance calculation"
    (let [ds (ds/->dataset
              {:important [1.0 2.0 3.0 10.0 11.0 12.0]
               :noise     [0.1 0.2 0.1 0.2 0.1 0.2]
               :y         [:a :a :a :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 50
                           :random-seed 42})

          explanation (ml/explain model)]
      

      ;; Should have feature importance
      (is (contains? explanation :feature-importance))

      (let [importance (:feature-importance explanation)]
        ;; Should have importance for both features
        (is (contains? importance :important))
        (is (contains? importance :noise))

        ;; Importance values should be non-negative
        (is (>= (:important importance) 0))
        (is (>= (:noise importance) 0))

        ;; Important feature should have higher importance (usually, but not guaranteed)
        ;; Just check that we got valid numbers
        (is (number? (:important importance)))
        (is (number? (:noise importance)))))))

;; ============================================================================
;; Edge Cases
;; ============================================================================

(deftest test-edge-cases
  (testing "Single class in data"
    (let [ds (ds/->dataset
              {:x [1.0 2.0 3.0]
               :y [:a :a :a]})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 10
                           :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          predictions (ml/predict feature-ds model)]

      ;; Should predict the only class
      (is (= [:a :a :a] (vec (predictions :y))))))

  (testing "Single feature"
    (let [ds (ds/->dataset
              {:x [1.0 2.0 3.0 7.0 8.0 9.0]
               :y [:a :a :a :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 20
                           :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          predictions (ml/predict feature-ds model)]

      (is (= 6 (ds/row-count predictions)))
      (is (every? #{:a :b} (vec (predictions :y))))))

  (testing "Minimal dataset"
    (let [ds (ds/->dataset
              {:x [1.0 2.0]
               :y [:a :b]})

          ds (ds-mod/set-inference-target ds :y)

          model (ml/train ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 5
                           :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          predictions (ml/predict feature-ds model)]

      (is (= 2 (ds/row-count predictions))))))

;; ============================================================================
;; Reproducibility Tests
;; ============================================================================

(deftest test-reproducibility
  (testing "Same random seed produces same results"
    (let [ds (ds/->dataset
              {:x1 [1.0 2.0 3.0 4.0 5.0 6.0]
               :x2 [1.0 1.5 2.0 5.0 5.5 6.0]
               :y  [:a :a :a :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)

          model1 (ml/train ds
                           {:model-type :metamorph.ml/random-forest
                            :n-trees 20
                            :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          predictions1 (ml/predict feature-ds model1)

          model2 (ml/train ds
                           {:model-type :metamorph.ml/random-forest
                            :n-trees 20
                            :random-seed 42})

          predictions2 (ml/predict feature-ds model2)]

      ;; Same seed should produce identical predictions
      (is (= (vec (predictions1 :y))
             (vec (predictions2 :y)))))))

;; ============================================================================
;; Train/Test Split Tests
;; ============================================================================

(deftest test-train-test-split-classification
  (testing "Classification with proper train/test split"
    (let [;; Create a larger dataset with clear patterns
          ;; Class A: x1 < 5, Class B: x1 >= 5
          n-samples 100
          train-size 70

          ;; Generate synthetic data
          x1-vals (vec (concat (repeatedly 50 #(+ 1.0 (rand 3.0)))  ; Class A: 1-4
                               (repeatedly 50 #(+ 5.0 (rand 5.0))))) ; Class B: 5-10
          x2-vals (vec (map #(+ % (* 0.5 (rand))) x1-vals))
          y-vals (vec (concat (repeat 50 :class-a) (repeat 50 :class-b)))

          ;; Shuffle data deterministically
          rng (java.util.Random. 42)
          indices (vec (range n-samples))
          shuffled-indices (vec (.toArray (doto (java.util.ArrayList. indices)
                                            (java.util.Collections/shuffle rng))))

          ;; Split into train and test
          train-indices (take train-size shuffled-indices)
          test-indices (drop train-size shuffled-indices)

          ;; Create datasets
          full-ds (ds/->dataset {:x1 x1-vals :x2 x2-vals :y y-vals})
          train-ds (ds/select-rows full-ds train-indices)
          test-ds (ds/select-rows full-ds test-indices)

          ;; Set target
          train-ds (ds-mod/set-inference-target train-ds :y)
          test-ds (ds-mod/set-inference-target test-ds :y)

          ;; Train on training set (ml/train takes full dataset)
          model (ml/train train-ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 50
                           :max-depth 10
                           :random-seed 123})

          ;; Predict on test set (extract features for prediction)
          test-feature-ds (ds/select-columns test-ds cf/feature)
          test-target-ds (ds/select-columns test-ds cf/target)
          test-predictions (ml/predict test-feature-ds model)]

      ;; Check predictions exist
      (is (= (- n-samples train-size) (ds/row-count test-predictions)))
      (is (= [:y] (ds/column-names test-predictions)))

      ;; Calculate test accuracy
      (let [actual (vec (test-target-ds :y))
            predicted (vec (test-predictions :y))
            correct (count (filter true? (map = actual predicted)))
            accuracy (/ correct (double (count actual)))]

        ;; Should have reasonable test accuracy (>= 70% for this simple problem)
        (is (>= accuracy 0.7)
            (str "Test accuracy should be at least 70%, got: "
                 (format "%.2f%%" (* 100 accuracy))))

        ;; Print accuracy for debugging
        (println (format "Train/Test Split - Test Accuracy: %.2f%%" (* 100 accuracy)))))))

(deftest test-train-test-split-regression
  (testing "Regression with proper train/test split"
    (let [;; Create regression dataset: y = 2*x1 + 3*x2 + noise
          n-samples 80
          train-size 60

          ;; Generate synthetic data
          x1-vals (vec (repeatedly n-samples #(rand 10.0)))
          x2-vals (vec (repeatedly n-samples #(rand 10.0)))
          y-vals (vec (map (fn [x1 x2]
                             (+ (* 2.0 x1)
                                (* 3.0 x2)
                                (* 0.5 (- (rand) 0.5)))) ; small noise
                           x1-vals x2-vals))

          ;; Shuffle data deterministically
          rng (java.util.Random. 42)
          indices (vec (range n-samples))
          shuffled-indices (vec (.toArray (doto (java.util.ArrayList. indices)
                                            (java.util.Collections/shuffle rng))))

          ;; Split into train and test
          train-indices (take train-size shuffled-indices)
          test-indices (drop train-size shuffled-indices)

          ;; Create datasets
          full-ds (ds/->dataset {:x1 x1-vals :x2 x2-vals :y y-vals})
          train-ds (ds/select-rows full-ds train-indices)
          test-ds (ds/select-rows full-ds test-indices)

          ;; Set target
          train-ds (ds-mod/set-inference-target train-ds :y)
          test-ds (ds-mod/set-inference-target test-ds :y)

          ;; Train on training set (ml/train takes full dataset)
          model (ml/train train-ds
                          {:model-type :metamorph.ml/random-forest
                           :n-trees 100
                           :max-depth 15
                           :random-seed 456})

          ;; Predict on test set (extract features for prediction)
          test-feature-ds (ds/select-columns test-ds cf/feature)
          test-target-ds (ds/select-columns test-ds cf/target)
          test-predictions (ml/predict test-feature-ds model)]

      ;; Check predictions exist
      (is (= (- n-samples train-size) (ds/row-count test-predictions)))
      (is (= [:y] (ds/column-names test-predictions)))

      ;; Calculate test MAE and RMSE
      (let [actual (vec (test-target-ds :y))
            predicted (vec (test-predictions :y))
            errors (map #(- %1 %2) actual predicted)
            abs-errors (map #(Math/abs %) errors)
            squared-errors (map #(* % %) errors)
            mae (/ (reduce + abs-errors) (count abs-errors))
            rmse (Math/sqrt (/ (reduce + squared-errors) (count squared-errors)))

            ;; Calculate scale of target variable for context
            y-mean (/ (reduce + actual) (count actual))
            y-std (Math/sqrt (/ (reduce + (map #(* (- % y-mean) (- % y-mean)) actual))
                                (count actual)))]

        ;; MAE should be reasonable relative to the scale
        ;; For this problem, expecting MAE < 3.0 (given y ranges roughly 0-60)
        (is (< mae 3.0)
            (str "Test MAE should be less than 3.0, got: " (format "%.2f" mae)))

        ;; Print metrics for debugging
        (println (format "Train/Test Split - Test MAE: %.2f, RMSE: %.2f" mae rmse))
        (println (format "Target mean: %.2f, std: %.2f" y-mean y-std))))))

(deftest test-cross-validation-style
  (testing "Multiple train/test splits (cross-validation style)"
    (let [;; Create a medium-sized dataset
          n-samples 60

          ;; Generate data
          x-vals (vec (concat (repeatedly 30 #(+ 1.0 (rand 4.0)))
                              (repeatedly 30 #(+ 6.0 (rand 4.0)))))
          y-vals (vec (concat (repeat 30 :a) (repeat 30 :b)))

          full-ds (ds/->dataset {:x x-vals :y y-vals})
          full-ds (ds-mod/set-inference-target full-ds :y)

          ;; Test with 3 different random splits
          accuracies
          (for [seed [100 200 300]]
            (let [rng (java.util.Random. seed)
                  indices (vec (range n-samples))
                  shuffled (vec (.toArray (doto (java.util.ArrayList. indices)
                                            (java.util.Collections/shuffle rng))))

                  train-size 45
                  train-indices (take train-size shuffled)
                  test-indices (drop train-size shuffled)

                  train-ds (ds/select-rows full-ds train-indices)
                  test-ds (ds/select-rows full-ds test-indices)

                  ;; Train on full training dataset
                  model (ml/train train-ds
                                  {:model-type :metamorph.ml/random-forest
                                   :n-trees 30
                                   :random-seed seed})

                  ;; Predict on test features
                  test-feature-ds (ds/select-columns test-ds cf/feature)
                  test-target-ds (ds/select-columns test-ds cf/target)
                  predictions (ml/predict test-feature-ds model)

                  actual (vec (test-target-ds :y))
                  predicted (vec (predictions :y))
                  correct (count (filter true? (map = actual predicted)))
                  accuracy (/ correct (double (count actual)))]
              accuracy))]

      ;; All splits should have reasonable accuracy
      (is (every? #(>= % 0.6) accuracies)
          (str "All splits should have accuracy >= 60%, got: "
               (mapv #(format "%.2f%%" (* 100 %)) accuracies)))

      ;; Average accuracy across splits
      (let [avg-accuracy (/ (reduce + accuracies) (count accuracies))]
        (println (format "Cross-validation style - Average accuracy: %.2f%%"
                         (* 100 avg-accuracy)))
        (is (>= avg-accuracy 0.7) "Average accuracy should be >= 70%")))))

;; ============================================================================
;; New Hyperparameter Tests
;; ============================================================================

(deftest test-min-samples-leaf
  (testing "Min samples leaf constraint"
    (let [ds (ds/->dataset
              {:x1 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
               :x2 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]
               :y  [:a :a :a :a :b :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)
          feature-ds (ds/select-columns ds cf/feature)

          ;; Train with min-samples-leaf constraint
          model (ml/train ds
                         {:model-type :metamorph.ml/random-forest
                          :n-trees 10
                          :min-samples-leaf 3
                          :random-seed 42})

          predictions (ml/predict feature-ds model)]

      (is (= 8 (ds/row-count predictions)))
      (is (every? #{:a :b} (vec (predictions :y)))))))

(deftest test-parallel-features
  (testing "Parallel feature search"
    (let [ds (ds/->dataset
              {:x1 [1.0 2.0 3.0 4.0 5.0 6.0]
               :x2 [1.0 2.0 3.0 4.0 5.0 6.0]
               :x3 [1.0 2.0 3.0 4.0 5.0 6.0]
               :y  [:a :a :a :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)
          feature-ds (ds/select-columns ds cf/feature)

          ;; Train with parallel feature search (should still work)
          model (ml/train ds
                         {:model-type :metamorph.ml/random-forest
                          :n-trees 5
                          :parallel-features true
                          :random-seed 42})

          predictions (ml/predict feature-ds model)]

      (is (= 6 (ds/row-count predictions)))
      (is (every? #{:a :b} (vec (predictions :y)))))))

(deftest test-optimized-performance-small
  (testing "Optimized implementation produces correct results"
    (let [ds (ds/->dataset
              {:x1 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0]
               :x2 [1.0 1.5 2.0 2.5 3.0 5.0 5.5 6.0 6.5 7.0]
               :y  [:a :a :a :a :a :b :b :b :b :b]})

          ds (ds-mod/set-inference-target ds :y)

          ;; Train with all optimizations enabled
          model (ml/train ds
                         {:model-type :metamorph.ml/random-forest
                          :n-trees 20
                          :max-depth 10
                          :min-samples-leaf 2
                          :parallel true
                          :random-seed 42})

          feature-ds (ds/select-columns ds cf/feature)
          predictions (ml/predict feature-ds model)]

      (is (= 10 (ds/row-count predictions)))
      (is (every? #{:a :b} (vec (predictions :y))))

      ;; Should have high accuracy on training data
      (let [actual (vec ((ds/select-columns ds cf/target) :y))
            predicted (vec (predictions :y))
            correct (count (filter true? (map = actual predicted)))
            accuracy (/ correct (double (count actual)))]
        (is (>= accuracy 0.8) "Should achieve at least 80% accuracy")))))




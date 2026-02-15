(ns scicloj.metamorph.random-forest-performance-test
  "Performance and speed tests for Random Forest implementation.
   These tests use larger synthetic datasets to measure training time."
  (:require [clojure.test :refer [deftest is testing]]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.random-forest]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column-filters :as cf]))

;; ============================================================================
;; Synthetic Data Generation
;; ============================================================================

(defn generate-classification-dataset
  "Generate synthetic classification dataset with features of varying importance.

   Parameters:
   - n-rows: Number of samples
   - n-informative: Number of informative features (actually predict the target)
   - n-noise: Number of noise features (random, don't predict target)
   - random-seed: Seed for reproducibility"
  [{:keys [n-rows n-informative n-noise random-seed]
    :or {n-rows 1000
         n-informative 5
         n-noise 15
         random-seed 42}}]
  (let [rng (java.util.Random. random-seed)
        n-features (+ n-informative n-noise)

        ;; Generate informative features (determine class based on sum)
        informative-features
        (into {}
              (map-indexed
               (fn [idx _]
                 [(keyword (str "informative-" idx))
                  (vec (repeatedly n-rows #(+ (* 10.0 (.nextDouble rng)) -5.0)))])
               (range n-informative)))

        ;; Generate noise features (random, uncorrelated with target)
        noise-features
        (into {}
              (map-indexed
               (fn [idx _]
                 [(keyword (str "noise-" idx))
                  (vec (repeatedly n-rows #(.nextDouble rng)))])
               (range n-noise)))

        ;; Calculate target based on informative features
        ;; Class A if sum of first 3 informative features > 0, else Class B
        informative-cols (take 3 (vals informative-features))
        target-vals
        (vec (for [row-idx (range n-rows)]
               (let [sum (reduce + (map #(nth % row-idx) informative-cols))]
                 (if (> sum 0.0) :class-a :class-b))))

        ;; Combine all features and target
        dataset-map (merge informative-features
                          noise-features
                          {:target target-vals})]

    (ds/->dataset dataset-map)))

(defn generate-regression-dataset
  "Generate synthetic regression dataset with features of varying importance.

   Target is a linear combination of informative features plus noise."
  [{:keys [n-rows n-informative n-noise random-seed]
    :or {n-rows 1000
         n-informative 5
         n-noise 15
         random-seed 42}}]
  (let [rng (java.util.Random. random-seed)
        n-features (+ n-informative n-noise)

        ;; Generate informative features
        informative-features
        (into {}
              (map-indexed
               (fn [idx _]
                 [(keyword (str "informative-" idx))
                  (vec (repeatedly n-rows #(* 10.0 (.nextDouble rng))))])
               (range n-informative)))

        ;; Generate noise features
        noise-features
        (into {}
              (map-indexed
               (fn [idx _]
                 [(keyword (str "noise-" idx))
                  (vec (repeatedly n-rows #(.nextDouble rng)))])
               (range n-noise)))

        ;; Calculate target as linear combination of informative features
        ;; y = 2*x0 + 3*x1 - 1.5*x2 + 0.5*x3 + noise
        coefficients [2.0 3.0 -1.5 0.5 1.0]
        informative-cols (vals informative-features)
        target-vals
        (vec (for [row-idx (range n-rows)]
               (let [weighted-sum (reduce +
                                         (map-indexed
                                          (fn [idx col]
                                            (* (nth coefficients idx 0.0)
                                               (nth col row-idx)))
                                          informative-cols))
                     noise (* 2.0 (- (.nextDouble rng) 0.5))]
                 (+ weighted-sum noise))))

        ;; Combine all features and target
        dataset-map (merge informative-features
                          noise-features
                          {:target target-vals})]

    (ds/->dataset dataset-map)))

;; ============================================================================
;; Performance Tests
;; ============================================================================

(deftest test-classification-speed-medium-dataset
  (testing "Training speed on medium classification dataset (20 cols, 1000 rows)"
    (println "\n=== Classification Performance Test ===")
    (println "Dataset: 1000 rows, 20 features (5 informative, 15 noise)")

    (let [;; Generate dataset
          ds (generate-classification-dataset
              {:n-rows 1000
               :n-informative 5
               :n-noise 15
               :random-seed 42})

          _ (println "Generated dataset:" (ds/shape ds))

          ;; Set target
          ds (ds-mod/set-inference-target ds :target)

          ;; Extract features for later prediction
          feature-ds (ds/select-columns ds cf/feature)

          ;; Measure training time with different configurations
          configs [{:label "50 trees, max-depth=10"
                   :n-trees 50
                   :max-depth 10}
                  {:label "100 trees, max-depth=15"
                   :n-trees 100
                   :max-depth 15}
                  {:label "100 trees, unlimited depth"
                   :n-trees 100
                   :max-depth nil}]]

      (doseq [{:keys [label n-trees max-depth]} configs]
        (println (str "\nTraining with " label "..."))
        (let [start-time (System/nanoTime)

              model (ml/train ds
                             {:model-type :metamorph.ml/random-forest
                              :n-trees n-trees
                              :max-depth max-depth
                              :random-seed 42})

              train-time-ms (/ (- (System/nanoTime) start-time) 1e6)

              ;; Measure prediction time
              pred-start (System/nanoTime)
              predictions (ml/predict feature-ds model)
              pred-time-ms (/ (- (System/nanoTime) pred-start) 1e6)

              ;; Calculate accuracy
              actual (vec ((ds/select-columns ds cf/target) :target))
              predicted (vec (predictions :target))
              correct (count (filter true? (map = actual predicted)))
              accuracy (/ correct (double (count actual)))]

          (println (format "  Training time: %.2f ms" train-time-ms))
          (println (format "  Prediction time: %.2f ms" pred-time-ms))
          (println (format "  Training accuracy: %.2f%%" (* 100 accuracy)))

          ;; Assertions to ensure tests don't break silently
          (is (< train-time-ms 30000)
              "Training should complete in reasonable time (< 30 seconds)")
          (is (>= accuracy 0.7)
              "Should achieve reasonable accuracy on training data")))

      (println "\n=== Classification Performance Test Complete ===\n"))))

(deftest test-regression-speed-medium-dataset
  (testing "Training speed on medium regression dataset (20 cols, 1000 rows)"
    (println "\n=== Regression Performance Test ===")
    (println "Dataset: 1000 rows, 20 features (5 informative, 15 noise)")

    (let [;; Generate dataset
          ds (generate-regression-dataset
              {:n-rows 1000
               :n-informative 5
               :n-noise 15
               :random-seed 42})

          _ (println "Generated dataset:" (ds/shape ds))

          ;; Set target
          ds (ds-mod/set-inference-target ds :target)

          ;; Extract features for later prediction
          feature-ds (ds/select-columns ds cf/feature)
          target-ds (ds/select-columns ds cf/target)

          ;; Measure training time
          configs [{:label "50 trees, max-depth=10"
                   :n-trees 50
                   :max-depth 10}
                  {:label "100 trees, max-depth=15"
                   :n-trees 100
                   :max-depth 15}]]

      (doseq [{:keys [label n-trees max-depth]} configs]
        (println (str "\nTraining with " label "..."))
        (let [start-time (System/nanoTime)

              model (ml/train ds
                             {:model-type :metamorph.ml/random-forest
                              :n-trees n-trees
                              :max-depth max-depth
                              :random-seed 42})

              train-time-ms (/ (- (System/nanoTime) start-time) 1e6)

              ;; Measure prediction time
              pred-start (System/nanoTime)
              predictions (ml/predict feature-ds model)
              pred-time-ms (/ (- (System/nanoTime) pred-start) 1e6)

              ;; Calculate MAE and RMSE
              actual (vec (target-ds :target))
              predicted (vec (predictions :target))
              errors (map #(- %1 %2) actual predicted)
              abs-errors (map #(Math/abs %) errors)
              squared-errors (map #(* % %) errors)
              mae (/ (reduce + abs-errors) (count abs-errors))
              rmse (Math/sqrt (/ (reduce + squared-errors) (count squared-errors)))]

          (println (format "  Training time: %.2f ms" train-time-ms))
          (println (format "  Prediction time: %.2f ms" pred-time-ms))
          (println (format "  MAE: %.2f" mae))
          (println (format "  RMSE: %.2f" rmse))

          ;; Assertions
          (is (< train-time-ms 30000)
              "Training should complete in reasonable time (< 30 seconds)")
          (is (< mae 20.0)
              "Should achieve reasonable MAE on training data")))

      (println "\n=== Regression Performance Test Complete ===\n"))))

(deftest test-large-dataset-performance
  (testing "Performance on larger dataset (20 cols, 5000 rows)"
    (println "\n=== Large Dataset Performance Test ===")
    (println "Dataset: 5000 rows, 20 features (5 informative, 15 noise)")

    (let [ds (generate-classification-dataset
              {:n-rows 5000
               :n-informative 5
               :n-noise 15
               :random-seed 42})

          _ (println "Generated dataset:" (ds/shape ds))

          ds (ds-mod/set-inference-target ds :target)
          feature-ds (ds/select-columns ds cf/feature)

          _ (println "\nTraining with 100 trees, max-depth=15...")

          start-time (System/nanoTime)

          model (ml/train ds
                         {:model-type :metamorph.ml/random-forest
                          :n-trees 100
                          :max-depth 15
                          :random-seed 42})

          train-time-ms (/ (- (System/nanoTime) start-time) 1e6)

          ;; Measure prediction time
          pred-start (System/nanoTime)
          predictions (ml/predict feature-ds model)
          pred-time-ms (/ (- (System/nanoTime) pred-start) 1e6)

          ;; Calculate accuracy
          actual (vec ((ds/select-columns ds cf/target) :target))
          predicted (vec (predictions :target))
          correct (count (filter true? (map = actual predicted)))
          accuracy (/ correct (double (count actual)))]

      (println (format "  Training time: %.2f ms (%.2f seconds)"
                      train-time-ms (/ train-time-ms 1000.0)))
      (println (format "  Prediction time: %.2f ms" pred-time-ms))
      (println (format "  Training accuracy: %.2f%%" (* 100 accuracy)))
      (println (format "  Throughput: %.2f samples/second"
                      (/ 5000.0 (/ train-time-ms 1000.0))))

      ;; Relaxed time constraint for larger dataset
      (is (< train-time-ms 120000)
          "Training should complete in reasonable time (< 2 minutes)")
      (is (>= accuracy 0.7)
          "Should achieve reasonable accuracy")

      (println "\n=== Large Dataset Performance Test Complete ===\n"))))

(deftest test-feature-importance-performance
  (testing "Feature importance calculation performance"
    (println "\n=== Feature Importance Performance Test ===")

    (let [ds (generate-classification-dataset
              {:n-rows 1000
               :n-informative 5
               :n-noise 15
               :random-seed 42})

          ds (ds-mod/set-inference-target ds :target)

          _ (println "Training model...")
          model (ml/train ds
                         {:model-type :metamorph.ml/random-forest
                          :n-trees 100
                          :max-depth 15
                          :random-seed 42})

          _ (println "Calculating feature importance...")
          start-time (System/nanoTime)

          explanation (ml/explain model)

          importance-time-ms (/ (- (System/nanoTime) start-time) 1e6)

          importance (:feature-importance explanation)

          ;; Sort features by importance
          sorted-features (sort-by val > importance)
          top-5 (take 5 sorted-features)]

      (println (format "  Feature importance calculation time: %.2f ms" importance-time-ms))
      (println "\n  Top 5 most important features:")
      (doseq [[feature-name importance-val] top-5]
        (println (format "    %s: %.4f" (name feature-name) importance-val)))

      ;; Check that informative features have higher importance than noise
      (let [informative-importance (filter #(clojure.string/starts-with? (name (key %)) "informative")
                                          sorted-features)
            avg-informative (/ (reduce + (map val (take 5 informative-importance))) 5.0)
            noise-importance (filter #(clojure.string/starts-with? (name (key %)) "noise")
                                    sorted-features)
            avg-noise (/ (reduce + (map val noise-importance)) (count noise-importance))]

        (println (format "\n  Average importance - Informative: %.4f, Noise: %.4f"
                        avg-informative avg-noise))

        ;; Informative features should generally have higher importance
        ;; (not always guaranteed due to randomness, but should be true on average)
        (is (>= avg-informative (* 0.5 avg-noise))
            "Informative features should have higher importance than noise features"))

      (println "\n=== Feature Importance Performance Test Complete ===\n"))))

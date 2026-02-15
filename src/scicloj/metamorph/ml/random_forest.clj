(ns scicloj.metamorph.ml.random-forest
  "Optimized Pure Clojure Random Forest implementation for classification and regression."
  (:require [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]))

;; ============================================================================
;; Helper Functions
;; ============================================================================

(defn- calculate-max-features
  "Calculate the number of features to consider at each split."
  [n-features max-features]
  (cond
    (= max-features :sqrt) (int (Math/floor (Math/sqrt n-features)))
    (= max-features :log2) (int (Math/floor (/ (Math/log n-features) (Math/log 2))))
    (number? max-features) (min max-features n-features)
    :else n-features))

(defn- my-shuffle
  "Return a random permutation of coll"
  [^java.util.Collection coll ^java.util.Random rng]
  (let [al (java.util.ArrayList. coll)]
    (java.util.Collections/shuffle al rng)
    (clojure.lang.RT/vector (.toArray al))))

(defn- random-sample
  "Sample k random indices from 0 to n-1."
  [n k rng]
  (take k (my-shuffle (range n) rng)))

;; ============================================================================
;; Data Structure Optimizations
;; ============================================================================

(defn- extract-feature-columns
  "Extract features as column-major format for faster access.
   Returns vector of columns where each column is a vector of values."
  [X]
  (let [n-features (count (first X))]
    (mapv (fn [feature-idx]
            (mapv #(double (nth % feature-idx)) X))
          (range n-features))))

(defn- get-feature-value
  "Fast feature value lookup from column-major format."
  [feature-columns feature-idx sample-idx]
  (nth (nth feature-columns feature-idx) sample-idx))

;; ============================================================================
;; Impurity and Split Metrics
;; ============================================================================

(defn- gini-impurity-indexed
  "Calculate Gini impurity for classification using indices (int array)."
  [y ^ints indices]
  (let [n (alength indices)]
    (if (zero? n)
      0.0
      (let [class-counts (loop [i 0
                                counts (transient {})]
                           (if (< i n)
                             (let [idx (aget indices i)
                                   class (nth y idx)]
                               (recur (inc i)
                                      (assoc! counts class (inc (get counts class 0)))))
                             (persistent! counts)))
            n-double (double n)
            sum-squares (reduce (fn [^double acc [_ cnt]]
                                 (let [p (/ (double cnt) n-double)]
                                   (+ acc (* p p))))
                               0.0
                               class-counts)]
        (- 1.0 sum-squares)))))

(defn- mse-indexed
  "Calculate mean squared error for regression using indices (int array)."
  [y ^ints indices]
  (let [n (alength indices)]
    (if (zero? n)
      0.0
      (let [n-double (double n)
            sum (loop [i 0 s 0.0]
                  (if (< i n)
                    (recur (inc i) (+ s (double (nth y (aget indices i)))))
                    s))
            mean (/ sum n-double)
            sum-squared-diffs (loop [i 0 ss 0.0]
                                (if (< i n)
                                  (let [val (double (nth y (aget indices i)))
                                        diff (- val mean)]
                                    (recur (inc i) (+ ss (* diff diff))))
                                  ss))]
        (/ sum-squared-diffs n-double)))))

(defn- calculate-impurity-indexed
  "Calculate impurity based on task type using indices."
  [y indices task]
  (case task
    :classification (gini-impurity-indexed y indices)
    :regression (mse-indexed y indices)))

;; ============================================================================
;; Optimized Splitting Logic
;; ============================================================================

(defn- split-indices
  "Split indices based on feature and threshold (returns int arrays).
   Works with column-major feature data."
  [feature-columns feature-idx threshold ^ints indices]
  (let [thresh (double threshold)
        n (alength indices)
        ;; First pass: count left and right
        counts (loop [i 0 n-left 0]
                 (if (< i n)
                   (let [idx (aget indices i)
                         val (get-feature-value feature-columns feature-idx idx)]
                     (recur (inc i) (if (<= val thresh) (inc n-left) n-left)))
                   n-left))
        n-left counts
        n-right (- n n-left)
        ;; Allocate arrays
        left (int-array n-left)
        right (int-array n-right)
        ;; Second pass: fill arrays
        _ (loop [i 0 left-pos 0 right-pos 0]
            (when (< i n)
              (let [idx (aget indices i)
                    val (get-feature-value feature-columns feature-idx idx)]
                (if (<= val thresh)
                  (do (aset left left-pos idx)
                      (recur (inc i) (inc left-pos) right-pos))
                  (do (aset right right-pos idx)
                      (recur (inc i) left-pos (inc right-pos)))))))]
    {:left left :right right}))

(defn- calculate-split-gain-indexed
  "Calculate information gain from a split using indices."
  [y indices left-indices right-indices task]
  (let [n (count indices)
        n-left (count left-indices)
        n-right (count right-indices)]
    (if (or (zero? n-left) (zero? n-right))
      0.0
      (let [parent-impurity (calculate-impurity-indexed y indices task)
            left-impurity (calculate-impurity-indexed y left-indices task)
            right-impurity (calculate-impurity-indexed y right-indices task)
            n-double (double n)
            left-weight (/ (double n-left) n-double)
            right-weight (/ (double n-right) n-double)
            weighted-child-impurity (+ (* left-weight left-impurity)
                                       (* right-weight right-impurity))]
        (- parent-impurity weighted-child-impurity)))))

(defn- get-thresholds-fast
  "Get candidate thresholds (max 10 for speed).
   Uses column-major feature data and int array indices."
  [feature-columns feature-idx ^ints indices]
  (let [n (alength indices)
        ;; Collect unique values in a set, then sort into a vector
        unique-vals (loop [i 0 vals (transient #{})]
                      (if (< i n)
                        (recur (inc i)
                               (conj! vals (get-feature-value feature-columns
                                                             feature-idx
                                                             (aget indices i))))
                        (persistent! vals)))
        values (vec (sort unique-vals))
        n-values (count values)
        max-thresholds 10]  ; Reduced from 20 for speed
    (cond
      (< n-values 2) []

      (<= n-values max-thresholds)
      ;; Few values: use all midpoints
      (loop [i 0
             result (transient [])]
        (if (< i (dec n-values))
          (let [a (double (nth values i))
                b (double (nth values (inc i)))]
            (recur (inc i)
                   (conj! result (* 0.5 (+ a b)))))
          (persistent! result)))

      :else
      ;; Many values: sample uniformly
      (let [step (/ (double (dec n-values)) (double max-thresholds))]
        (loop [i 0
               result (transient [])]
          (if (< i max-thresholds)
            (let [idx (long (* (double i) step))
                  idx-next (min (inc idx) (dec n-values))
                  a (double (nth values idx))
                  b (double (nth values idx-next))]
              (recur (inc i)
                     (conj! result (* 0.5 (+ a b)))))
            (persistent! result)))))))

(defn- find-best-split-for-feature
  "Find best split for a single feature."
  [feature-columns y indices feature-idx task]
  (let [thresholds (get-thresholds-fast feature-columns feature-idx indices)]
    (loop [thresholds thresholds
           best-gain 0.0
           best-threshold nil]
      (if (empty? thresholds)
        {:gain best-gain :threshold best-threshold}
        (let [thresh (first thresholds)
              {:keys [left right]} (split-indices feature-columns feature-idx thresh indices)
              gain (calculate-split-gain-indexed y indices left right task)]
          (if (> gain best-gain)
            ;; Early stopping: if gain > 0.99, we have a near-perfect split
            (if (>= gain 0.99)
              {:gain gain :threshold thresh}
              (recur (rest thresholds) gain thresh))
            (recur (rest thresholds) best-gain best-threshold)))))))

(defn- find-best-split
  "Find the best feature and threshold to split on.
   Optionally parallelizes feature search."
  [feature-columns y indices feature-indices task parallel-features?]
  (let [find-fn (fn [feature-idx]
                 (let [{:keys [gain threshold]} (find-best-split-for-feature feature-columns y indices feature-idx task)]
                   {:feature-idx feature-idx :gain gain :threshold threshold}))
        results (if parallel-features?
                  (pmap find-fn feature-indices)
                  (map find-fn feature-indices))
        best (reduce (fn [best current]
                      (if (> (:gain current) (:gain best))
                        current
                        best))
                    {:feature-idx nil :gain 0.0 :threshold nil}
                    results)]
    best))

;; ============================================================================
;; Decision Tree Construction
;; ============================================================================

(defn- create-leaf-indexed
  "Create a leaf node with prediction value from indices (int array)."
  [y ^ints indices task]
  (let [n-samples (alength indices)]
    (case task
      :classification
      (let [class-counts (loop [i 0
                                counts (transient {})]
                           (if (< i n-samples)
                             (let [class (nth y (aget indices i))]
                               (recur (inc i)
                                      (assoc! counts class (inc (get counts class 0)))))
                             (persistent! counts)))
            majority-class (key (apply max-key val class-counts))]
        {:type :leaf
         :value majority-class
         :class-counts class-counts
         :n-samples n-samples})

      :regression
      (let [sum (loop [i 0 s 0.0]
                  (if (< i n-samples)
                    (recur (inc i) (+ s (double (nth y (aget indices i)))))
                    s))
            mean (/ sum (double (max 1 n-samples)))]
        {:type :leaf
         :value mean
         :n-samples n-samples}))))

(defn- should-stop-split?
  "Check if we should stop splitting (works with int array indices)."
  [y ^ints indices depth max-depth min-samples-split min-samples-leaf task]
  (let [n-samples (alength indices)]
    (or
     ;; Reached max depth
     (and max-depth (>= depth max-depth))
     ;; Too few samples to split
     (< n-samples min-samples-split)
     ;; Would create too-small leaves
     (and min-samples-leaf (< n-samples (* 2 min-samples-leaf)))
     ;; Pure node (classification only)
     (and (= task :classification)
          (or (<= n-samples 1)
              (= 1 (count (loop [i 0 classes (transient #{})]
                            (if (< i n-samples)
                              (recur (inc i) (conj! classes (nth y (aget indices i))))
                              (persistent! classes))))))))))

(defn- build-tree-indexed
  "Recursively build a decision tree using indices (avoids data copying)."
  [feature-columns y indices depth max-depth min-samples-split min-samples-leaf
   max-features-per-split task rng parallel-features?]
  (let [n-samples (count indices)
        n-features (count feature-columns)]

    ;; Check stopping criteria
    (if (should-stop-split? y indices depth max-depth min-samples-split min-samples-leaf task)
      (create-leaf-indexed y indices task)

      ;; Try to split
      (let [;; Randomly select features to consider
            feature-indices (if max-features-per-split
                              (random-sample n-features max-features-per-split rng)
                              (range n-features))
            {:keys [feature-idx threshold gain]} (find-best-split feature-columns y indices
                                                                  feature-indices task parallel-features?)]

        ;; If no valid split found, create leaf
        (if (or (nil? feature-idx) (<= gain 0.0))
          (create-leaf-indexed y indices task)

          ;; Create split node
          (let [{:keys [left right]} (split-indices feature-columns feature-idx threshold indices)
                left-child (build-tree-indexed feature-columns y left (inc depth) max-depth
                                              min-samples-split min-samples-leaf max-features-per-split
                                              task rng parallel-features?)
                right-child (build-tree-indexed feature-columns y right (inc depth) max-depth
                                               min-samples-split min-samples-leaf max-features-per-split
                                               task rng parallel-features?)]
            {:type :split
             :feature-idx feature-idx
             :threshold threshold
             :left left-child
             :right right-child
             :n-samples n-samples}))))))

;; ============================================================================
;; Prediction
;; ============================================================================

(defn- predict-tree
  "Predict a single sample using a decision tree.
   Sample should be a vector of feature values."
  [tree sample]
  (loop [node tree]
    (case (:type node)
      :leaf (:value node)
      :split (let [{:keys [feature-idx threshold left right]} node
                   feature-value (double (nth sample feature-idx))
                   thresh (double threshold)]
               (if (<= feature-value thresh)
                 (recur left)
                 (recur right))))))

(defn- predict-forest-batched
  "Predict using all trees with batched processing for better cache locality."
  [trees X task]
  (let [n-trees (count trees)
        n-samples (count X)]

    (case task
      :classification
      ;; Batch prediction: predict all samples with each tree, then aggregate
      (let [;; Get predictions from all trees (trees × samples matrix)
            all-predictions (mapv (fn [tree]
                                   (mapv #(predict-tree tree %) X))
                                 trees)]
        ;; Aggregate per sample with majority voting
        (mapv (fn [sample-idx]
                (let [predictions (mapv #(nth % sample-idx) all-predictions)
                      vote-counts (frequencies predictions)]
                  (key (apply max-key val vote-counts))))
              (range n-samples)))

      :regression
      ;; Average predictions per sample
      (let [all-predictions (mapv (fn [tree]
                                   (mapv #(predict-tree tree %) X))
                                 trees)]
        (mapv (fn [sample-idx]
                (let [predictions (mapv #(nth % sample-idx) all-predictions)
                      sum (reduce + predictions)]
                  (/ sum (double n-trees))))
              (range n-samples))))))

;; ============================================================================
;; Random Forest
;; ============================================================================

(defn- bootstrap-indices
  "Create bootstrap sample indices as int array (sample with replacement).
   Returns int array for performance."
  [n ^java.util.Random rng]
  (let [arr (int-array n)]
    (dotimes [i n]
      (aset arr i (.nextInt rng n)))
    arr))

(defn- train-single-tree
  "Train a single decision tree on bootstrap sample using indices (int arrays)."
  [feature-columns y max-depth min-samples-split min-samples-leaf max-features-per-split
   bootstrap? parallel-features? task rng]
  (let [n (count y)
        indices (if bootstrap?
                  (bootstrap-indices n rng)
                  (let [arr (int-array n)]
                    (dotimes [i n]
                      (aset arr i i))
                    arr))]
    (build-tree-indexed feature-columns y indices 0 max-depth min-samples-split min-samples-leaf
                       max-features-per-split task rng parallel-features?)))

(defn- train-forest
  "Train multiple decision trees to form a random forest.
   Trees are trained in parallel using pmap for better performance."
  [X y {:keys [n-trees max-depth min-samples-split min-samples-leaf max-features
               bootstrap random-seed parallel parallel-features]
        :or {n-trees 100
             max-depth nil
             min-samples-split 2
             min-samples-leaf 1
             max-features :sqrt
             bootstrap true
             random-seed nil
             parallel true
             parallel-features false}}
   task]
  (let [n-features (count (first X))
        n-samples (count X)

        ;; Pre-compute feature columns once (major optimization)
        feature-columns (extract-feature-columns X)

        max-features-per-split (calculate-max-features n-features max-features)
        base-rng (if random-seed
                   (java.util.Random. random-seed)
                   (java.util.Random.))
        ;; Generate seeds for each tree to ensure reproducibility
        tree-seeds (vec (repeatedly n-trees #(.nextInt base-rng)))

        ;; Function to train a single tree with its own RNG
        train-fn (fn [tree-seed]
                   (let [rng (java.util.Random. tree-seed)]
                     (train-single-tree feature-columns y max-depth min-samples-split min-samples-leaf
                                       max-features-per-split bootstrap parallel-features task rng)))]

    ;; Use pmap for parallel execution (unless disabled)
    (if parallel
      (vec (pmap train-fn tree-seeds))
      (mapv train-fn tree-seeds))))

;; ============================================================================
;; metamorph.ml Integration
;; ============================================================================

(defn- train-random-forest
  "Train a random forest model following metamorph.ml conventions."
  [feature-ds target-ds options]
  (let [;; Extract feature matrix as row vectors
        X (vec (ds/rowvecs feature-ds))

        ;; Extract target values
        target-col (first (ds/column-names target-ds))
        y (vec (target-ds target-col))

        ;; Determine task type
        target-meta (meta (target-ds target-col))
        categorical? (:categorical? target-meta)
        task (if categorical? :classification :regression)

        ;; Get categorical map if classification
        categorical-map (when categorical?
                          (:categorical-map target-meta))

        ;; Train forest
        trees (train-forest X y options task)]

    {:forest {:trees trees
              :n-trees (count trees)
              :task task
              :feature-names (vec (ds/column-names feature-ds))
              :target-column target-col
              :categorical-map categorical-map}}))

(defn- predict-random-forest
  "Predict using a trained random forest model."
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps]}]
  (let [;; Extract feature matrix
        X (vec (ds/rowvecs feature-ds))

        ;; Get model components
        {:keys [trees task target-column categorical-map]} (:forest thawed-model)

        ;; Make predictions with batched processing
        predictions (predict-forest-batched trees X task)

        ;; Use target-column from training, or fall back to target-columns from options
        pred-col-name (or target-column (first target-columns))

        ;; Use categorical-map from training, or fall back to target-categorical-maps
        cat-map (or categorical-map (get target-categorical-maps pred-col-name))]

    ;; Create prediction dataset with proper metadata
    (ds/new-dataset
     [(ds/new-column
       pred-col-name
       predictions
       (merge {:column-type :prediction}
              (when cat-map
                {:categorical-map cat-map})))])))

;; ============================================================================
;; Optional: Model Explanation Functions
;; ============================================================================

(defn- calculate-feature-importance
  "Calculate feature importance based on mean decrease in impurity."
  [trees feature-names]
  (let [n-features (count feature-names)
        importance (atom (vec (repeat n-features 0.0)))

        ;; Helper to traverse tree and accumulate importance
        traverse-tree
        (fn traverse [node]
          (when (= (:type node) :split)
            (let [feature-idx (:feature-idx node)
                  n-samples (:n-samples node)
                  left (:left node)
                  right (:right node)
                  ;; Importance contribution is weighted by samples
                  contribution (* n-samples 1.0)]
              (swap! importance update feature-idx + contribution)
              (traverse left)
              (traverse right))))]

    ;; Accumulate importance across all trees
    (doseq [tree trees]
      (traverse-tree tree))

    ;; Normalize by number of trees
    (let [total (reduce + @importance)
          normalized (if (> total 0)
                      (mapv #(/ % total) @importance)
                      @importance)]
      (zipmap feature-names normalized))))

(defn- explain-random-forest
  "Provide feature importance for the trained model."
  [thawed-model model options]
  (let [{:keys [trees feature-names]} (:forest thawed-model)]
    {:feature-importance (calculate-feature-importance trees feature-names)}))

;; ============================================================================
;; Model Registration
;; ============================================================================

(ml/define-model! :metamorph.ml/random-forest
  train-random-forest
  predict-random-forest
  {:hyperparameters {:n-trees {:type :int}
                     :max-depth {:type :int}
                     :min-samples-split {:type :int}
                     :min-samples-leaf {:type :int}
                     :max-features {}
                     :bootstrap {:type :boolean}
                     :random-seed {:type :int}
                     :parallel {:type :boolean}
                     :parallel-features {:type :boolean}}
   :explain-fn explain-random-forest})

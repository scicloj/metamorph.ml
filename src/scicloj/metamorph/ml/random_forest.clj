(ns scicloj.metamorph.ml.random-forest
   "Pure Clojure Random Forest implementation for classification and regression."
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

 (defn my-shuffle
   "Return a random permutation of coll"
   {:added "1.2"
    :static true}
   [^java.util.Collection coll ^java.util.Random rng]
   (let [al (java.util.ArrayList. coll)]
     (java.util.Collections/shuffle al rng)
     (clojure.lang.RT/vector (.toArray al))))

(defn- random-sample
  "Sample k random indices from 0 to n-1."
  [n k rng]
  (take k (my-shuffle (range n) rng)))

;; ============================================================================
;; Impurity and Split Metrics
;; ============================================================================

(defn- gini-impurity
  "Calculate Gini impurity for classification."
  [y]
  (let [n (long (count y))]
    (if (zero? n)
      0.0
      (let [class-counts (frequencies y)
            n-double (double n)
            sum-squares (reduce (fn [^double acc [_ cnt]]
                                 (let [p (/ (double cnt) n-double)]
                                   (+ acc (* p p))))
                               0.0
                               class-counts)]
        (- 1.0 sum-squares)))))

(defn- mse
  "Calculate mean squared error for regression."
  [y]
  (if (empty? y)
    0.0
    (let [n (long (count y))
          n-double (double n)
          sum (reduce (fn [^double acc val] (+ acc (double val))) 0.0 y)
          mean (/ sum n-double)
          sum-squared-diffs (reduce (fn [^double acc val]
                                     (let [diff (- (double val) mean)]
                                       (+ acc (* diff diff))))
                                   0.0
                                   y)]
      (/ sum-squared-diffs n-double))))

(defn- calculate-impurity
  "Calculate impurity based on task type."
  [y task]
  (case task
    :classification (gini-impurity y)
    :regression (mse y)))

;; ============================================================================
;; Splitting Logic
;; ============================================================================

(defn- split-data
  "Split data based on feature and threshold using indices (avoids copying data)."
  [X y feature-idx threshold]
  (let [n (count X)
        thresh (double threshold)
        left-indices (transient [])
        right-indices (transient [])]
    ;; Single pass through data, accumulate indices
    (loop [i 0]
      (when (< i n)
        (let [sample (nth X i)
              feature-val (double (nth sample feature-idx))]
          (if (<= feature-val thresh)
            (conj! left-indices i)
            (conj! right-indices i)))
        (recur (inc i))))
    (let [left-idx (persistent! left-indices)
          right-idx (persistent! right-indices)]
      {:left-X (mapv #(nth X %) left-idx)
       :left-y (mapv #(nth y %) left-idx)
       :right-X (mapv #(nth X %) right-idx)
       :right-y (mapv #(nth y %) right-idx)})))

(defn- calculate-split-gain
  "Calculate information gain from a split."
  [y left-y right-y task]
  (let [n (long (count y))
        n-left (long (count left-y))
        n-right (long (count right-y))]
    (if (or (zero? n-left) (zero? n-right))
      0.0
      (let [parent-impurity (calculate-impurity y task)
            left-impurity (calculate-impurity left-y task)
            right-impurity (calculate-impurity right-y task)
            n-double (double n)
            left-weight (/ (double n-left) n-double)
            right-weight (/ (double n-right) n-double)
            weighted-child-impurity (+ (* left-weight left-impurity)
                                       (* right-weight right-impurity))]
        (- parent-impurity weighted-child-impurity)))))

(defn- get-thresholds
  "Get candidate thresholds for a feature (sample up to max-thresholds for speed).

   For large datasets, trying all unique value midpoints is very slow.
   We sample at most 20 thresholds uniformly across the value range."
  [X feature-idx]
  (let [values (vec (sort (distinct (map #(nth % feature-idx) X))))
        n-values (count values)
        max-thresholds 20]
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

(defn- find-best-split
  "Find the best feature and threshold to split on."
  [X y feature-indices task]
  (let [n-samples (count X)]
    (loop [features feature-indices
           best-gain 0.0
           best-feature nil
           best-threshold nil]
      (if (empty? features)
        {:feature-idx best-feature
         :threshold best-threshold
         :gain best-gain}
        (let [feature-idx (first features)
              thresholds (get-thresholds X feature-idx)
              {:keys [gain threshold]}
              (reduce (fn [best thresh]
                        (let [{:keys [left-y right-y]} (split-data X y feature-idx thresh)
                              gain (calculate-split-gain y left-y right-y task)]
                          (if (> gain (:gain best))
                            {:gain gain :threshold thresh}
                            best)))
                      {:gain 0.0 :threshold nil}
                      thresholds)]
          (if (> gain best-gain)
            (recur (rest features) gain feature-idx threshold)
            (recur (rest features) best-gain best-feature best-threshold)))))))

;; ============================================================================
;; Decision Tree Construction
;; ============================================================================

(defn- create-leaf
  "Create a leaf node with prediction value."
  [y task]
  (let [n-samples (count y)]
    (case task
      :classification
      (let [class-counts (frequencies y)
            majority-class (key (apply max-key val class-counts))]
        {:type :leaf
         :value majority-class
         :class-counts class-counts
         :n-samples n-samples})

      :regression
      (let [mean (/ (reduce + y) (max 1 n-samples))]
        {:type :leaf
         :value mean
         :n-samples n-samples}))))

(defn- should-stop-split?
  "Check if we should stop splitting."
  [y depth max-depth min-samples-split task]
  (let [n-samples (count y)]
    (or
     ;; Reached max depth
     (and max-depth (>= depth max-depth))
     ;; Too few samples
     (< n-samples min-samples-split)
     ;; Pure node (classification only) - check cheaply first
     (and (= task :classification)
          (or (<= n-samples 1)
              (= (count (distinct y)) 1))))))

(defn- build-tree
  "Recursively build a decision tree."
  [X y depth max-depth min-samples-split max-features-per-split task rng]
  (let [n-samples (count y)
        n-features (count (first X))]

    ;; Check stopping criteria
    (if (should-stop-split? y depth max-depth min-samples-split task)
      (create-leaf y task)

      ;; Try to split
      (let [;; Randomly select features to consider
            feature-indices (if max-features-per-split
                              (random-sample n-features max-features-per-split rng)
                              (range n-features))
            {:keys [feature-idx threshold gain]} (find-best-split X y feature-indices task)]

        ;; If no valid split found, create leaf
        (if (or (nil? feature-idx) (<= gain 0.0))
          (create-leaf y task)

          ;; Create split node
          (let [{:keys [left-X left-y right-X right-y]} (split-data X y feature-idx threshold)
                left-child (build-tree left-X left-y (inc depth) max-depth min-samples-split max-features-per-split task rng)
                right-child (build-tree right-X right-y (inc depth) max-depth min-samples-split max-features-per-split task rng)]
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
  "Predict a single sample using a decision tree."
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

(defn- predict-tree-all
  "Predict all samples using a decision tree."
  [tree X]
  (mapv #(predict-tree tree %) X))

;; ============================================================================
;; Random Forest
;; ============================================================================

(defn- bootstrap-sample
  "Create a bootstrap sample (sample with replacement)."
  [X y ^java.util.Random rng]
  (let [n (count X)
        indices (repeatedly n #(.nextInt rng n))]
    {:X (mapv #(nth X %) indices)
     :y (mapv #(nth y %) indices)}))

(defn- train-single-tree
  "Train a single decision tree on bootstrap sample."
  [X y max-depth min-samples-split max-features-per-split bootstrap? task rng]
  (let [{sample-X :X sample-y :y} (if bootstrap?
                                    (bootstrap-sample X y rng)
                                    {:X X :y y})]
    (build-tree sample-X sample-y 0 max-depth min-samples-split max-features-per-split task rng)))

(defn- train-forest
  "Train multiple decision trees to form a random forest.

   Trees are trained in parallel using pmap for better performance."
  [X y {:keys [n-trees max-depth min-samples-split max-features bootstrap random-seed parallel]
        :or {n-trees 100
             max-depth nil
             min-samples-split 2
             max-features :sqrt
             bootstrap true
             random-seed nil
             parallel true}}
   task]
  (let [n-features (count (first X))
        max-features-per-split (calculate-max-features n-features max-features)
        base-rng (if random-seed
                   (java.util.Random. random-seed)
                   (java.util.Random.))
        ;; Generate seeds for each tree to ensure reproducibility
        tree-seeds (vec (repeatedly n-trees #(.nextInt base-rng)))

        ;; Function to train a single tree with its own RNG
        train-fn (fn [tree-seed]
                   (let [rng (java.util.Random. tree-seed)]
                     (train-single-tree X y max-depth min-samples-split
                                       max-features-per-split bootstrap task rng)))]

    ;; Use pmap for parallel execution (unless disabled)
    (if parallel
      (vec (pmap train-fn tree-seeds))
      (mapv train-fn tree-seeds))))

(defn- predict-forest
  "Predict using all trees and aggregate results."
  [trees X task]
  (let [n-trees (count trees)
        n-samples (count X)]

    (case task
      :classification
      ;; Majority voting - process sample by sample to avoid large intermediate structures
      (loop [sample-idx 0
             result (transient [])]
        (if (< sample-idx n-samples)
          (let [sample (nth X sample-idx)
                ;; Collect predictions from all trees for this sample
                predictions (loop [tree-idx 0
                                   preds (transient [])]
                              (if (< tree-idx n-trees)
                                (recur (inc tree-idx)
                                       (conj! preds (predict-tree (nth trees tree-idx) sample)))
                                (persistent! preds)))
                vote-counts (frequencies predictions)
                majority (key (apply max-key val vote-counts))]
            (recur (inc sample-idx)
                   (conj! result majority)))
          (persistent! result)))

      :regression
      ;; Average predictions
      (loop [sample-idx 0
             result (transient [])]
        (if (< sample-idx n-samples)
          (let [sample (nth X sample-idx)
                ;; Sum predictions from all trees
                sum (loop [tree-idx 0
                          acc 0.0]
                      (if (< tree-idx n-trees)
                        (recur (inc tree-idx)
                               (+ acc (double (predict-tree (nth trees tree-idx) sample))))
                        acc))
                avg (/ sum (double n-trees))]
            (recur (inc sample-idx)
                   (conj! result avg)))
          (persistent! result))))))

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

        ;; Make predictions
        predictions (predict-forest trees X task)

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
                     :max-features {}
                     :bootstrap {:type :boolean}
                     :random-seed {:type :int}
                     :parallel {:type :boolean}}
   :explain-fn explain-random-forest})

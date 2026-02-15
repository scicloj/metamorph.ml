(ns scicloj.metamorph.ml.random-forest
   "Pure Clojure Random Forest implementation for classification and regression."
   (:require [scicloj.metamorph.ml :as ml]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.modelling :as ds-mod]))

(set! *unchecked-math* :warn-on-boxed)

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
  (let [n (count y)
        class-counts (frequencies y)]
    (if (zero? n)
      0.0
      (- 1.0
         (reduce + (map (fn [[_ ^long cnt]]
                          (let [p (/ (double cnt) (double n))]
                            (* p p)))
                        class-counts))))))

(defn- mse
  "Calculate mean squared error for regression."
  [y]
  (if (empty? y)
    0.0
    (let [n (count y)
          mean (/ (reduce + y) (double n))
          squared-diffs (map #(let [diff (- (double %) mean)] (* diff diff)) y)]
      (/ (reduce + squared-diffs) (double n)))))

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
  "Split data based on feature and threshold."
  [X y feature-idx threshold]
  (let [indices (range (count X))
        left-indices (filter #(<= (get-in X [% feature-idx]) threshold) indices)
        right-indices (filter #(> (get-in X [% feature-idx]) threshold) indices)]
    {:left-X (mapv #(nth X %) left-indices)
     :left-y (mapv #(nth y %) left-indices)
     :right-X (mapv #(nth X %) right-indices)
     :right-y (mapv #(nth y %) right-indices)}))

(defn- calculate-split-gain
  "Calculate information gain from a split."
  [y left-y right-y task]
  (let [n (count y)
        n-left (count left-y)
        n-right (count right-y)]
    (if (or (zero? n-left) (zero? n-right))
      0.0
      (let [parent-impurity (calculate-impurity y task)
            left-impurity (calculate-impurity left-y task)
            right-impurity (calculate-impurity right-y task)
            weighted-child-impurity (+ (* (/ n-left n) left-impurity)
                                       (* (/ n-right n) right-impurity))]
        (- parent-impurity weighted-child-impurity)))))

(defn- get-thresholds
  "Get candidate thresholds for a feature (sample up to max-thresholds for speed).

   For large datasets, trying all unique value midpoints is very slow.
   We sample at most 20 thresholds uniformly across the value range."
  [X feature-idx]
  (let [values (vec (sort (distinct (map #(nth % feature-idx) X))))
        n-values (count values)
        max-thresholds 20]
    (if (< n-values 2)
      []
      (if (<= n-values max-thresholds)
        ;; Few values: use all midpoints
        (map (fn [[a b]] (/ (+ a b) 2.0))
             (partition 2 1 values))
        ;; Many values: sample uniformly
        (let [step (/ (dec n-values) (double max-thresholds))
              indices (map #(int (* % step)) (range max-thresholds))]
          (map (fn [idx]
                 (let [a (nth values idx)
                       b (nth values (min (inc idx) (dec n-values)))]
                   (/ (+ a b) 2.0)))
               indices))))))

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
     ;; Pure node (classification only)
     (and (= task :classification)
          (= (count (distinct y)) 1)))))

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
  (case (:type tree)
    :leaf (:value tree)
    :split (let [{:keys [feature-idx threshold left right]} tree
                 feature-value (nth sample feature-idx)]
             (if (<= feature-value threshold)
               (recur left sample)
               (recur right sample)))))

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
        all-predictions (mapv #(predict-tree-all % X) trees)]

    ;; Transpose predictions: per-sample predictions across all trees
    (case task
      :classification
      ;; Majority voting
      (mapv (fn [sample-idx]
              (let [tree-predictions (map #(nth % sample-idx) all-predictions)
                    vote-counts (frequencies tree-predictions)]
                (key (apply max-key val vote-counts))))
            (range (count X)))

      :regression
      ;; Average predictions
      (mapv (fn [sample-idx]
              (let [tree-predictions (map #(nth % sample-idx) all-predictions)]
                (/ (reduce + tree-predictions) n-trees)))
            (range (count X))))))

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

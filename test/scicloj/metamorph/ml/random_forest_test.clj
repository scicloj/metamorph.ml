(ns scicloj.metamorph.ml.random-forest-test 
  (:require
   [scicloj.metamorph.ml.random-forest :refer [random-forest predict-forest]]))

(def seed 1234)

(def random
  (java.util.Random. seed))

(defn seeded-rand [n]
  (* n (.nextDouble random)))



(defn deterministic-shuffle
  [^java.util.Collection coll]
  (let [al (java.util.ArrayList. coll)
        rng (java.util.Random. seed)]
    (java.util.Collections/shuffle al rng)
    (clojure.lang.RT/vector (.toArray al))))

(defn train-test-split [data test-ratio]
  (let [shuffled (deterministic-shuffle data)
        test-size (int (* test-ratio (count data)))]
    {:test (take test-size shuffled)
     :train (drop test-size shuffled)}))

;; Function to calculate accuracy

(defn accuracy [predictions labels]
  (let [correct (count (filter true? (map = predictions labels)))
        total (count labels)]
    (/ correct total)))

;; Function to generate synthetic dataset of size n

(defn generate-dataset [n]
  (let [rand-feature (fn [] (seeded-rand 10))
        rand-label (fn [f1 f2]
                     (if (> (+ f1 f2) 10)
                       1
                       0))]
    (for [_ (range n)]
      (let [f1 (rand-feature)
            f2 (rand-feature)
            label (rand-label f1 f2)]
        {:feature1 f1
         :feature2 f2
         :label label}))))

;; Main function

;; Generate a dataset with 100 samples
(def dataset (generate-dataset 100))
(def test-ratio 0.3) ;; 30% of data for testing
(def data-split (train-test-split dataset test-ratio))
(def train-data (:train data-split))
(def test-data (:test data-split))
 ;; Build the random forest using training data
(def n-trees 10) ;; Number of trees (increased for a larger dataset)
(def max-depth 10);; Maximum depth of each tree
(def min-size 1);; Minimum size of groups (leaf nodes)
(def sample-size (count train-data)) ;; Number of samples per tree
(def n-features 2) ;; Number of features to consider at each split
(def forest (random-forest train-data n-trees max-depth min-size sample-size n-features))
;; Make predictions on test data
(defn get-predictions [forest test-data]
  (map #(predict-forest forest %) test-data))
(def test-labels (map :label test-data))
(def predictions (get-predictions forest test-data))
;; Calculate accuracy
(def acc (accuracy predictions test-labels))
(println "Accuracy:" (* 100.0 acc) "%")

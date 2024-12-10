(ns scicloj.metamorph.ml.random-forest-test
  (:require
   [clojure.test :refer [is deftest]]
   [scicloj.metamorph.ml.random-forest :refer [random-forest predict-forest]]
   [tablecloth.api :as tc]))

(def seed 1234)


(defn seeded-rand [n random]
  (* n (.nextDouble random)))




(defn deterministic-shuffle
  [^java.util.Collection coll random]
  (let [al (java.util.ArrayList. coll)
        ]
    (java.util.Collections/shuffle al random)
    (clojure.lang.RT/vector (.toArray al))))

(defn train-test-split [data test-ratio random]
  (let [shuffled (deterministic-shuffle data random)
        test-size (int (* test-ratio (count data)))]
    {:test (take test-size shuffled)
     :train (drop test-size shuffled)}))

;; Function to calculate accuracy

(defn accuracy [predictions labels]
  (let [correct (count (filter true? (map = predictions labels)))
        total (count labels)]
    (/ correct total)))

;; Function to generate synthetic dataset of size n

(defn generate-dataset [n random]
  (let [rand-feature (fn [] (seeded-rand 10 random))
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


(defn get-predictions [forest test-data]
  (map #(predict-forest forest %) test-data))

(require '[clj-async-profiler.core :as prof])

(System/getProperty "jdk.attach.allowAttachSelf")
(deftest random-forest-test
;; Generate a dataset with 100 samples
  
  (let [ random (java.util.Random. seed)

        dataset (generate-dataset 1000 random)
        test-ratio 0.3 ;; 30% of data for testing
        data-split (train-test-split dataset test-ratio random)
        train-data (:train data-split)
        test-data (:test data-split)

        n-trees 10 ;; Number of trees (increased for a larger dataset)
        max-depth 10 ;; Maximum depth of each tree
        min-size 1 ;; Minimum size of groups (leaf nodes)
        sample-size (count train-data) ;; Number of samples per tree
        n-features 2 ;; Number of features to consider at each split
        forest 
        (time
         (doall
          (random-forest train-data n-trees max-depth min-size sample-size n-features)))

;; Make predictions on test data
        test-labels (map :label test-data)
        predictions (get-predictions forest test-data)
;; Calculate accuracy
        acc (accuracy predictions test-labels)
        ]
    (is (= 0.97 (double acc)))
    ))

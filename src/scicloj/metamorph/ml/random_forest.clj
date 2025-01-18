(ns scicloj.metamorph.ml.random-forest
  (:require [clojure.set :as set]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(def seed 1234)

(defn seeded-rand-int [random n]
  (.nextInt random n))
(defn seeded-rand-nth [random coll]
  (nth coll (seeded-rand-int random (count coll))))

(defn deterministic-shuffle
  [^java.util.Collection coll random]
  (let [al (java.util.ArrayList. coll)]
    (java.util.Collections/shuffle al random)
    (clojure.lang.RT/vector (.toArray al))))


;; Function to generate bootstrap samples

(defn bootstrap-sample [data sample-size random]
  (repeatedly sample-size #(seeded-rand-nth random data)))

;; Function to compute the Gini impurity

(defn gini-impurity [labels]
  (let [total (count labels)
        freqs (vals (frequencies labels))
        probs (map #(/ % total) freqs)]
    (- 1 (reduce + (map #(* % %) probs)))))

;; Function to split the dataset based on a feature and a value

(defn split-dataset [data feature value]
  (let [left (filter #(<= (feature %) value) data)
        right (filter #(> (feature %) value) data)]
    [left right]))

;; Function to find the best split

(defn best-split [data features]
  (let [base-gini (gini-impurity (map :label data))]
  (reduce
   (fn [best feature]
     (reduce
      (fn [best value]
        (let [[left right] (split-dataset data feature value)
              p-left (/ (count left) (count data))
              p-right (/ (count right) (count data))
              gini-left (gini-impurity (map :label left))
              gini-right (gini-impurity (map :label right))
              gini-split (+ (* p-left gini-left) (* p-right gini-right))]

          (if (< gini-split (:gini best))
            {:feature feature
             :value value
             :gini gini-split
             :groups [left right]}
            best)))
      best
      (distinct (map feature data))))

   {:gini base-gini}
   features)))

;; Function to create a terminal node (leaf)

(defn to-terminal [group]
  (let [labels (map :label group)
        freqs (frequencies labels)]
    (key (apply max-key val freqs))))

;; Recursive function to build a decision tree

(defn build-tree [data max-depth min-size n-features depth random]
  (let [labels (map :label data)]
    (if (or (empty? data)
            (<= (count (distinct labels)) 1)
            (>= depth max-depth)
            (<= (count data) min-size))
      (to-terminal data)
      (let [all-features (remove #{:label} (keys (first data)))
            features (take n-features (deterministic-shuffle all-features random))
            split (best-split data features)]
        (if (or (nil? (:groups split))
                (empty? (first (:groups split)))
                (empty? (second (:groups split))))
          (to-terminal data)
          (let [[left right] (:groups split)
                node {:feature (:feature split)
                      :value (:value split)}]
            (assoc node
                   :left (build-tree left max-depth min-size n-features (inc depth) random)
                   :right (build-tree right max-depth min-size n-features (inc depth) random))))))))

;; Function to build a random forest

(defn random-forest [data n-trees max-depth min-size sample-size n-features map-fn]
  (doall
   (let [random (java.util.Random. 1234)]
     (map-fn 
      (fn [_] (let [sample (bootstrap-sample data sample-size random)]
                (build-tree sample max-depth min-size n-features 1 random)))
      (range n-trees)
      ))))

;; Function to make a prediction with a single tree

(defn predict-tree [tree sample]
  (if (map? tree)
    (let [feature (:feature tree)
          value (:value tree)
          subtree (if (<= (feature sample) value)
                    (:left tree)
                    (:right tree))]
      (predict-tree subtree sample))
    tree))

;; Function to make a prediction with the forest

(defn predict-forest [forest sample]
   
  (let [predictions (map #(predict-tree % sample) forest)
        freqs (frequencies predictions)]
    (key (apply max-key val freqs))))


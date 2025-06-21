(ns scicloj.metamorph.ml.loss
  "Simple loss functions."
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.argops :as argops]))

(defn mse
  "mean squared error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (double (-> (dfn/- predictions labels)
                    (dfn/pow 2)
                    (dfn/reduce-+)))
        n))))


(defn rmse
  "root mean squared error"
  ^double [predictions labels]
  (-> (mse predictions labels)
      (Math/sqrt)))


(defn mae
  "mean absolute error"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (let [n (count predictions)]
    (double
     (/ (double (-> (dfn/- predictions labels)
                    dfn/abs
                    (dfn/reduce-+)))
        n))))


(defn- validate-accuracy-inputs [lhs rhs]
    (let [lhs-all-numeric (every? number? lhs)
          rhs-all-numeric (every? number? rhs)]

     (errors/when-not-error (not (-> lhs meta :categorical-map)) "lhs should not have categorical map, please revert it before calculating accuracy. See https://scicloj.github.io/noj/noj_book.prepare_for_ml.html#categorical-maps-attached-to-a-column-change-semantic-value-of-the-column")
     (errors/when-not-error (not (-> rhs meta :categorical-map)) "rhs should not have categorical map, please revert it before calculating accuracy. See https://scicloj.github.io/noj/noj_book.prepare_for_ml.html#categorical-maps-attached-to-a-column-change-semantic-value-of-the-column")
     (errors/when-not-errorf (= lhs-all-numeric rhs-all-numeric)
                             "lhs / rhs need to be either both numeric or both non-numeric: lhs: %s rhs: %s" lhs rhs))

  (errors/when-not-errorf
   (= (dtype/ecount lhs)
      (dtype/ecount rhs))
   "Ecounts do not match: %d %d"
   (dtype/ecount lhs) (dtype/ecount rhs)))
 

(defn classification-accuracy
  "correct/total.
Model output is a sequence of probability distributions.
label-seq is a sequence of values.  The answer is considered correct
if the key highest probability in the model output entry matches
that label."
  ^double [lhs rhs]
  (validate-accuracy-inputs lhs rhs)
  (/ (dtype/ecount (argops/binary-argfilter :tech.numerics/eq lhs rhs))
     (dtype/ecount lhs)))


(defn classification-loss
  "1.0 - classification-accuracy."
  ^double [lhs rhs]
  (- 1.0
     (classification-accuracy lhs rhs)))


(defn auc
  "Calculates area under the ROC curve. Uses AUC formula from R's 'mlr' package.
  (sum(r[i]) - n.pos * (n.pos + 1) / 2) / (n.pos * n.neg)
  See https://github.com/mlr-org/mlr/blob/main/R/measures.R"
  ^double [predictions labels]
  (assert (= (count predictions) (count labels)))
  (assert (= (set labels) #{0 1}))
  (let [sort-by-index (fn [s index]
                        (let [sorted-pairs (map (fn [x y] (list x y)) s index)]
                          (map first (sort-by second sorted-pairs))))
        sorted-labels (sort-by-index labels predictions)
        n (count labels)
        n-pos (reduce + labels)
        n-neg (- n n-pos)
        rank (range 1 (inc n))]
    (/ (- (reduce + (map * rank sorted-labels))
          (* n-pos (inc n-pos) 1/2))
       (* n-pos n-neg))))



(comment
  ;; not sure how tghis relates to fn auc
  ;; this needs probability score
  (defn compute-roc-auc
    
    "Calculates the ROC AUC given true labels and predicted scores.  
    
  Parameters:  
  - y_true: A sequence of true binary labels (0 or 1).  
  - y_scores: A sequence of predicted scores or probabilities.  
    
  Returns:  
  - AUC value between 0 and 1."
    [y_true y_scores]
    (let [n (count y_true)
          n_pos (count (filter #(= % 1) y_true))
          n_neg (count (filter #(= % 0) y_true))
        ;; Combine scores and labels into a sequence of tuples  
          data (map vector y_scores y_true)
        ;; Sort data by predicted scores in ascending order  
          sorted-data (sort-by first data)
        ;; Assign ranks, handling tied scores by assigning average ranks  
          ranks (let [grouped (vals (group-by first (map-indexed vector sorted-data)))
                      assign-ranks (fn [groups]
                                     (loop [gs groups
                                            current-rank 1
                                            ranks []]
                                       (if (empty? gs)
                                         ranks
                                         (let [group (first gs)
                                               size (count group)
                                               avg-rank (+ current-rank (/ (dec size) 2.0))]
                                           (recur (rest gs)
                                                  (+ current-rank size)
                                                  (concat ranks (repeat size avg-rank)))))))
                      ranks (assign-ranks grouped)]
                  ranks)
        ;; Sum ranks of positive samples  
          sum-positive-ranks (reduce + (map first
                                            (filter #(= (second (second %)) 1)
                                                    (map vector ranks sorted-data))))
        ;; Calculate AUC using the Mann-Whitney U statistic  
          auc (/ (- sum-positive-ranks (* n_pos (+ n_pos 1) 0.5))
                 (* n_pos n_neg))]
      auc))
  )

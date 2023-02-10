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


(defn classification-accuracy
  "correct/total.
Model output is a sequence of probability distributions.
label-seq is a sequence of values.  The answer is considered correct
if the key highest probability in the model output entry matches
that label."
  ^double [lhs rhs]
  (errors/when-not-errorf
   (= (dtype/ecount lhs)
      (dtype/ecount rhs))
   "Ecounts do not match: %d %d"
   (dtype/ecount lhs) (dtype/ecount rhs))
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

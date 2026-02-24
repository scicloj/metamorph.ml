(ns scicloj.metamorph.ml.metrics
  "Excellent metrics tools from the cortex project."
  (:require [tech.v3.datatype :as dtype]
            [fastmath.core :as fm]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.datatype.functional :as dfn]))

(defn wrongs
  "Identifies incorrect predictions in binary classification.

  `y` - Array of ground truth labels
  `y_hat` - Array of classifier predictions

  Returns an array with 1.0 where predictions don't match ground truth, 0.0 where
  they match. Arrays must have the same shape.

  Useful for computing error rates and analyzing misclassification patterns.

  See also: `error-rate`, `accuracy`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (dfn/not-eq y y_hat))

(defn error-rate
  "Calculates the proportion of incorrect predictions.

  `y` - Array of true class labels
  `y_hat` - Array of predicted class values

  Returns the error rate as a float in [0, 1], where 0 indicates perfect
  classification and 1 indicates all predictions are wrong. Computed as the
  number of misclassifications divided by total predictions.

  Arrays must have the same shape.

  See also: `accuracy`, `wrongs`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (let [wrong (dfn/sum (wrongs y y_hat))
        len (float (dtype/ecount y))]
    (/ wrong len)))

(defn accuracy
  "Calculates the proportion of correct predictions.

  `y` - Array of true class labels
  `y_hat` - Array of predicted class values

  Returns the accuracy as a float in [0, 1], where 1.0 indicates perfect
  classification. Computed as 1.0 minus the error rate, equivalent to the
  number of correct predictions divided by total predictions.

  Arrays must have the same shape.

  See also: `error-rate`, `precision`, `recall`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (- 1.0 (error-rate y y_hat)))

(defn false-positives
  "Identifies false positive predictions in binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns an array with 1.0 for false positives (predicted 1, actual 0) and 0.0
  elsewhere. Arrays must have the same shape.

  False positives are also known as Type I errors.

  See also: `false-negatives`, `true-positives`, `true-negatives`, `fpr`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (dfn/eq 1 (dfn/- y_hat y)))

(defn false-negatives
  "Identifies false negative predictions in binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns an array with 1.0 for false negatives (predicted 0, actual 1) and 0.0
  elsewhere. Arrays must have the same shape.

  False negatives are also known as Type II errors.

  See also: `false-positives`, `true-positives`, `true-negatives`, `fnr`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (dfn/eq 1 (dfn/- y y_hat)))

(defn true-positives
  "Identifies true positive predictions in binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns an array with 1.0 for true positives (predicted 1, actual 1) and 0.0
  elsewhere. Arrays must have the same shape.

  True positives represent correctly identified positive cases.

  See also: `true-negatives`, `false-positives`, `false-negatives`, `tpr`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (dfn/* y y_hat))

(defn true-negatives
  "Identifies true negative predictions in binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns an array with 1.0 for true negatives (predicted 0, actual 0) and 0.0
  elsewhere. Arrays must have the same shape.

  True negatives represent correctly identified negative cases.

  See also: `true-positives`, `false-positives`, `false-negatives`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (dfn/eq 0 (dfn/+ y y_hat)))

(defn recall
  "Calculates recall (sensitivity, true positive rate) for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns recall as a double in [0, 1], computed as true positives divided by
  total actual positives (TP / (TP + FN)). Recall measures the proportion of
  actual positive cases that were correctly identified.

  High recall means few false negatives. Arrays must have the same shape.

  Also known as sensitivity, hit rate, or true positive rate.

  See also: `precision`, `tpr`, `fnr`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (let [true-count (dfn/sum y)
        true-pos-count (dfn/sum (true-positives y y_hat))]
    (/ (double true-pos-count) true-count)))

(defn precision
  "Calculates precision (positive predictive value) for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns precision as a float in [0, 1], computed as true positives divided by
  total predicted positives (TP / (TP + FP)). Precision measures the proportion
  of positive predictions that were correct.

  High precision means few false positives. Arrays must have the same shape.

  See also: `recall`, `fpr`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (let [true-pos-count (dfn/sum (true-positives y y_hat))
        false-pos-count (dfn/sum (false-positives y y_hat))]
    (/ true-pos-count (float (+ true-pos-count false-pos-count)))))

(defn fpr
  "Calculates false positive rate for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns FPR as a double in [0, 1], computed as false positives divided by
  total predicted negatives (FP / (FP + TN)). Uses the strict ROC definition.

  FPR measures the proportion of actual negatives incorrectly classified as
  positive. Lower values are better. Arrays must have the same shape.

  See also: `tpr`, `false-positives`, `precision`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (/ (dfn/sum (false-positives y y_hat))
     (dfn/sum (dfn/not-eq 1 y_hat))))

(defn tpr
  "Calculates true positive rate (sensitivity, recall) for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns TPR as a double in [0, 1], computed as true positives divided by
  total predicted positives (TP / (TP + FP)). Uses the strict ROC definition.

  TPR measures the proportion of positive predictions that are true positives.
  Higher values are better. Arrays must have the same shape.

  Also known as recall or sensitivity.

  See also: `fpr`, `fnr`, `recall`, `true-positives`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (/ (dfn/sum (true-positives y y_hat))
     (dfn/sum y_hat)))

(defn fnr
  "Calculates false negative rate for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_hat` - Array of predicted binary values (0 or 1)

  Returns FNR as a double in [0, 1], computed as 1 minus the true positive rate.
  Uses the strict ROC definition.

  FNR measures the proportion of actual positives incorrectly classified as
  negative. Lower values are better. Arrays must have the same shape.

  See also: `tpr`, `false-negatives`, `recall`"
  [y y_hat]
  {:pre [(= (dtype/shape y) (dtype/shape y_hat))]}
  (- 1 (tpr y y_hat)))

(defn threshold
  "Creates a binary mask by thresholding estimated probabilities.

  `y_est` - Array of estimated probabilities or scores
  `thresh` - Threshold value for binarization

  Returns an array where values >= thresh are true/1 and values < thresh are false/0.

  Used to convert probability outputs into binary predictions for ROC curve
  analysis and threshold optimization.

  See also: `roc-curve`, `equal-error-point`"
  [y_est thresh]
  (dfn/>= y_est thresh))

(defn unit-space
  "Generates evenly-spaced values in the unit interval [0.0, 1.0].

  `divs` - Number of divisions (bins) to create

  Returns an array with `divs + 1` values evenly distributed from 0.0 to 1.0,
  inclusive. For example, `divs=4` produces [0.0 0.25 0.5 0.75 1.0].

  Used internally for generating threshold values in ROC curve computation.

  See also: `roc-curve`"
  [divs]
  (dfn/* (/ 1.0 (double divs)) (range (inc divs))))

(defn- roc-dedupe
  "Dedupes all values for the roc curve in which the same true and false positive
  rates are stored so that we only maintain the (discretized) boundaries for when
  we change rates."
  [triplet-seq]
  (reduce (fn [fp-tp-thresh [fp tp thresh]]
            (let [[fp-prev tp-prev] (last fp-tp-thresh)]
              (if-not (= [fp-prev tp-prev] [fp tp])
                (conj fp-tp-thresh [fp tp thresh])
                fp-tp-thresh)))
          []
          triplet-seq))

(defn roc-curve
  "Computes an ROC (Receiver Operating Characteristic) curve for binary classification.

  `y` - Array of true binary labels (0 or 1)
  `y_est` - Array of estimated probabilities or scores
  `bins` - Number of threshold discretization levels (default: 100)

  Returns a sequence of [fpr tpr threshold] triplets, de-duplicated to include
  only boundary points where FPR or TPR changes. Thresholds range from 0.0 to 1.0.

  The ROC curve visualizes the trade-off between true positive rate and false
  positive rate across different classification thresholds.

  Note: This is a basic implementation. Consider using dedicated libraries for
  production ROC analysis.

  See also: `tpr`, `fpr`, `threshold`, `equal-error-point`, `eer-accuracy`"
  ([y y_est] (roc-curve y y_est 100))
  ([y y_est bins]
   (let [threshold-space (unit-space bins)
         thresholds (remove
                     (fn [th] ; thresholds of all yes or all no create nonsense output
                       (let [pred-count (dfn/sum (threshold y_est th))]
                         (or (zero? pred-count)
                             (= pred-count (float (dtype/ecount y_est))))))
                     threshold-space)
         fprs (map #(fpr y (threshold y_est %)) thresholds)
         tprs (map #(tpr y (threshold y_est %)) thresholds)]
     (roc-dedupe (map vector fprs tprs thresholds)))))

(defn equal-error-point
  "Finds the classification threshold that minimizes the difference between FPR and (1 - TPR).

  `y` - Array of true binary labels (0 or 1)
  `y_est` - Array of continuous estimated probabilities or scores (normalized)
  `bins` - Number of threshold discretization levels (default: 100)

  Returns the threshold value where false positive rate and false negative rate
  are approximately equal. This is the equal error rate (EER) operating point,
  commonly used in biometric verification systems.

  See also: `eer-accuracy`, `roc-curve`"
  ([y y_est] (equal-error-point y y_est 100))
  ([y y_est bins]
   (->> (roc-curve y y_est bins)
        (map (fn [[fp tp thresh]]
               [thresh
                (dfn/abs (- fp (- 1 tp)))]))
        (apply min-key second)
        first)))


(defn eer-accuracy
  "Calculates accuracy at the equal error rate (EER) operating point.

  `y` - Array of true binary labels (0 or 1)
  `y_est` - Array of continuous estimated probabilities or scores
  `bins` - Number of threshold discretization levels (default: 100)

  Returns a map with:
  - `:accuracy` - Classification accuracy at the EER threshold
  - `:threshold` - The threshold value where TPR and FPR are balanced

  EER accuracy is the standard metric in biometric systems (e.g., facial
  recognition) where false accept and false reject errors are equally weighted.

  See also: `equal-error-point`, `accuracy`"
  ([y y_est] (eer-accuracy y y_est 100))
  ([y y_est bins]
   (let [thresh (equal-error-point y y_est bins)
         y_hat (threshold y_est thresh)]
     {:accuracy (accuracy y y_hat)
      :threshold thresh})))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; METRICS FOR LOCALIZATION WITH CLASSIFICATION ;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- map-values
  "Apply a function val-fn element-wise the the values of a hashmap hm."
  [val-fn hm]
  (zipmap (keys hm) (map val-fn (vals hm))))

(defn- bb-matches
  "Returns the list of label-prediction pairs that have a 'good' bounding box match with one another,
  where 'good' is detemined by the iou function and the iou-threshold.
  How the list is built:
  1. Calculate a list IOU containing the intersection over union (iou) for each label-prediction pair (l, p).
  2. Add the pair (l,p) with highest iou to the list C of matches.
  3. Remove any pairs from IOU that have either l or p in them, since those boxes can only be used once.
  4. If IOU is non-empty return to step 2, otherwise return C."
  [labels predictions iou-fn iou-threshold]
  (let [IOU (->> (for [l labels p predictions] {:iou (iou-fn l p) :label l :prediction p})
                 (remove #(> iou-threshold (:iou %)))
                 (sort-by :iou >))]
    (loop [M IOU C []]
      (if (empty? M)
        C
        (let [match (first M)]
          (recur (remove #(or (= (:label match) (:label %))
                              (= (:prediction match) (:prediction %))) M)
                 (conj C match)))))))


(defn- per-class-metrics
  "Returns the sensitivity, precision, and F1 scores given labels and predictions, which are assumed to be of the same class c.
   Since finding and classifying objects is not a binary valued test, these are only analogues of the well-known sensitivity, precision and F1 for binary tests.
   The definitions are as follows:
  Location Sensitivity = #(correctly located and classified objects with class c) / #(labels with class c)
  Location Precision = #(correctly located and classified objects with class c) / #(predictions with class c)
  Location F1 = harmonic mean of sensitivity and precision."
  [labels predictions iou-fn iou-threshold]
  (let [bb-matches (bb-matches labels predictions iou-fn iou-threshold)
        label-count (count labels)
        pred-count (count predictions)
        bb-count (count bb-matches)]
    (map-values double
                {:location-sensitivity (if (= 0 label-count)
                                         1
                                         (/ bb-count label-count))
                 :location-precision   (if (= 0 pred-count)
                                         1
                                         (/ bb-count pred-count))
                 :location-F1          (if (= 0 (+ label-count pred-count))
                                         1
                                         (/ (* 2 bb-count) (+ label-count pred-count)))})))

(defn- global-metrics
  "Returns 5 numbers:
  1. Location Sensitivity (also called RECALL) = #(bb-matches) / #(labels)
  2. Location Precision = #(bb-matches) / #(predictions)
  3. Location F1 = harmonic mean of (1) and (2) = 2 * #(bb-matches) / #(labels) + #(predictions)
  4. Classification accuracy = #(bb-matches with correct class) / #(bb-matches)
  5. Global F1 = (Location F1) * (Classification accuracy) = 2 * #(bb-matches with correct class) / #(labels) + #(predictions)"
  [labels predictions label->class-fn iou-fn iou-threshold]
  (let [bb-matches (bb-matches labels predictions iou-fn iou-threshold)
        bb-matches-with-correct-class (filter #(= (label->class-fn (:label %)) (label->class-fn (:prediction %))) bb-matches)
        label-count (count labels)
        pred-count (count predictions)
        bb-count (count bb-matches)
        bb-with-class-count (count bb-matches-with-correct-class)]
    (map-values double
                {:location-sensitivity    (if (= 0 label-count)
                                            1
                                            (/ bb-count label-count))
                 :location-precision      (if (= 0 pred-count)
                                            1
                                            (/ bb-count pred-count))
                 :location-F1             (if (= 0 (+ label-count pred-count))
                                            1
                                            (/ (* 2 bb-count) (+ label-count pred-count)))
                 :classification-accuracy (if (= 0 bb-count)
                                            1
                                            (/ bb-with-class-count bb-count))
                 :global-F1               (if (= 0 (+ label-count pred-count))
                                            1
                                            (/ (* 2 bb-with-class-count) (+ label-count pred-count)))})))

(defn all-metrics
  "Returns global and per-class metrics for a given set of labels and predictions.
  - label->class-fn should take a label or prediction and return the class as a string or keyword.
  - iou-fn should take a label and prediction and return the intersection over union score
  - iou-threshold determines what iou value constitutes a matching bounding box.
  ** NOTE: If labels and predictions are produced from a sequence of images,
     ensure that the bounding boxes are shifted in each image so that there is not an overlap."
  [labels predictions label->class-fn iou-fn iou-threshold]
  (let [labels-by-class      (group-by label->class-fn labels)
        predictions-by-class (group-by label->class-fn predictions)],
    (merge {:global-metrics (global-metrics labels predictions label->class-fn iou-fn iou-threshold)}
           {:per-class-metrics
            (for [class (sort (keys labels-by-class))]
              (merge {:class class}
                     (per-class-metrics
                       (get labels-by-class class)
                       (get predictions-by-class class)
                       iou-fn
                       iou-threshold)))})))




(defn AIC
  "Calculates the Akaike Information Criterion (AIC) for model selection.

  `model` - Trained model map
  `y` - Actual target values
  `yhat` - Predicted values
  `feature-count` - Number of features used in the model

  Returns AIC = 2k - 2L, where k = 2 + p (parameters) and L is the log-likelihood.
  Lower AIC values indicate better model fit with complexity penalty.

  See also: `scicloj.metamorph.ml.metrics/BIC`, `scicloj.metamorph.ml/loglik`"
  [model y yhat feature-count]
  (let [
        l (ml/loglik model
                     y
                     yhat)
        p feature-count
        k (+ 2 p)]
    (-
     (* 2 k)
     (* 2 l))))


(defn BIC
  "Calculates the Bayesian Information Criterion (BIC) for model selection.

  `model` - Trained model map
  `y` - Actual target values
  `yhat` - Predicted values
  `sample-size` - Number of samples in the dataset
  `feature-count` - Number of features used in the model

  Returns BIC = -2L + k*ln(n), where L is the log-likelihood, k = 2 + p (parameters),
  and n is the sample size. Lower BIC values indicate better model fit. BIC penalizes
  model complexity more heavily than AIC for larger sample sizes.

  See also: `scicloj.metamorph.ml.metrics/AIC`, `scicloj.metamorph.ml/loglik`"
  [model y yhat sample-size feature-count]
  (let [l (ml/loglik model y yhat)
        n sample-size
        p feature-count
        k (+ 2 p)]
    (+  (* -2 l)
        (* k  (fm/ln n)))))

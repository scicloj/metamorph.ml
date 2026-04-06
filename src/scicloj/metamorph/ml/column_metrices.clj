(ns scicloj.metamorph.ml.column-metrices
  (:require
   [clojure.set :as set]
   [scicloj.metamorph.ml.classification]
   [scicloj.metamorph.ml.loss :as loss]
   [tablecloth.column.api :as tc-col]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column :as col]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.casting :as casting])
  (:import
   [org.tribuo Prediction]
   [org.tribuo.classification Label LabelFactory]
   [org.tribuo.classification.evaluation LabelConfusionMatrix]
   [org.tribuo.classification.evaluation ConfusionMetrics]
   [org.tribuo.evaluation.metrics MetricTarget]
   [org.tribuo.evaluation.metrics EvaluationMetric$Average]
   [org.tribuo.impl ArrayExample]))

(def *insist* (atom true))

(defmacro insist
  "Evaluates expression x and throws an AssertionError with optional
  message if x does not evaluate to logical true.

  Assertion checks are omitted from compiled code if '*assert*' is
  false."
  {:added "1.0"}
  ([x]
   (when @*insist*
     `(when-not ~x
        (throw (new AssertionError (str "Assert failed: " (pr-str '~x)))))))
  ([x message]
   (when @*insist*
     `(when-not ~x
        (throw (new AssertionError (str "Assert failed: " ~message "\n" (pr-str '~x))))))))


(defn insist-uniform! [cols]
  (let [dtypes
        (map dt/datatype
             (apply concat cols))

        dtypes-freq (frequencies dtypes)]

    (insist (= 1 (count dtypes-freq))
            (format "Requires uniform elements, but found several: %s" dtypes-freq))))

(defn insist-discrete! [cols]
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist (or (tech.v3.datatype.casting/integer-type? datatype)
                  (= :keyword datatype)
                  (= :string datatype))
              (format "Requires `integer` datatype, but found: %s" datatype)))
   (apply concat cols)))

(defn insist-continous! [cols]
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist  (tech.v3.datatype.casting/float-type? datatype)
              (format "Requires `float` datatype, but found: %s" datatype)))
   (apply concat cols)))

(defn insist-dataset! [y-true y-pred]
  (insist (ds/dataset? y-true)
          (format "Type of 'y-true' is not 'dataset', but '%s' " (type y-true)))
  (insist (ds/dataset? y-pred)
          (format "Type of 'y-pred' is not 'dataset', but '%s' " (type y-pred))))



(defn ->prediction-cols [y-pred]
  (->
   (ds-cat/reverse-map-categorical-xforms y-pred)
   (cf/prediction)
   (ds/columns)))


(defn ->truth-cols [y-true]
  (-> y-true
      (ds-cat/reverse-map-categorical-xforms)
      (cf/target)
      (ds/columns)))

(defn insist-single-label! [cols name]
  (insist (= 1 (count cols))
          (format "Function require '1' '%s' column, but found: '%s' " name (count cols) )))


(defn insist-no-missing!
  ([cols]
   (run!
    (fn [col] (insist (empty? (col/missing col))
                      (format "Expect no missing values in column '%s', but found missing in indexes: %s "
                              (col/column-name col)
                              (col/missing col))))
    cols)))

(defn insist-same-row-number!
  ([cols]

   (let [distinct-element-counts
         (into #{} (map count cols))]
     (insist (= 1 (count distinct-element-counts))
             (format "Expect all columns to have same element count, but found %s" distinct-element-counts)))))

(defn convert-and-validate [y-true y-pred data-constraint-validators]
  (run!
   #(% y-pred y-true)
   (:dataset data-constraint-validators))

  (let [prediction-cols (->prediction-cols y-pred)
        trueth-cols (->truth-cols y-true)
        _
        (run!
         #(% prediction-cols "predict")
         (:predict-cols data-constraint-validators))

        _
        (run!
         #(% trueth-cols "truth / target")
         (:truth-cols data-constraint-validators))
        _
        (run!
         #(% (concat prediction-cols trueth-cols))
         (:every-col data-constraint-validators))

        prediction-col-0 (first prediction-cols)
        first-truth-col-0 (first trueth-cols)]
    [prediction-col-0 first-truth-col-0]))

(defn accuracy-score [y-true y-pred]

  (let [[prediction-col-0 truth-col-0] 
        (convert-and-validate y-true 
                              y-pred
                              {:dataset [insist-dataset!]
                               :predict-cols [insist-single-label!]
                               :truth-cols [insist-single-label!]
                               :every-col [insist-no-missing!
                                           insist-discrete!
                                           insist-uniform!
                                           insist-same-row-number!]}
                              )]


    (double
     (/
      (apply +
             (map #(if
                    (= %1 %2)
                     1
                     0)
                  prediction-col-0
                  truth-col-0))
      (count prediction-col-0)))))

(defn classification-metric
  ([y-true y-pred metric averaging-strategy options]


   (let [[prediction-col-0 truth-col-0]
         (convert-and-validate y-true
                               y-pred
                               {:dataset [insist-dataset!]
                                :predict-cols [insist-single-label!]
                                :truth-cols [insist-single-label!]
                                :every-col [insist-no-missing!
                                            insist-discrete!
                                            insist-uniform!
                                            insist-same-row-number!]})
         label-factory (LabelFactory.)
         labels
         (into #{}
               (concat prediction-col-0 truth-col-0))

         mutable-output-info (.generateInfo label-factory)
         label-map
         (zipmap
          labels
          (map
           #(Label. (str %))
           labels))
         _
         (run!
          #(.observe mutable-output-info %)
          (vals label-map))


         imuutable-output-info (.generateImmutableOutputInfo  mutable-output-info)

         predictions
         (map
          #(Prediction. (get label-map %1)
                        0
                        (ArrayExample. (get label-map %2)))
          prediction-col-0
          truth-col-0)

         cm (LabelConfusionMatrix. imuutable-output-info predictions)
         metric-target (MetricTarget. (case averaging-strategy
                                        :macro EvaluationMetric$Average/MACRO
                                        x           :micro EvaluationMetric$Average/MICRO))]

     (case metric
       :accuracy (ConfusionMetrics/accuracy metric-target cm)
       :f1 (ConfusionMetrics/f1 metric-target cm)
       :balanced-error-rate (ConfusionMetrics/balancedErrorRate metric-target)
       :fn (ConfusionMetrics/fn metric-target cm)
       :fp (ConfusionMetrics/fp metric-target cm)
       :tn (ConfusionMetrics/tn metric-target cm)
       :tp (ConfusionMetrics/tp metric-target cm)
       :fscore (ConfusionMetrics/fscore metric-target cm (:beta options))
       :precision (ConfusionMetrics/precision metric-target cm)
       :recall  (ConfusionMetrics/recall metric-target cm))))
  ([y-true y-pred metric averaging-strategy] (classification-metric y-true y-pred metric averaging-strategy {}))
  ([y-true y-pred metric] (classification-metric y-true y-pred metric :micro {})))




(defn roc_auc-score [y-true y-score multi-class average]
  ;; uses ovr
  (assert (= :ovr multi-class))
  (insist-dataset! y-true y-score)
  (let [target-cols (ds/columns (cf/target y-true))
        probability-columns (ds/columns (cf/probability-distribution y-score))
        ]
    (insist-no-missing! target-cols)
    (insist-uniform! target-cols)
    (insist-single-label! target-cols "y_true")
    (insist-discrete! target-cols)
    (insist-continous! probability-columns)
    (insist-same-row-number!
     (concat  target-cols probability-columns)))


  (let [label-column-name (first (ds/column-names y-true))
        col-names (into #{} (-> y-score ds/column-names))
        one-vs-rest
        (map
         #(hash-map  :one %
                     :rest (set/difference col-names #{%}))
         (-> y-score ds/column-names))
        aucs (map
              (fn [{:keys [one rest]}]

                (let [binarized-labels
                      (ds/categorical->one-hot y-true [label-column-name])
                      probs-one  (get y-score one)]
                  (loss/auc probs-one (get binarized-labels (keyword (format "%s-%s" (name label-column-name) one))))))
              one-vs-rest)]
    (case average
      nil aucs
      :macro (tc-col/mean aucs))))



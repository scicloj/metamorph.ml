(ns scicloj.metamorph.ml.column-metric
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
   [tech.v3.datatype.casting :as casting]
   [fastmath.stats :as stats]
   [fastmath.core :as m]
   )
  (:import
   [org.tribuo Prediction]
   [org.tribuo.classification Label LabelFactory]
   [org.tribuo.classification.evaluation LabelConfusionMatrix]
   [org.tribuo.classification.evaluation ConfusionMetrics]
   [org.tribuo.evaluation.metrics MetricTarget]
   [org.tribuo.evaluation.metrics EvaluationMetric$Average]
   [org.tribuo.impl ArrayExample]))

;; todo: take from fastmath
(defn- fastmath-multiclass-measure
  ([actual prediction] (fastmath-multiclass-measure actual prediction nil))
  ([actual prediction {:keys [average ^double beta metric weighted?]
                       :or {average stats/mean beta 0.5 metric :f1-score weighted? false}}]
   (let [all-labels (distinct (concat actual prediction))
         measures (->> all-labels
                       (map (fn [label] (let [all-bin-measures (stats/binary-measures-all actual prediction #{label})
                                              res (get all-bin-measures  metric)
                                              _ (assert res (format "binary measure not existing '%s' " metric ))
                                              ]
                                          (if (fn? res) (res beta) res))))
                       (map (fn [^double v] (if (m/nan? v) 0.0 v))))]
     (cond
       (not average) (zipmap all-labels measures)
       (= average :micro) (m// (stats/L0 actual prediction) (double (count actual)))
       weighted? (let [weights (map (frequencies actual) all-labels)]
                   (average measures weights))
       :else (average measures)))))


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

(defn- insist-no-NaN! [cols]
  (run!
   (fn [col]
     (let [no-nan (not-any?
                   #(and (double? %)
                         (.isNaN %))
                    col)]
       (insist
        no-nan
        "Column should not have any Double/NaN")))
   cols
   ))

(defn- insist-uniform! [cols]
  (let [dtypes
        (map dt/datatype
             (apply concat cols))

        dtypes-freq (frequencies dtypes)]

    (insist (= 1 (count dtypes-freq))
            (format "Requires uniform elements, but found several: %s" dtypes-freq))))

(defn- insist-discrete! [cols]
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist (or (tech.v3.datatype.casting/integer-type? datatype)
                  (= :keyword datatype)
                  (= :string datatype))
              (format "Requires `integer or :keyword or :string` datatype, but found: %s" datatype)))
   (apply concat cols)))

(defn- insist-discrete--integer! [cols]
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist (tech.v3.datatype.casting/integer-type? datatype)
              (format "Requires `integer` datatype, but found: %s" datatype)))
   (apply concat cols)))


(defn- insist-continous! [cols]
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist  (tech.v3.datatype.casting/float-type? datatype)
              (format "Requires `float` datatype, but found: %s" datatype)))
   (apply concat cols)))

(defn- insist-dataset! [y-true y-pred]
  (insist (ds/dataset? y-true)
          (format "Type of 'y-true' is not 'dataset', but '%s' " (type y-true)))
  (insist (ds/dataset? y-pred)
          (format "Type of 'y-pred' is not 'dataset', but '%s' " (type y-pred))))



(defn- ->prediction-cols [y-pred]
  (->
   (ds-cat/reverse-map-categorical-xforms y-pred)
   (cf/prediction)
   (ds/columns)))


(defn- ->target-cols [y-true]
  (-> y-true
      (ds-cat/reverse-map-categorical-xforms)
      (cf/target)
      (ds/columns)))

(defn- insist-single-label! [cols name]
  (insist (= 1 (count cols))
          (format "Function require '1' '%s' column, but found: '%s' " name (count cols) )))


(defn- insist-no-missing!
  ([cols]
   (run!
    (fn [col] (insist (empty? (col/missing col))
                      (format "Expect no missing values in column '%s', but found missing in indexes: %s "
                              (col/column-name col)
                              (col/missing col))))
    cols)))

(defn- insist-same-row-number!
  ([cols]

   (let [distinct-element-counts
         (into #{} (map count cols))]
     (insist (= 1 (count distinct-element-counts))
             (format "Expect all columns to have same element count, but found %s" distinct-element-counts)))))

(defn- cat-revert-and-validate [y-true y-pred data-constraint-validators pre-validate-fn]
  (run!
   #(% y-pred y-true)
   (:dataset data-constraint-validators))

  (let [prediction-cols (->prediction-cols y-pred)
        trueth-cols (->target-cols y-true)

        [prediction-cols trueth-cols] (pre-validate-fn prediction-cols trueth-cols)
        _ (def prediction-cols prediction-cols)
        _ (def trueth-cols trueth-cols)
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


(defn classification-metric--tribuo
  ([y-true y-pred metric averaging-strategy options]


   (let [[prediction-col-0 truth-col-0]
         (cat-revert-and-validate y-true
                               y-pred
                               {:dataset [insist-dataset!]
                                :predict-cols [insist-single-label!]
                                :truth-cols [insist-single-label!]
                                :every-col [insist-no-missing!
                                            insist-discrete!
                                            insist-uniform!
                                            insist-same-row-number!]}
                                  (fn [x y] [x y])
                                  )
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
  ([y-true y-pred metric averaging-strategy] (classification-metric--tribuo y-true y-pred metric averaging-strategy {}))
  ([y-true y-pred metric] (classification-metric--tribuo y-true y-pred metric :micro {})))




(defn roc_auc-score [y-true y-pred multi-class-handling averaging-method]
  ;; uses ovr
  (assert (= :ovr multi-class-handling))
  (insist-dataset! y-true y-pred)
  
  (let [target-cols (->target-cols y-true)
        probability-columns (ds/columns (cf/probability-distribution y-pred))
        ]
    
    (insist-no-missing! target-cols)
    (insist-uniform! target-cols)
    (insist-single-label! target-cols "y_true")
    (insist-discrete! target-cols)
    (insist-continous! probability-columns)
    (insist-same-row-number!
     (concat  target-cols probability-columns)))


  (let [label-column-name (first (ds/column-names y-true))
        col-names (into #{} (-> y-pred ds/column-names))
        one-vs-rest
        (map
         #(hash-map  :one %
                     :rest (set/difference col-names #{%}))
         (-> y-pred ds/column-names))
        aucs (map
              (fn [{:keys [one rest]}]

                (let [binarized-labels
                      (ds/categorical->one-hot y-true [label-column-name])
                      probs-one  (get y-pred one)]
                  (loss/auc probs-one (get binarized-labels (keyword (format "%s-%s" (name label-column-name) one))))))
              one-vs-rest)]
    (case averaging-method
      nil aucs
      :macro (tc-col/mean aucs))))

(defn cols->int [prediction-cols trueth-cols]
  ;; fastmath need numbers
  (let [values
        (distinct
         (concat
          (apply concat trueth-cols)
          (apply concat prediction-cols)))
        remap-map
        (zipmap

         values
         (range))]
    [(map #(map remap-map %) prediction-cols)
     (map #(map remap-map %) trueth-cols)]))

(defn classification-metric-fastmath
  ([y-true y-pred metric averaging options]

   (let [[prediction-col-0 truth-col-0]
         (cat-revert-and-validate y-true
                                  y-pred
                                  {:dataset [insist-dataset!]
                                   :predict-cols [insist-single-label!]
                                   :truth-cols [insist-single-label!]
                                   :every-col [insist-no-NaN!
                                               insist-no-missing!
                                               insist-discrete--integer!
                                               insist-uniform!
                                               insist-same-row-number!]}
                                  cols->int)]

     (fastmath-multiclass-measure truth-col-0
                                  prediction-col-0
                                  {:metric metric
                                   :average (case averaging
                                              :macro stats/mean
                                              :micro :micro)
                                   :beta (:beta options)})))
  ([y-true y-pred metric averaging] (classification-metric-fastmath y-true y-pred metric averaging {})))


(defn regression-metric--fastmath [y-true y-pred metric-fn]

  (let [[prediction-col-0 truth-col-0]
        (cat-revert-and-validate y-true
                              y-pred
                              {:dataset [insist-dataset!]
                               :predict-cols [insist-single-label!]
                               :truth-cols [insist-single-label!]
                               :every-col [insist-no-NaN!
                                           insist-no-missing!
                                           insist-continous!
                                           insist-uniform!
                                           insist-same-row-number!]}
                                 (fn [x y] [x y])
                                 )
        fastmath-stats-fn (requiring-resolve
                           (symbol (format "fastmath.stats/%s" (name metric-fn))))]
    (insist
     (some? fastmath-stats-fn)
     (format "Function '%s' does not existin in fastmat." (format "fastmath.stats/%s" (name metric-fn))))
    (fastmath-stats-fn
     truth-col-0 prediction-col-0)))



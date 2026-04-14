(ns scicloj.metamorph.ml.column-metric
  (:require
   [clojure.set :as set]
   [scicloj.metamorph.ml.loss :as loss]
   [tablecloth.column.api :as tc-col]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column :as col]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.casting :as casting]
   [fastmath.stats :as stats]
   [fastmath.core :as m]))

;; todo: take from fastmath
(defn- fastmath-multiclass-measure
  ([actual prediction] (fastmath-multiclass-measure actual prediction nil))
  ([actual prediction {:keys [average ^double beta metric weighted?]
                       :or {average stats/mean beta 0.5 metric :f1-score weighted? false}}]
   (let [all-labels (distinct (concat actual prediction))
         measures (->> all-labels
                       (mapv (fn [label] (let [all-bin-measures (stats/binary-measures-all actual prediction #{label})
                                              res (get all-bin-measures  metric)
                                              _ (assert (some? res) (format "binary measure not existing in fastmath : '%s' " metric ))
                                              ]
                                          (if (fn? res) (res beta) res))))
                       (mapv (fn [^double v] (if (m/nan? v) 0.0 v))))]
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
  (assert (not (ds/dataset? cols)))
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
  (assert (not (ds/dataset? cols)))
  (let [dtypes
        (map dt/datatype
             (apply concat cols))

        dtypes-freq (frequencies dtypes)]

    (insist (= 1 (count dtypes-freq))
            (format "Requires uniform elements, but found several: %s" dtypes-freq))))

(defn- insist-discrete! [cols]
  (assert (not (ds/dataset? cols)))
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist (or (tech.v3.datatype.casting/integer-type? datatype)
                  (= :keyword datatype)
                  (= :string datatype))
              (format "Requires `integer or :keyword or :string` datatype, but found: %s" datatype)))
   (apply concat cols)))

(defn- insist-discrete--integer! [cols]
  (assert (not (ds/dataset? cols)))
  (run!
   #(let [datatype
          (dt/datatype %)]
      (insist (tech.v3.datatype.casting/integer-type? datatype)
              (format "Requires `integer` datatype, but found: %s" datatype)))
   (apply concat cols)))


(defn- insist-continous! [cols]
  (assert (not (ds/dataset? cols)))
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



(defn- ->prediction-ds [y-pred-ds]
  (assert (ds/dataset? y-pred-ds))

  (->
   (ds-cat/reverse-map-categorical-xforms y-pred-ds)
   (cf/prediction)))


(defn- ->target-ds [y-true-ds]
  (assert (ds/dataset? y-true-ds))
  (-> y-true-ds
      (ds-cat/reverse-map-categorical-xforms)
      (cf/target)))

(defn- ->prob-ds [y-true-ds]
  (assert (ds/dataset? y-true-ds))
  (-> y-true-ds
      (ds-cat/reverse-map-categorical-xforms)
      (cf/probability-distribution)))


(defn- insist-single-label! [cols name]
  (assert (not (ds/dataset? cols)))
  (insist (= 1 (count cols))
          (format "Function require '1' '%s' column, but found: '%s' " name (count cols) )))


(defn- insist-no-missing! [cols]
  (assert (not (ds/dataset? cols)))
  (run!
   (fn [col] (insist (empty? (col/missing col))
                     (format "Expect no missing values in column '%s', but found missing in indexes: %s "
                             (col/column-name col)
                             (col/missing col))))
   cols))

(defn- insist-same-row-number! [cols]
  (assert (not (ds/dataset? cols)))
  (let [distinct-element-counts
        (into #{} (map count cols))]
    (insist (= 1 (count distinct-element-counts))
            (format "Expect all columns to have same element count, but found %s" 
                    distinct-element-counts))))

(defn- datasets->single-cols [y-true-ds y-pred-ds to-cols-fn data-constraint-validators]
  (run!
   #(% y-pred-ds y-true-ds)
   (:dataset data-constraint-validators))

  (let [prediction-ds (->prediction-ds y-pred-ds)
        trueth-ds (->target-ds y-true-ds)
        preprocessed (to-cols-fn {:prediction-ds prediction-ds
                                  :truth-ds trueth-ds})

        prediction-cols (ds/columns (:prediction-ds preprocessed))
        trueth-cols (ds/columns (:truth-ds preprocessed))
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
        truth-col-0 (first trueth-cols)]
    {:prediction-col prediction-col-0
     :truth-col truth-col-0}))



(defn roc_auc-score [y-true y-pred multi-class-handling averaging-method]
  ;; uses ovr
  (assert (= :ovr multi-class-handling))
  (insist-dataset! y-true y-pred)
  
  (let [target-ds (->target-ds y-true)
        probability-ds (->prob-ds y-pred)
        ]
    
    (insist-no-missing! (ds/columns target-ds))
    (insist-uniform! (ds/columns target-ds))
    (insist-single-label! (ds/columns target-ds) "y_true")
    (insist-discrete! (ds/columns target-ds))
    (insist-continous! (ds/columns probability-ds))
    (insist-same-row-number!
     (concat  (ds/columns target-ds) 
              (ds/columns probability-ds)))


    (let [label-column-name (first (ds/column-names target-ds))
          col-names (into #{} (ds/column-names probability-ds))
          one-vs-rest
          (map
           #(hash-map  :one %
                       :rest (set/difference col-names #{%}))
           (-> probability-ds ds/column-names))
          aucs (map
                (fn [{:keys [one rest]}]

                  (let [binarized-labels
                        (ds/categorical->one-hot target-ds [label-column-name])
                        probs-one  (get y-pred one)
                        labels (get binarized-labels (keyword (format "%s-%s" (name label-column-name) (name one))))
                        ]
                    
                    (assert (some? labels)
                            {:message  "No labels found in `binarized-label`" 
                             :binaritized-labels binarized-labels
                             :one one}
                            )
                    (loss/auc probs-one labels)))
                one-vs-rest)]
      (case averaging-method
        nil aucs
        :macro (tc-col/mean aucs)))))

(defn- ds->int-ds [{:keys [prediction-ds truth-ds]}]
  (assert (ds/dataset? prediction-ds))
  (assert (ds/dataset? truth-ds))
  (assert (= 1 (ds/column-count prediction-ds)))
  (assert (= 1 (ds/column-count truth-ds)))

  (assert (= (-> prediction-ds ds/columns first dt/elemwise-datatype)
             (-> truth-ds ds/columns first dt/elemwise-datatype)
             ))

  (let [datatype (-> prediction-ds ds/columns first dt/elemwise-datatype)]

    (if (casting/numeric-type? datatype)
      {:prediction-ds prediction-ds
       :truth-ds truth-ds}

      (let [all-values
            (concat
             (flatten (ds/rowvecs truth-ds))
             (flatten (ds/rowvecs prediction-ds)))

            distinc-values
            (distinct all-values)

            distinct-datatypes (distinct (map dt/datatype distinc-values))
            _ (assert (= 1 (count distinct-datatypes)))

            elem->int-map (zipmap distinc-values (range))]
        {:prediction-ds (ds/update-elemwise prediction-ds elem->int-map)
         :truth-ds (ds/update-elemwise truth-ds elem->int-map)}))))

  


(defn classification-metric--fastmath
  ([y-true y-pred metric averaging options]

   (let [{:keys [prediction-col truth-col]}
          (datasets->single-cols y-true
                                 y-pred
                                 ds->int-ds
                                 {:dataset [insist-dataset!]
                                  :predict-cols [insist-single-label!]
                                  :truth-cols [insist-single-label!]
                                  :every-col [insist-no-NaN!
                                              insist-no-missing!
                                              insist-discrete--integer!
                                              insist-uniform!
                                              insist-same-row-number!]})]
     (def metric metric)
     (def averaging averaging)
     (def truth-col truth-col)
     (def prediction-col prediction-col)
     (def options options)
     (fastmath-multiclass-measure truth-col
                                  prediction-col
                                  {:metric metric
                                   :average (case averaging
                                              :macro stats/mean
                                              :micro :micro)
                                   :beta (:beta options)})))
  ([y-true y-pred metric averaging] (classification-metric--fastmath y-true y-pred metric averaging {})))


(defn regression-metric--fastmath [y-true y-pred metric-fn]

  (let [{:keys [prediction-col truth-col]}
        (datasets->single-cols y-true
                               y-pred
                               identity
                               {:dataset [insist-dataset!]
                                :predict-cols [insist-single-label!]
                                :truth-cols [insist-single-label!]
                                :every-col [insist-no-NaN!
                                            insist-no-missing!
                                            insist-continous!
                                            insist-uniform!
                                            insist-same-row-number!]})
        fastmath-stats-fn (requiring-resolve
                           (symbol (format "fastmath.stats/%s" (name metric-fn))))]
    (insist
     (some? fastmath-stats-fn)
     (format "Function '%s' does not exist in fastmat." (format "fastmath.stats/%s" (name metric-fn))))
    (fastmath-stats-fn
     truth-col prediction-col)))



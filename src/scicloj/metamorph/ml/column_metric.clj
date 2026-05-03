(ns scicloj.metamorph.ml.column-metric
  "Model evaluation metrics for classification and regression tasks.

   This namespace provides functions to compute standard machine learning metrics
   on model predictions vs. ground truth labels, with support for both binary and
   multiclass classification as well as regression tasks.

   Key Functions:
   - `classification-metric`: Evaluate classification model predictions
   - `regression-metric`: Evaluate regression model predictions

   Classification Metrics (from fastmath.stats):
   Supports binary and multiclass metrics including accuracy, precision, recall,
   F1-score, and more. Multiclass metrics can be averaged using:
   - `:macro` - Unweighted mean of per-class metrics
   - `:micro` - Aggregated true/false positives globally
   Also supports `:roc-auc` for multiclass AUC scoring.

   Regression Metrics (from fastmath.stats):
   Distance and similarity metrics such as MAE, MSE, RMSE, R², etc.

   Data Format:
   - Input datasets must be tech.ml.dataset (TMD) format
   - Must have appropriate column metadata (:prediction, :target, etc.)
   - Support categorical mappings via :categorical-map metadata
   - Missing values and NaNs are detected and rejected appropriately

   Validation:
   The functions perform extensive validation including:
   - Column metadata correctness
   - Missing values and NaN detection
   - Type and datatype uniformity
   - Row count alignment between datasets
   - Single-label assumption (multi-label not yet supported)

   Example:
   ```
   (classification-metric y-true y-pred :f1 :macro {})
   (regression-metric y-true y-pred :mse)
   ```

   See also: `fastmath.stats` documentation for available metric names"
  (:require
   [fastmath.stats :as stats]
   [fastmath.vector :as v]
   [scicloj.metamorph.ml.classification]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column :as col]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.casting :as casting]))



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

  
(defn- roc_auc-score [y-true y-pred averaging-method]
  (insist-dataset! y-true y-pred)

  (let [target-ds (cf/target y-true)
        probability-ds (->prob-ds y-pred)
        average (case averaging-method
                  :macro v/average
                  :micro :micro
                  nil nil)]

    (insist-no-missing! (ds/columns target-ds))
    (insist-uniform! (ds/columns target-ds))
    (insist-single-label! (ds/columns target-ds) "y_true")
    (insist-discrete! (ds/columns target-ds))
    (insist-continous! (ds/columns probability-ds))
    (insist-same-row-number!
     (concat  (ds/columns target-ds)
              (ds/columns probability-ds)))

    (stats/multiclass-auc
     (-> target-ds ds-cat/reverse-map-categorical-xforms ds/columns first)
     probability-ds
     {:average average})))
  

(defn classification-metric
  "Calculates various classification metrics, supporting binary and multiclass data.
   Return a single float number  
   
   * `y-true` A TMD dataset, having the truth
   * `y-pred` A TMD dataset, having the prediction
   * `metric` A keyword, supports any metric from: https://generateme.github.io/fastmath/clay/stats.html#binary-classification-metrics
              and :roc-auc
   * `averaging` How the mostly binary metrices get averaged, supports :macro and :micro
   * `options` Options for the :metric-fn

  
  Multi-label data is so far not supported.
 
  Both datasets need to have columns containing the appropriate column metadata
  as foreseen by TMD, see here:https://techascent.github.io/tech.ml.dataset/tech.v3.dataset.column-filters.html 
   , eg:
   * :column-type being :prediction, :probability-distribution
   * :inference-target true
   * :categorical-map column metadata is explicitely supported and get handled properly when present, so gets taken into consideration
   when comparing columns

   The `ml/predict` fn is producing these type of datasets.

  The function validates various aspects and ev. rejects data which has:
   * wrong column metadata
   * missing values or NaNs
   * non-discrete values in :prediction column
   * non-uniform datatypes
   * multi-label data ( having > 1 :inference-target column)
   * mistmatch in shape between `y-true` and `y-pred`
   * others
   
   This might depend on the concrete metric-fn used.
   "
  ([y-true y-pred metric averaging options]

   (if (= :roc-auc metric)
     (roc_auc-score y-true y-pred averaging)

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
       (stats/multiclass-measure
        truth-col
        prediction-col
        {:metric metric
         :average (case averaging
                    :macro stats/mean
                    :micro :micro)
         :beta (get options :beta 0.5)}))))





  ([y-true y-pred metric averaging] (classification-metric y-true y-pred metric averaging {})))


(defn regression-metric 
    "Calculates various regression metrics and return a single float number  
     
     * `y-true` A TMD dataset, having the truth
     * `y-pred` A TMD dataset, having the prediction
     * `metric` A keyword, supports any metric from: https://generateme.github.io/fastmath/clay/stats.html#distance-and-similarity-metrics
  
    
   
    Both datasets need to have columns containing the appropriate column metadata
    as foreseen by TMD, see here:https://techascent.github.io/tech.ml.dataset/tech.v3.dataset.column-filters.html 
     , eg:
     * :column-type being :prediction
     * :inference-target true
  
     The `ml/predict` fn is producing these type of datasets.
  
    The function validates various aspects and ev. rejects data which has:
     * wrong column metadata
     * missing values or NaNs
     * non-continous values in :prediction column
     * non-uniform datatypes
     * is multi-label data ( having > 1 :inference-target column)
     * mistmatch in shape between `y-true` and `y-pred`
     * others
     
     This might depend on the concrete metric-fn used.
     "

  [y-true y-pred metric-fn]

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
     (format "Function '%s' does not exist in fastmath.stats." (format "fastmath.stats/%s" (name metric-fn))))
    (fastmath-stats-fn
     truth-col prediction-col)))



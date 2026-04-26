(ns scicloj.metamorph.ml
  "Core machine learning framework integrating metamorph pipelines with standardized model APIs.

   This is the central namespace of metamorph.ml, providing infrastructure for:
   - Registering and using machine learning models
   - Training models and making predictions
   - Evaluating pipelines via cross-validation
   - Standardized model diagnostics (glance, tidy, augment)
   - Optional caching of computationally expensive operations

   Key Concepts:

   **Model Registration**: Models are registered using `define-model!` and can be
   referenced by keyword (e.g., `:fastmath/ols`, `:metamorph.ml/dummy-classifier`).
   Models define a train-fn, predict-fn, and optional diagnostic functions.

   **Training and Prediction**:
   - `train`: Train a model on a dataset given options including :model-type
   - `predict`: Make predictions using a trained model
   - `train-predict-cache`: Optional cache to avoid redundant computations

   **Pipeline Evaluation**:
   - `evaluate-pipelines`: Evaluate multiple pipelines across train/test splits
   - `evaluate-one-pipeline`: Evaluate a single pipeline with cross-validation
   - Returns results sorted by metric performance with optional filtering
   - Supports parallel evaluation (:map/:pmap/:ppmap)

   **Model Diagnostics** (following tidymodels conventions):
   - `glance`: One-row model summary (goodness-of-fit)
   - `tidy`: One-row-per-component output (coefficients with statistics)
   - `augment`: One-row-per-observation output (predictions, residuals)

   Main API Functions:

   - `define-model!`: Register a new model type with train/predict/diagnostic functions
   - `train`: Train a model with a specified model-type
   - `predict`: Generate predictions from a trained model
   - `evaluate-pipelines`: Evaluate pipelines with cross-validation
   - `glance`: Get model summary statistics
   - `tidy`: Extract coefficient-level results
   - `augment`: Add predictions and residuals to data

   Pipeline Integration:

   Models integrate with metamorph pipelines via the `model` step, which:
   - Trains in :fit mode using training data
   - Predicts in :transform mode on new data
   - Stores model output column metadata for later evaluation

   Example Usage:

   ;; Register a custom model (rarely needed - use existing models)
   (define-model! :my/custom-model train-fn predict-fn {...})

   ;; Train a model
   (let [model (train iris-data {:model-type :fastmath/ols
                                 :target-columns [:Sepal.Width]
                                 :feature-columns [:Sepal.Length]})]
     ;; Get diagnostics
     (glance model)
     (tidy model)
     ;; Make predictions
     (predict iris-data model))

   ;; Evaluate multiple pipelines in cross-validation
   (evaluate-pipelines
     [pipeline1 pipeline2]
     train-test-splits
     metric-fn
     :accuracy
     {:map-fn :pmap})

   Built-in Models:

   **Regression**:
   - `:metamorph.ml/ols`: Apache Commons Math OLS
   - `:fastmath/ols`: FastMath OLS
   - `:fastmath/glm`: FastMath GLM
   - `:metamorph.ml/dummy-regressor`: Mean baseline

   **Classification**:
   - `:metamorph.ml/dummy-classifier`: Majority class or random baseline

   **Preprocessing**:
   See specific namespaces for transformers:
   - `scicloj.metamorph.ml.preprocessing`: Scaling and normalization
   - `scicloj.metamorph.ml.categorical`: One-hot encoding
   - `scicloj.metamorph.ml.r-model-matrix`: R formula features

   See also: `scicloj.metamorph.core` for metamorph pipeline mechanics,
   `scicloj.metamorph.ml.tidy-models` for diagnostic validation"
  (:require
   [clojure.set :as set]
   [scicloj.metamorph.ml.loss :as loss]
   [malli.core :as m]
   [malli.error :as me]
   [scicloj.metamorph.ml.malli :as malli]
   [scicloj.metamorph.ml.tidy-models :as tidy]
   [scicloj.metamorph.ml.impl.hyper-opt :as hyper-opt]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.datatype.export-symbols :as exporter]
   [tech.v3.datatype :as dt])
  (:import
   java.util.UUID))


(exporter/export-symbols scicloj.metamorph.ml.ensemble ensemble-pipe)

(def train-predict-cache
  "Controls , if train/predict invocations are cached.
   if 'use-cache' is true, the get-fn and set-fn functions ar called accorddngly"
  (atom {:use-cache false
         :get-fn (fn [key] nil)
         :set-fn (fn [key value] nil)}))


(defn- get-categorical-maps [ds]
  (->> (ds/column-names ds)
       (map #(list % (-> ds (get %) meta :categorical-map)))
       (remove #(nil? (second %)))
       (map #(hash-map (first %) (second %)))
       (apply merge)))





(defn evaluate-pipelines
  "Evaluates the performance of a seq of metamorph pipelines, which are suposed to have a model as last step under key :model,
  which behaves correctly  in mode :fit and  :transform. The function `scicloj.metamorph.ml/model` is such function behaving correctly.
  
   This function calculates the accuracy or loss, given as `metric-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits
  given in  `train-test-split-seq`.

   It runs the pipelines  in mode  :fit and in mode :transform for each pipeline-fn in `pipe-fn-seq` for each split in `train-test-split-seq`.

   The function returns a seq of seqs of evaluation results per pipe-fn per train-test split.
   Each of the evaluation results is a context map, which is specified in the malli schema attached to this function.

   * `pipeline-fn-or-decl-seq` need to be  sequence of pipeline functions or pipline declarations which follow the metamorph approach.
      These type of functions get produced typically by calling `scicloj.metamorph/pipeline`. Documentation is here:

   * `train-test-split-seq` need to be a sequence of maps containing the  train and test dataset (being tech.ml.dataset) at keys :train and :test.
    `tablecloth.api/split->seq` produces such splits. Supervised models require both keys (:train and :test), while unsupervised models only use :train

   * `metric-fn` Metric function to use. Typically comming from `tech.v3.ml.loss`. For supervised models the metric-fn receives the trueth
      and predicted values and should return a single double number.  The metric fns receives a a seq *without* categorical maps. These
      get reverse-applied to the prediction , if present, before passing the values to the metriic fn.
      For unsupervised models the function receives the fitted ctx
      and should return a singel double number as well. This metric will be used to sort and eventualy filter the result, depending on the options
      (:return-best-pipeline-only   and :return-best-crossvalidation-only). The notion of `best` comes from metric-fn combined with loss-and-accuracy
  

   * `loss-or-accuracy` If the metric-fn is a loss or accuracy calculation. Can be :loss or :accuracy. Decided the notion of `best` model.
      In case of :loss pipelines with lower metric are better, in case of :accuracy pipelines with higher value are better.

  * `options` map controls some mainly performance related parameters. These function can potentialy result in a large ammount of data,
    able to bring the JVM into out-of-memory. We can control how many details the function returns by the following parameter: 
     The default are quite aggresive in removing details, and this can be tweaked further into more or less details via:
     


       * `:return-best-pipeline-only` - Only return information of the best performing pipeline. Default is true.
       * `:return-best-crossvalidation-only` - Only return information of the best crossvalidation (per pipeline returned). Default is `true`.
       * `:map-fn` - Controls parallelism, so if we use map (:map) , pmap (:pmap) or :mapv to map over different pipelines. Default is `:map`
       * `:evaluation-handler-fn` - Gets called once with the complete result of an individual pipeline evaluation.
           It can be used to adapt the data returned for each evaluation and / or to make side effects using
           the evaluatio data.
           The result of this function is taken as evaluation result. It need to  contain as a minumum this 2 key paths:
           [:train-transform :metric]
           [:test-transform :metric]
           All other evalution data can be removed, if desired.

           It can be used for side effects as well, like experiment tracking on disk.
           The passed in evaluation result is a map with all information on the current evaluation, including the datasets used.

           The default handler function is:  `scicloj.metamorph.ml/default-result-dissoc--in-fn` which removes the often large
           model object and the training data.
           `identity` can be use to get all evaluation data.
           `scicloj.metamorph.ml/result-dissoc-in-seq--all` reduces even more agressively.


  
       * `:other-metrics` Specifies other metrices to be calculated during evaluation

   This function expects as well the ground truth of the target variable into
   a specific key in the context at key `:model :scicloj.metamorph.ml/target-ds`
   See here for the simplest way to set this up: https://github.com/behrica/metamorph.ml/blob/main/README.md
   The function [[scicloj.ml.metamorph/model]] does this correctly.
  "

  {:malli/schema
   [:function
    [:=>
     [:cat
      :scicloj.metamorph.ml/optimize-hyperparams--pipeline-fn-or-decl-seq
      :scicloj.metamorph.ml/optimize-hyperparams--train-test-split-seq
      :scicloj.metamorph.ml/optimize-hyperparams--metric-fn
      :scicloj.metamorph.ml/optimize-hyperparams--loss-or-accuracy]
     :scicloj.metamorph.ml/optimize-hyperparams--evaluation-result]

    [:=>
     [:cat
      :scicloj.metamorph.ml/optimize-hyperparams--pipeline-fn-or-decl-seq
      :scicloj.metamorph.ml/optimize-hyperparams--train-test-split-seq
      :scicloj.metamorph.ml/optimize-hyperparams--metric-fn
      :scicloj.metamorph.ml/optimize-hyperparams--loss-or-accuracy
      :scicloj.metamorph.ml/evaluate-pipelines--options
      ]
     :scicloj.metamorph.ml/optimize-hyperparams--evaluation-result
     
     ]]}
  

  ([pipeline-fn-or-decl-seq train-test-split-seq metric-fn loss-or-accuracy options]
   (let [result (hyper-opt/optimize-hyperparameter
                 pipeline-fn-or-decl-seq
                 train-test-split-seq
                 {:metric metric-fn
                  :loss-or-accuracy loss-or-accuracy}
                 (assoc options :other-metrics
                        (mapv
                         (fn [{:keys [name metric-fn] :as m}]
                           {:name name
                            :metric-def {:metric metric-fn}})
                         (:other-metrics options))))]
     
     (for [r1 result] 
       (for [r2 r1]
         (if (some? (:metric-def r2))
           (-> r2
               (assoc :metric-fn (-> r2 :metric-def :metric))
               (dissoc :metric-def))
           r2)
         )
       )
     )
   )
  

  ([pipeline-fn-seq train-test-split-seq metric-fn loss-or-accuracy]
   (evaluate-pipelines pipeline-fn-seq train-test-split-seq metric-fn loss-or-accuracy {})))



(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an
  options map and returns a model.  A model is something that, combined with a dataset,
  produces a inferred dataset."
  {:malli/schema [:=> [:cat :keyword fn? fn? [:map
                                              [:hyperparameters {:optional true} [:maybe map?]]
                                              [:thaw-fn {:optional true} fn?]
                                              [:explain-fn {:optional true} fn?]
                                              [:loglik-fn {:optional true} fn?]
                                              [:tidy-fn {:optional true} fn?]
                                              [:augment-fn {:optional true} fn?]
                                              [:glance-fn {:optional true} fn?]
                                              [:options {:optional true} vector?]
                                              [:documentation {:optional true} [:map
                                                                                [:javadoc {:optional true} [:maybe string?]]
                                                                                [:user-guide {:optional true} [:maybe string?]]
                                                                                [:code-example {:optional true} [:maybe string?]]]]
                                              [:unsupervised? {:optional true} boolean?]]]
                  :keyword]}

  [model-kwd train-fn predict-fn {:keys [hyperparameters
                                         thaw-fn
                                         explain-fn
                                         loglik-fn
                                         tidy-fn
                                         glance-fn
                                         augment-fn
                                         options
                                         documentation
                                         unsupervised?]
                                  :as opts}]

  (println "Register model: " model-kwd)

  (malli/model-options->full-schema opts) ;; throws on invalid malli schema for options


  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn
                                             :loglik-fn loglik-fn
                                             :glance-fn glance-fn
                                             :tidy-fn tidy-fn
                                             :augment-fn augment-fn
                                             :options options
                                             :unsupervised? unsupervised?
                                             :documentation documentation})


  :ok)


(defn model-definition-names
  "Returns a list of all registered model definition names.

  Returns a sequence of keywords representing all model types that have been
  registered via `define-model!`. These can be used as the `:model-type` value
  when training models.

  Example: `[:metamorph.ml/dummy-classifier ...]`

  See also: `define-model!`, `options->model-def`"
  []
  (keys @model-definitions*))


(defn options->model-def
  "Retrieves the model definition corresponding to the `:model-type` option.

  `options` - Map containing at minimum a `:model-type` keyword

  Returns the model definition map registered for the given `:model-type`.
  Throws an exception if the model type is not found, suggesting a missing
  namespace require.

  Used internally to look up train/predict functions and model metadata.

  See also: `define-model!`, `model-definition-names`, `hyperparameters`"
  [options]
  {:pre [(contains? options :model-type)]}
  (if-let [model-def (get @model-definitions* (:model-type options))]
    model-def
    (errors/throwf "Failed to find model %s.  Is a require missing?" (:model-type options))))


(defn hyperparameters
  "Retrieves the hyperparameters definition for a model type.

  `model-kwd` - Keyword identifying the model type (e.g., `:smile.classification/random-forest`)

  Returns the hyperparameters map specified during model registration, or nil
  if no hyperparameters were defined. Hyperparameters describe tunable options
  for the model.

  Used for introspection and hyperparameter tuning/grid search.

  See also: `define-model!`, `options->model-def`"
  [model-kwd]
  (:hyperparameters (options->model-def {:model-type model-kwd})))

(defn- validate-options [model-options options]
  (let [options-schema (malli/model-options->full-schema model-options)]

    (when (not (m/validate options-schema options))
      (throw (ex-info "Invalid options: "
                      (->
                       (m/explain options-schema options)
                       (me/humanize)))))))

(defn- assert-categorical-consistency [dataset]
  (when (dataset? dataset)
    (let [distinc-datatypes
          (->> dataset ds-cat/dataset->categorical-maps
               (map #(->> % :lookup-table vals (map dt/datatype)))
               flatten
               distinct)]
      (assert (contains? #{0 1} 
                         (count distinc-datatypes)) (str "Non uniform cat map " (->> dataset ds-cat/dataset->categorical-maps vec)))
      (assert
       (or
        (empty? distinc-datatypes)
        (contains? #{:int :int32 :int64}
                   (first distinc-datatypes)))
       (str "Non :int cat map " (->> dataset ds-cat/dataset->categorical-maps vec)))))
  )

(defn- verify-train-fn-result! [model-data]
  ;(assert (some? model-data))
  )

(def prediction-column-meta-schema
  (atom
   [:map {:closed true}
    [:categorical? {:optional true} :boolean]
    [:column-type [:enum :prediction]]
    [:categorical-map {:optional true}
     [:maybe
      [:map {:closed true}
       [:lookup-table :map]
       [:src-column [:or :keyword :string]]
       [:result-datatype [:enum :int :int16 :int32 :int64 :float32 :float64]]]]]
    [:name :any]
    [:datatype [:enum :int :int16 :int32 :int64 :float32 :float64 :string :keyword]]
    [:n-elems :int]]))

(def probability-column-meta-schema
  (atom
   [:map {:closed true}
    [:name :any]
    [:datatype [:enum :float32 :float64]]
    [:n-elems :int]
    [:column-type [:enum :probability-distribution]]]))

(defn- validate-col-meta! [ds type schema model-type]
  (assert pos? (ds/column-count ds))
  (when 
   ;TODO : https://github.com/scicloj/scicloj.ml.xgboost/issues/10
   (not (= model-type :xgboost/classification))
    (run!
     #(let [column-meta (meta %)
            explanation
            (m/explain schema column-meta)]
        (when explanation
          (throw (ex-info (format "invalid model result schema of model type: %s" model-type)
                          {:column-meta column-meta
                           :type type
                           :schema schema
                           :malli-error (me/humanize explanation)}))))
     (ds/columns ds)))) 


  
(defn- validate-predict-fn-result! [pred-ds model-type]
  (assert (ds/dataset? pred-ds)
          (format "Prediction result should be 'dataset', but is: %s " (type pred-ds)))
  (assert (pos? (ds/column-count pred-ds))
          (format "Prediction result should have pos? nr of columns, but has: %s " (ds/column-count pred-ds)))

  (assert
   (pos?
    (-> pred-ds
        cf/prediction
        ds/column-count))
   (format "Prediction result should have pos? nr of 'prediction' columns, but has none "))


  (validate-col-meta!
   (cf/prediction pred-ds)
   "prediction"
   @prediction-column-meta-schema model-type)

  (validate-col-meta!
   (cf/probability-distribution pred-ds)
   "probability-distribution"
   @probability-column-meta-schema model-type))



(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns` - vector of column names.
  * `:target-columns` - vector of column names.
  * `:target-datatypes` - map of target columns names -> target columns type 
  * `:target-categorical-maps` - the categorical maps of the target columns, if present 
   
 A well behaving model implementaion should use 
   :target-column  
   :target-datatypes 
   :target-categorical-maps  
   
   to construct its prediction dataset so that its matches with the train data target column.
   "
  {:malli/schema [:=> [:cat 
                       [:fn (fn [x]
                              (or (dataset? x)
                                  (= (-> x class .getName) "ml.dmlc.xgboost4j.java.DMatrix")
                                  ))
                        ]
                       map?]
                  [map?]]}
  [data options]

  (assert-categorical-consistency data)


  (let [model-options (options->model-def options)
        _ (when (some? (:options model-options))
            (validate-options model-options options))

        combined-hash (when (:use-cache @train-predict-cache)
                        (str  (hash data) "___"  (hash options)))

        cached (when combined-hash ((:get-fn @train-predict-cache) combined-hash))]

    (if cached
      cached
      (let [{:keys [train-fn unsupervised?]} model-options
            model (cond (dataset? data)
                        (let [feature-ds (cf/feature  data)
                              _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                                       "No features provided")
                              target-ds (if unsupervised?
                                          nil
                                          (do
                                            (errors/when-not-error (> (ds/row-count (cf/target data)) 0) "No target columns provided, see tech.v3.dataset.modelling/set-inference-target")
                                            (cf/target data)))
                              model-data (train-fn feature-ds target-ds options)
                              _(verify-train-fn-result! model-data)

                              targets-datatypes
                              (zipmap
                               (keys target-ds)
                               (->>
                                (vals target-ds)
                                (map meta)
                                (map :datatype)))
                              cat-maps (ds-mod/dataset->categorical-xforms target-ds)
                              
                              
                              model
                              (merge
                               {:model-data model-data
                                :options options
                                :train-input-hash combined-hash
                                :id (UUID/randomUUID)
                                :feature-columns (vec (ds/column-names feature-ds))
                                :target-columns (vec (ds/column-names target-ds))
                                :target-datatypes targets-datatypes}
                               (when-not (== 0 (count cat-maps))
                                 {:target-categorical-maps cat-maps}))
                              ]
                          (when combined-hash
                            ((:set-fn @train-predict-cache) combined-hash model))
                          model
                          )
                        
                        (= (-> data class .getName) "ml.dmlc.xgboost4j.java.DMatrix")
                        (let [model-data (train-fn data nil options)]
                          {:model-data model-data
                           :options options
                           :feature-columns []
                           :target-columns []
                           }
                          )
                        
                        
                        :else (throw (ex-info (format "Unexpected dataset class: %s" (class data))
                                              {:dataset data}
                                              ))
                        )

            ;; _ (errors/when-not-error (:model-as-bytes model-data)  "train-fn need to return a map with key :model-as-bytes")

            ]
        

        model))))




(defn thaw-model
  "Thaws a frozen model for use in predictions.

  Models returned from `train` may be 'frozen' (serialized) for storage efficiency.
  A 'thaw' operation deserializes the model for use. This happens automatically
  during `predict`, but you can manually thaw and cache the model under
  `:thawed-model` for faster repeated predictions on small datasets.

  `model` - Model map from `train` containing `:model-data`
  `opts` - Optional map with `:thaw-fn` to override the model's thaw function

  Returns the thawed model data ready for prediction. If already thawed and
  cached, returns the cached version.

  See also: `train`, `predict`"
  {:malli/schema [:function
                  [:=> [:cat [:map [:model-data any?]] map?] map?]
                  [:=> [:cat [:map [:model-data any?]]] map?]]}



  ([model {:keys [thaw-fn] :as opts}]
   (if-let [cached-model (get model :thawed-model)]
     cached-model
     (if thaw-fn
       (thaw-fn (get model :model-data))
       (get model :model-data))))
  ([model]
   (let [thaw-fn
         (:thaw-fn
          (options->model-def (:options model)))]
     (thaw-fn (:model-data model)))))

(def enable-strict-prediction-validations
  "Atom controlling strict prediction validation behavior.

  When set to `true` (via `reset!` or `swap!`), enables validation that throws
  exceptions during prediction if:

  * Target categorical maps don't match between training and prediction datasets
  * Predicted values are not present in the prediction categorical map

  Defaults to `false` for backward compatibility. Set to `true` to catch
  potential prediction inconsistencies early.

  Example: `(reset! enable-strict-prediction-validations true)`"
  (atom false))

(defn- validate-inconsistent-maps [model pred-ds]
  (let [target-cat-maps-from-train (-> model :target-categorical-maps)
        target-cat-maps-from-predict (-> pred-ds get-categorical-maps)
        simple-predicted-values--int (map int (-> pred-ds cf/prediction (get (first (keys target-cat-maps-from-predict))) seq))
        inverse-predicted-map (-> target-cat-maps-from-predict vals first :lookup-table set/map-invert)]


      ;; should not throw  
    (when target-cat-maps-from-train
      (ds-cat/reverse-map-categorical-xforms pred-ds))

    (when
     (and @enable-strict-prediction-validations
          (not (= target-cat-maps-from-predict target-cat-maps-from-train))
          ;TODO : https://github.com/scicloj/scicloj.ml.xgboost/issues/10
          (not (= :xgboost/classification (-> model :options :model-type)))
          )

      (throw (Exception.
              (format
               "target categorical maps do not match between train and predict for model '%s' . \n train: %s \n predict: %s "
               (-> model :options :model-type) target-cat-maps-from-train target-cat-maps-from-predict))))





    (when
     (and @enable-strict-prediction-validations
          ;TODO : https://github.com/scicloj/scicloj.ml.xgboost/issues/10
          (not (= :xgboost/classification (-> model :options :model-type)))
          (or

           (not (every? some?
                        (map inverse-predicted-map
                             (distinct simple-predicted-values--int))))
           (not (=
                 (-> target-cat-maps-from-predict vals first :lookup-table keys)
                 (-> target-cat-maps-from-train vals first :lookup-table keys)))))
      (throw (Exception.
              (format
               "Some predicted values are not in prediction categorical map. Maybe invalid predict fn.
                            predicted values: %s
                            categorical map from predict: %s 
                            categorical map from train %s
    
                    "
               (vec (distinct simple-predicted-values--int))
               (-> target-cat-maps-from-predict)
               (-> target-cat-maps-from-train)))))))



(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the
    target
  * For classification, a dataset is returned with a float64 column for each target
    value and values that describe the probability distribution.
   
  Each implementing model should construct its prediction in a shape expressed by
   :target-column  
   :target-datatypes 
   :target-categorical-maps  

   it is receiving.

   Any implementing model need to behave symetric between the 'datatype in the target columns
   of training data' and the 'datatype of the prediction columns`
   A model can decide to not accept certain dataypes in the target columns of training data.
   (and fail with exception). But any model should try to minimize this and accept for categorical data:

   - all numeric types ( :int32, :int64, :float32, :float64)
   - string
   - categorical maps

   It NEED to be symetric, and return the same datatype in prediction as it receives in training:
   numeric in train -> same numeric in predict
   string in train -> string in predict
   categorical map in train -> equivalent categorical map in predict
   
   ml/train passes the needed information of the train target column to the model implementaion to do this.

   "
  {:malli/schema [:=> [:cat [:fn (fn [x]
                                   (or (dataset? x)
                                       (= (-> x class .getName) "ml.dmlc.xgboost4j.java.DMatrix")))]
                       [:map [:options map?]
                        [:feature-columns sequential?]
                        [:target-columns sequential?]]]

                  [map?]]}

  [dataset-or-dmatrix {:keys [feature-columns options train-input-hash]
                       :as model}]
  (when (ds/dataset? dataset-or-dmatrix )
    (assert-categorical-consistency dataset-or-dmatrix))
  (let [predict-hash (when (:use-cache @train-predict-cache) (str train-input-hash "--" (hash dataset-or-dmatrix)))
        cached (when predict-hash ((:get-fn @train-predict-cache) predict-hash))
        column-names (when (ds/dataset? dataset-or-dmatrix)
                       (ds/column-names dataset-or-dmatrix))
        feature-columns (if (empty? feature-columns) 
                          column-names
                          feature-columns)

        pred-ds
        (if cached
          cached

          (let [{:keys [predict-fn] :as model-def} (options->model-def options)

                feature-ds
                (if (dataset? dataset-or-dmatrix)
                  (ds/select-columns dataset-or-dmatrix feature-columns)
                  dataset-or-dmatrix)
                thawed-model (thaw-model model model-def)
                pred-ds (predict-fn feature-ds
                                    thawed-model
                                    model)]
            (validate-predict-fn-result! pred-ds (:model-type options))
            (validate-inconsistent-maps model pred-ds)

            (when predict-hash
              ((:set-fn @train-predict-cache) predict-hash pred-ds))

            pred-ds))]
    pred-ds))



(defn loglik
  "Calculates the log-likelihood for the given model and predictions.

  `model` - Trained model map containing `:options` with model definition
  `y` - Actual target values (ground truth)
  `yhat` - Predicted values from the model

  Returns the log-likelihood value by calling the model's `:loglik-fn` function.
  The specific log-likelihood function used depends on the model type.

  See also: `scicloj.metamorph.ml/tidy`, `scicloj.metamorph.ml/glance`"
  [model y yhat]
  (let [loglik-fn
        (get
         (options->model-def (:options model))
         :loglik-fn)]
    (loglik-fn y yhat)))

(defn tidy
  "summarizes information about model components.
  Returns a dataset with rows from this list:
 https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-tidy.edn

  No other row names should be used.
 Each model will only return a small subset of possible rows.
 The list of allowed row names might change over time.

 A model might not implement this function, and then an empty dataset will be returned.

  "
  [model]
  (let [tidy-fn
        (get
         (options->model-def (:options model))
         :tidy-fn)]

    (if tidy-fn
      (tidy/validate-tidy-ds
       (tidy-fn model))
      (ds/->dataset {}))))



(defn glance
  "Gives a glance on the model, returning a dataset with model information
  about the entire model.

  Potential row names are these:
  https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-glance.edn

 No other row names should be used.
 Each model will only return a small subset of possible rows.
 The list of allowed row names might change over time.

 A model might not implement this function, and then an empty dataset will be returned.
 "
  [model]
  (let [glance-fn
        (get
         (options->model-def (:options model))
         :glance-fn)]
    (if glance-fn
      (tidy/validate-glance-ds
       (glance-fn model))
      (ds/->dataset {}))))



(defn augment
  "
  Adds informations about observations to a dataset

  Potential row names are these:
  https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-augment.edn

 No other row names should be used.
 Each model will only return a small subset of possible rows.

  A model might not implement this function, and then the dataset is
  returned unchanged.

"
  [model data]

  (let [augment-fn
        (get
         (options->model-def (:options model))
         :augment-fn)]
    (if augment-fn
      (tidy/validate-augment-ds
       (augment-fn model data)
       data)
      data)))



(defn explain
  "Explain (if possible) an ml model.  A model explanation is a model-specific map
  of data that usually indicates some level of mapping between features and importance"
  {:malli/schema [:=> [:cat map? [:* any?]]
                  [map?]]}
  [model & [options]]
  (let [{:keys [explain-fn] :as model-def}
        (options->model-def (:options model))]
    (when explain-fn
      (explain-fn (thaw-model model model-def) model options))))



(defn default-loss-fn
  "Given a datset which must have exactly 1 inference target column return a default
  loss fn. If column is categorical, loss is tech.v3.ml.loss/classification-loss, else
  the loss is tech.v3.ml.loss/mae (mean average error)."
  {:malli/schema [:=> [:cat [:fn dataset?]]
                  [fn?]]}
  [dataset]
  (let [target-ds (cf/target dataset)]
    (errors/when-not-errorf
     (== 1 (ds/column-count target-ds))
     "Dataset has more than 1 target specified: %d"
     (ds/column-count target-ds))
    (if (:categorical? (meta (first (vals target-ds))))
      loss/classification-loss
      loss/mae)))


(defn model
  "Executes a machine learning model in train/predict (depending on :mode)
  from the `metamorph.ml` model registry.

  The model is passed between both invocation via the shared context ctx in a
  key (a step indentifier) which is passed in key `:metamorph/id` and guarantied to be unique for each
  pipeline step.

  The function writes and reads into this common context key.

  Options:
  - `:model-type` - Keyword for the model to use

  Further options get passed to `train` functions and are model specific.

  See here for an overview for the models build into scicloj.ml:


  https://scicloj.github.io/scicloj.ml-tutorials/userguide-models.html

  Other libraries might contribute other models,
  which are documented as part of the library.


  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Calls `scicloj.metamorph.ml/train` using data in `:metamorph/data` and `options`and stores trained model in ctx under key in `:metamorph/id`
  Behaviour in mode :transform         | Reads trained model from ctx and calls `scicloj.metamorph.ml/predict` with the model in $id and data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads trained model to use for prediction from key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores trained model in key $id and writes feature-ds and target-ds before prediction into ctx at `:scicloj.metamorph.ml/feature-ds` /`:scicloj.metamorph.ml/target-ds`




  See as well:

  * `scicloj.metamorph.ml/train`
  * `scicloj.metamorph.ml/predict`

  "
  {:malli/schema [:=> [:cat map?] [map?]]}
  [options]

  (malli/instrument-mm
   (fn [{:metamorph/keys [id data mode] :as ctx}]
     (case mode
       :fit (assoc ctx
                   id (assoc (train data options)
                             ::unsupervised? (get (options->model-def options) :unsupervised? false)))

       :transform  (if (get-in ctx [id ::unsupervised?])
                     ctx
                     (-> ctx
                         (update
                          id
                          assoc
                          ::feature-ds (cf/feature data)
                          ::target-ds (cf/target data))
                         (assoc
                          :metamorph/data (predict data (get ctx id)))))))))


(defn optimize-hyperparameter 
  "Optimize hyperparameter"
  {:malli/schema
   [:function
    [:=>
     [:cat
      :scicloj.metamorph.ml/optimize-hyperparams--pipeline-fn-or-decl-seq
      :scicloj.metamorph.ml/optimize-hyperparams--train-test-split-seq
      :scicloj.metamorph.ml/optimize-hyperparams--metric-def]
     :scicloj.metamorph.ml/evaluate-pipelines--evaluation-result]

    [:=>
     [:cat
      :scicloj.metamorph.ml/optimize-hyperparams--pipeline-fn-or-decl-seq
      :scicloj.metamorph.ml/optimize-hyperparams--train-test-split-seq
      :scicloj.metamorph.ml/optimize-hyperparams--metric-def
      :scicloj.metamorph.ml/optimize-hyperparams--options]
     :scicloj.metamorph.ml/evaluate-pipelines--evaluation-result]]}
  
  ([pipeline-fn-or-decl-seq train-test-split-seq metric-def options]
   (hyper-opt/optimize-hyperparameter pipeline-fn-or-decl-seq 
                                      train-test-split-seq 
                                      metric-def
                                      options))  
  ([pipeline-fn-or-decl-seq train-test-split-seq metric-def]
   (optimize-hyperparameter pipeline-fn-or-decl-seq
                            train-test-split-seq
                            metric-def
                            {}))  
  
  )


(malli/instrument-ns 'scicloj.metamorph.ml)


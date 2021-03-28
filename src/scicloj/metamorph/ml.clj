(ns scicloj.metamorph.ml
  (:require [pppmap.core :as ppp]
            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.functional :as dfn]
            [scicloj.metamorph.core]
            [tech.v3.datatype.export-symbols :as exporter]
            )
  (:import java.util.UUID))


(defn- dissoc-in
  "Dissociate a value in a nested assocative structure, identified by a sequence
  of keys. Any collections left empty by the operation will be dissociated from
  their containing structures."
  [m ks]
  (if-let [[k & ks] (seq ks)]
    (if (seq ks)
      (let [v (dissoc-in (get m k) ks)]
        (if (empty? v)
          (dissoc m k)
          (assoc m k v)))
      (dissoc m k))
    m))

(defn- slice
"Divide coll into n approximately equal slices."
  [n coll]
  (loop [num n, slices [], items (vec coll)]
    (if (empty? items)
      slices
      (let [size (Math/ceil (/ (count items) num))]
        (recur (dec num) (conj slices (subvec items 0 size)) (subvec items size))))))

(defn- calc-metric [pipeline-fn metric-fn train-ds test-ds result-dissoc-seq]
  ;; (def pipeline-fn pipeline-fn)
  ;; (def metric-fn metric-fn)
  ;; (def train-ds train-ds)
  ;; (def test-ds test-ds)
  (try
    (let [fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
          ;; _ (def fitted-ctx fitted-ctx)
          predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data test-ds}) )
          ;; _ (def predicted-ctx predicted-ctx)
          predictions (:metamorph/data predicted-ctx)
          target (cf/target (:metamorph/data fitted-ctx) )
          _ (errors/when-not-error target "No inference-target column marked in dataset")
          target-colname (first (ds/column-names target))
          ;; _ (def target-colname target-colname)
          true-target (get-in predicted-ctx [::target-ds target-colname])
          ;; _ (def true-target true-target)
          ;; _ (def predictions predictions)
          _ (errors/when-not-error true-target (str  "Pipeline context need to have the true prediction target as a dataset at key
 " ::target-ds "Maybe a `model` is missing in the pipeline.") )

          true-target-mapped-back
          (mapv
           (ds-mod/inference-target-label-inverse-map (::target-ds predicted-ctx) [target-colname])
           (map int true-target))

          predictions-mapped-back
          (mapv
           (ds-mod/inference-target-label-inverse-map predictions [target-colname])
           (map int (predictions target-colname)))

          ;; _ (def true-target-mapped-back true-target-mapped-back)
          ;; _ (def predictions-mapped-back predictions-mapped-back)

          metric (metric-fn predictions-mapped-back true-target-mapped-back)

          result
          {:fit-ctx  fitted-ctx ;; (dissoc fitted-ctx :metamorph/data)
           :transform-ctx  predicted-ctx ;; (dissoc predicted-ctx
                          ;;         :metamorph/data
                          ;;         :scicloj.metamorph.ml/target-ds
                          ;;         :scicloj.metamorph.ml/feature-ds
                          ;;         )
           :metric metric}]
      (reduce
          (fn [x y]
            (dissoc-in x y))
          result
          result-dissoc-seq)

      )
    (catch Exception e
      (throw e)
      (do
        (println e)
        {:fit-ctx nil
         :transfor-ctx nil
         :metric nil}))))


;; [[:fit-ctx :metamorph/data]
;;                          [:transform-ctx :metamorph/data]
;;                          [:transform-ctx :scicloj.metamorph.ml/target-ds]
;;                          [:transform-ctx :scicloj.metamorph.ml/feature-ds]
;;                          ]





(defn evaluate-pipeline [pipe-fn train-test-split-seq metric-fn loss-or-accuracy keep-best-only result-dissoc-seq]
  ;; (def train-test-split-seq train-test-split-seq)
  ;; (def pipe-fn-seq pipe-fn-seq)
  ;; (def metric-fn metric-fn)
  ;; (def loss-or-accuracy loss-or-accuracy)
  ;; (def keep-best-only keep-best-only)


  (let [split-eval-results
        (->>
         (for [train-test-split train-test-split-seq]
           (let [{:keys [train test]} train-test-split]
             (assoc (calc-metric pipe-fn metric-fn train test result-dissoc-seq)
                    :metric-fn metric-fn
                    :pipe-fn pipe-fn)))
         (remove #(nil? (:metric %))))
        metric-vec (mapv :metric split-eval-results)
        metric-vec-stats
        (dfn/descriptive-statistics [:min :max :mean] metric-vec)
        sorted-evaluations
        (->>
         (map
          #(merge % metric-vec-stats)
          split-eval-results)
         (sort-by :metric))]
    (if keep-best-only
      (case loss-or-accuracy
        :loss (take 1 sorted-evaluations)
        :accuracy (take-last 1 sorted-evaluations))
      sorted-evaluations)
    ))

(defn evaluate-pipelines
  "Evaluates performance of a seq of metamorph pipelines, which are suposed to have a  model as last step, which behaves correctly  in mode :fit and 
   :transform
   It calculates the loss, given as `loss-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits given in
   `train-test-split-seq`.

    It runs the pipelines  in mode  :fit and in mode :transform for each pipeline-fn in `pipe-fn-seq` for each split in `train-test-split-seq`.

   `pipe-fn-seq` need to be  sequence of functions which follow the metamorph approach. They should take as input the metamorph context map,
    which has the dataset under key :metamorph/data, manipulate it as needed for the transformation pipeline and read and write only to the
    context as needed.
   `train-test-split-seq` need to be a sequence of maps containing the  train and test dataset (being tech.ml.dataset) at keys :train and :test.
   `metric-fn` Metric function to use. Typically comming from `tech.v3.ml.loss`
   `loss-or-accuracy` If the metric-fn is a loss or accuracy calculation. Can be :loss or :accuracy.

    The next tune-options map controls varias performance related parameters, which are:

  `result-dissoc-seq`  - Controls how much information is returned for each cross validation. We call `dissoc-in`
  on every seq of this for the `fit-ctx` and `transform-ctx` before returning them. Default is
  ```
  [[:fit-ctx :metamorph/data]
   [:transform-ctx :metamorph/data]
   [:transform-ctx :scicloj.metamorph.ml/target-ds]
   [:transform-ctx :scicloj.metamorph.ml/feature-ds]
  ]
  ```

  `return-best-pipeline-only` - Only return information of the best performing pipeline. Default is true.
  `return-best-crossvalidation-only` - Only return information of the best crossvalidation (per pipeline returned). Default is true.
  `map-fn` Controls parralelism, so if we use map (:map) or pmap (:pmap) to map over different pipelines. Default :map



  This funtcion expects as well the ground truth of the target variable into
  a specific key in the context :scicloj.metamorph.ml/target-ds
  See here for the simplest way to set this up: https://github.com/behrica/metamorph.ml/blob/main/README.md

  The function `scicloj.metamorph.ml/model` does this correctly.
  "
  ([pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy & {:keys [result-dissoc-seq
                                                                          map-fn
                                                                          return-best-pipeline-only
                                                                          return-best-crossvalidation-only]
                                                                   :or {result-dissoc-seq
                                                                        [[:fit-ctx :metamorph/data]
                                                                         [:transform-ctx :metamorph/data]
                                                                         [:transform-ctx :scicloj.metamorph.ml/target-ds]
                                                                         [:transform-ctx :scicloj.metamorph.ml/feature-ds]
                                                                         ]
                                                                        map-fn :map
                                                                        return-best-pipeline-only true
                                                                        return-best-crossvalidation-only true}
                                                                   :as tune-options
                                                                   }]
   ;; (def tune-options tune-options)
   ;; (def train-test-split-seq train-test-split-seq)
   ;; (def pipe-fn-seq pipe-fn-seq)
   ;; (def metric-fn metric-fn)
   ;; (def loss-or-accuracy loss-or-accuracy)
   (let [map-fn
         (case map-fn
           :pmap (partial ppp/pmap-with-progress "pmap: evaluate pipelines ")
           :map (partial ppp/map-with-progress "map: evaluate pipelines"))
         pipe-evals
         (->> (map-fn #(evaluate-pipeline
                        %
                        train-test-split-seq
                        metric-fn
                        loss-or-accuracy
                        return-best-crossvalidation-only
                        result-dissoc-seq
                        )
                      pipe-fn-seq)
              (sort-by :mean))
         pipe-evals
         (if return-best-pipeline-only
           (case loss-or-accuracy
             :loss (take 1 pipe-evals)
             :accuracy (take-last 1 pipe-evals))
           pipe-evals)]

     ;; (def pipe-evals pipe-evals)
     (for [pipe-eval pipe-evals]
       (for [cv-eval pipe-eval]
         (reduce
          (fn [x y]
            (dissoc-in x y))
          cv-eval
          result-dissoc-seq)))))
  ;; ([pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy]
  ;;  (evaluate-pipelines pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy

  ;;                      ))
  )
(defn predict-on-best-model
  "Helper function for the very common case, to consider the pipeline with lowest average loss being the best.
   It allows to make a prediction on new data, given the list of all evaluation results.
  
   `evaluations` The list of pipeline-fn evaluations as returned from `evaluate-pipelines`.
   `new-ds` Dataset with the data to run teh best model from evaluations againts
   `loss-or-accuracy` : either :loss or :accuracy, if the metrics is loos or accuracy
 "
  [evaluations new-ds loss-or-accuracy]
  (let [sorted-evals
        (->>
         (group-by :pipe-fn evaluations)
         vals
         (map first)
         (sort-by :mean))

        evalution-with-best-avg-metric
        (case loss-or-accuracy
          :loss (first sorted-evals)
          :accuracy (last sorted-evals)
          )
        fitted-ctx (evalution-with-best-avg-metric :fit-ctx)
        target-column  (first (ds-mod/inference-target-column-names new-ds ))]
    (->   ((evalution-with-best-avg-metric :pipe-fn)
           (merge fitted-ctx
                  {:metamorph/data new-ds
                   :metamorph/mode :transform}))
          (:metamorph/data)
          (ds-mod/column-values->categorical target-column)
           seq)))



(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an
  options map and returns a model.  A model is something that, combined with a dataset,
  produces a inferred dataset."
  [model-kwd train-fn predict-fn {:keys [hyperparameters
                                         thaw-fn
                                         explain-fn
                                         options
                                         documentation
                                         ]}]
  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn
                                             :options options
                                             :documentation documentation

                                             })
  :ok)

(defn model-definition-names
  "Return a list of all registered model defintion names."
  []
  (keys @model-definitions*))


(defn options->model-def
  "Return the model definition that corresponse to the :model-type option"
  [options]
  (if-let [model-def (get @model-definitions* (:model-type options))]
    model-def
    (errors/throwf "Failed to find model %s.  Is a require missing?" (:model-type options))))


(defn hyperparameters
  "Get the hyperparameters for this model definition"
  [model-kwd]
  (:hyperparameters (options->model-def {:model-type model-kwd})))


(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns - vector of column names.
  * `:target-columns - vector of column names."
  [dataset options]
  (let [{:keys [train-fn]} (options->model-def options)

        feature-ds (cf/feature  dataset)
        _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                 "No features provided")
        target-ds (cf/target dataset)
        _ (errors/when-not-error (> (ds/row-count target-ds) 0)
                                 "No target columns provided
see tech.v3.dataset.modelling/set-inference-target")
        model-data (train-fn feature-ds target-ds options)
        cat-maps (ds-mod/dataset->categorical-xforms target-ds)]
    (merge
     {:model-data model-data
      :options options
      :id (UUID/randomUUID)
      :feature-columns (vec (ds/column-names feature-ds))
      :target-columns (vec (ds/column-names target-ds))}
     (when-not (== 0 (count cat-maps))
       {:target-categorical-maps cat-maps}))))


(defn thaw-model
  "Thaw a model.  Model's returned from train may be 'frozen' meaning a 'thaw'
  operation is needed in order to use the model.  This happens for you during preduct
  but you may also cached the 'thawed' model on the model map under the
  ':thawed-model'  keyword in order to do fast predictions on small datasets."
  ([model {:keys [thaw-fn]}]
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


(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the
    target
  * For classification, a dataset is returned with a float64 column for each target
    value and values that describe the probability distribution."
  [dataset model]
  (let [{:keys [predict-fn] :as model-def} (options->model-def (:options model))
        feature-ds (ds/select-columns dataset (:feature-columns model))
        label-columns (:target-columns model)
        thawed-model (thaw-model model model-def)
        pred-ds (predict-fn feature-ds
                            thawed-model
                            model)]

    (if (= :classification (:model-type (meta pred-ds)))
      (-> (ds-mod/probability-distributions->label-column
           pred-ds (first label-columns))
          (ds/update-column (first label-columns)
                            #(vary-meta % assoc :column-type :prediction)))
      pred-ds)))


(defn explain
  "Explain (if possible) an ml model.  A model explanation is a model-specific map
  of data that usually indicates some level of mapping between features and importance"
  [model & [options]]
  (let [{:keys [explain-fn] :as model-def}
        (options->model-def (:options model))]
    (when explain-fn
      (explain-fn (thaw-model model model-def) model options))))


(defn default-loss-fn
  "Given a datset which must have exactly 1 inference target column return a default
  loss fn. If column is categorical, loss is tech.v3.ml.loss/classification-loss, else
  the loss is tech.v3.ml.loss/mae (mean average error)."
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
  "Executes a machine learning model in train or predict
  from the `metamorph.ml` model registry.

  Options:
  - `:model-type` - Keyword for the model too use

  Further options get passed to `train` functions and are model specific.

  See here for an overview for the models build into Samskara:

  https://behrica.github.io/samskara/userguide-models.html

  Other libraries might contribute other models,
  which are documented as part of the library.


  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Calls `train` on given model and stores trained model in ctx
  Behaviour in mode :transform         | Reads trained model from ctx and calls `predict` on it
  Reads keys from ctx                  | Reads trained model to use for prediction from $id in mode `:transform`
  Writes keys to ctx                   | Stores trained model in key $id in mode `:fit` . Writes target-ds before prediction into `:scicloj.metamorph.ml/target-ds`




  See as well:

  * `scicloj.metamorph.ml/train`
  * `scicloj.metamorph.ml/predict`

  "

  [options]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
    (def data data)
    (def mode mode)
    (def ctx ctx)
    (def id id)
    (case mode
      :fit (assoc ctx id (train data options))
      :transform  (do
                    (assoc ctx
                           ::feature-ds (cf/feature data)
                           ::target-ds (cf/target data)
                           :metamorph/data (predict data (get ctx id)))))))
(comment
(sc.api/defsc 20)
 )

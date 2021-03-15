(ns scicloj.metamorph.ml
  (:require [pppmap.core :as ppp]
            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.functional :as dfn])
  (:import java.util.UUID))

(defn slice
  "Divide coll into n approximately equal slices."
  [n coll]
  (loop [num n, slices [], items (vec coll)]
    (if (empty? items)
      slices
      (let [size (Math/ceil (/ (count items) num))]
        (recur (dec num) (conj slices (subvec items 0 size)) (subvec items size))))))

(defn calc-ctx-with-metric [pipeline-fn metric-fn train-ds test-ds]
  (try
    (let [fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
          predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data test-ds}) )
          predictions (:metamorph/data predicted-ctx)
          target-colname (first (ds/column-names (cf/target (:metamorph/data fitted-ctx) )))
          true-target (get-in predicted-ctx [::target-ds target-colname])
          _ (errors/when-not-error true-target (str  "Pipeline context need to have the true prediction target as a dataset at key " ::target-ds))
          metric (metric-fn (predictions target-colname)
                            true-target)]
      {:fitted-ctx fitted-ctx
       :prediction-ctx predicted-ctx
       :metric metric})
    (catch Exception e
      (do
        (println e)
        {:fitted-ctx nil
         :prediction-ctx nil
         :metric nil}))
    ))


(defn evaluate-pipeline [pipe-fn train-test-split-seq metric-fn loss-or-accuracy]
  (let [split-eval-results
        (->>
         (for [train-test-split train-test-split-seq]
           (let [{:keys [train test]} train-test-split]
             (assoc (calc-ctx-with-metric pipe-fn metric-fn train test)
                    :metric-fn metric-fn
                    :pipe-fn pipe-fn)))
         (remove #(nil? (:metric %))))
        metric-vec (mapv :metric split-eval-results)
        _ (println metric-vec)
        metric-vec-stats (dfn/descriptive-statistics [:min :max :mean] metric-vec)
        sorted-evaluations
        (->>
         (map
          #(merge % metric-vec-stats)
          split-eval-results)
         (sort-by :metric))

        ]
    (case loss-or-accuracy
                       :loss (first sorted-evaluations)
                       :accuracy (last sorted-evaluations)
                       )
    )
  )

(defn evaluate-pipelines
  "Evaluates performance of a seq of metamorph pipelines, which are suposed to have a  model as last step, which behaves correctly  in mode :fit and 
   :transform
   It calculates the loss, given as `loss-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits given in
   `train-test-split-seq`.

   `pipe-fn-seq` need to be  sequence of functions which follow the metamorph approach. They should take as input the metamorph context map,
    which has the dataset under key :metamorph/data, manipulate it as needed for the transformation pipeline and read and write only to the
    context as needed.
   `train-test-split-seq` need to be a sequence of maps containing the  train and test dataset (being tech.ml.dataset) at keys :train and :test.
   `metric-fn` Metric function to use. Typically comming from `tech.v3.ml.loss`
   `loss-or-accuracy` If the metric-fn is a loss or accuracy calculation. Can be :loss or :accuracy.
   `n-slices`  Decides how many results are returned. By default one evaluation results for each pipeline-fn is returned.

    In case of a larger number of pipelines, this can become a memory issue. Setting n-slices to lower value, returns max n-slices models,
    where each model is the best (according to metyric-fn and loss-or-accuracy) of its slice.


  This function runs the pipeline  in mode  :fit and in mode :transform for each pipeline-fn in `pipe-fn-seq` for each split in `train-test-split-seq`.
  
  The pipeline-fns need to set as well the ground truth of the target variable into a specific key :scicloj.metamorph.ml/target-ds
  See here for the simplest way to set this up: https://github.com/behrica/metamorph.ml/blob/main/README.md
  "
  ([pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy n-slices]
   (->> (slice n-slices pipe-fn-seq)

        (ppp/pmap-with-progress
         "evaluate pipelines"
         (fn [pipe-fns]
           (let [sorted-evals
                 (->> (mapv #(evaluate-pipeline % train-test-split-seq metric-fn loss-or-accuracy) pipe-fns)
                      (sort-by (juxt :mean :metric)))]
             (case loss-or-accuracy
               :loss (first sorted-evals)
               :accuracy (last sorted-evals)))))
        doall))
  ([pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy]
   (evaluate-pipelines pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy Integer/MAX_VALUE))
  )

(defn predict-on-best-model [evaluations new-ds loss-or-accuracy]
  "Helper function for the very common case, to consider the pipeline with lowest average loss being the best.
   It allows to make a prediction on new data, given the list of all evaluation results.
  
   `evaluations` The list of pipeline-fn evaluations as returned from `evaluate-pipelines`.
   `new-ds` Dataset with the data to run teh best model from evaluations againts
   `loss-or-accuracy` : either :loss or :accuracy, if the metrics is loos or accuracy

 "
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
        fitted-ctx (evalution-with-best-avg-metric :fitted-ctx)
        target-column  (first (ds-mod/inference-target-column-names  (:metamorph/data fitted-ctx) ))]

  
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
]}]
  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn

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

(defn identity-preprocess [ds options]
  {:dataset ds
  :options options
     }
  )

(defn preprocess [dataset options]
  (let [fn-symbol  (or (options :preprocess-fn) 'scicloj.metamorph.ml/identity-preprocess)
        fun (requiring-resolve fn-symbol)]
    (fun dataset options)))

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

        preprocess-result (preprocess dataset options)
        preprocessed-dataset (:dataset preprocess-result)
        feature-ds (cf/feature  preprocessed-dataset)
        options (merge options (:options preprocess-result))
        _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                 "No features provided")
        target-ds (cf/target preprocessed-dataset)
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
        preprocess-result (preprocess dataset (:options model))
        dataset (:dataset preprocess-result)
        model (assoc model :options (merge (:options model) (:options preprocess-result)))
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

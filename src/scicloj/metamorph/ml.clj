(ns scicloj.metamorph.ml
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.dataset.modelling :as ds-mod]
            [pppmap.core :as ppp]))


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
                      ;; flatten
                      (sort-by (juxt :mean :metric))

                      )

                 ]
             (case loss-or-accuracy
               :loss (first sorted-evals)
               :accuracy (last sorted-evals)
               )
             )

           ))
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

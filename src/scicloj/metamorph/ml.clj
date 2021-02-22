(ns scicloj.metamorph.ml
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.dataset.modelling :as ds-mod]
            [pppmap.core :as ppp]))

(defn calc-ctx-with-loss [pipeline-fn loss-fn train-ds test-ds]
  (let [fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
        predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data test-ds}) )
        predictions (:metamorph/data predicted-ctx)
        target-colname (first (ds/column-names (cf/target (:metamorph/data fitted-ctx) )))
        true-target (get-in predicted-ctx [::target-ds target-colname])
        _ (errors/when-not-error true-target (str  "Pipeline context need to have the true prediction target as a dataset at key " ::target-ds))
        loss (loss-fn (predictions target-colname)
                      true-target)]
    {:fitted-ctx fitted-ctx
     :prediction-ctx predicted-ctx
     :loss loss}))




(defn evaluate-pipelines
  "Evaluates performance of a seq of metamorph pipelines, which are suposed to have a ml model as last step.
   It calculates the loss, given as `loss-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits given in
   `train-test-split-seq`.
   `pipeline-fn-seq` need to be a sequence of maps containing the  train and dataset (being tech.ml.dataset) at keys :train and :test.
   `pipe-fn-seq` need to be  sequence of functions which follow the metamorph approach. They should take as input the metamorph context map,
    which has the dataset under key :metamorph/data, manipulate it as needed for the transformation pipeline and read and write only to the
    context as needed.
  This function runs the pipeline ones in mode  :fit and ones in mode :transform.
  The pipeline-fn need to set as well the ground truth of the target variable into a key :scicloj.metamorph.ml/target-ds
  See here for the simples way to set this up: https://github.com/behrica/metamorph.ml/blob/main/README.md
  "
  [pipe-fn-seq
   train-test-split-seq
   loss-fn]
  (->> pipe-fn-seq
       (ppp/map-with-progress
        "evaluate pipelines"
        (fn [pipe-fn]
          (let [split-eval-results
                (for [train-test-split train-test-split-seq]
                  (let [{:keys [train test]} train-test-split]
                    (assoc (calc-ctx-with-loss pipe-fn loss-fn train test)
                           :loss-fn loss-fn
                           :pipe-fn pipe-fn)))

                loss-vec (mapv :loss split-eval-results)
                loss-vec-stats (dfn/descriptive-statistics [:min :max :mean] loss-vec)]
            (map
             #(merge % loss-vec-stats)
             split-eval-results))))

       flatten))


(defn predict-on-best-model [evaluations new-ds]
  "Helper function for the very common case, to consider the pipeline with lowest average loss being the best.
   It allows to make a prediction on new data, given the list of all evaluation results.
 "
  (let [evalution-with-lowest-avg-loss
        (->>
         (group-by :pipe-fn evaluations)
         vals
         (map first)
         (sort-by :avg-loss)
         (first))
        fitted-ctx (evalution-with-lowest-avg-loss :fitted-ctx)
        target-column  (first (ds-mod/inference-target-column-names  (:metamorph/data fitted-ctx) ))
        ]

  
    (->   ((evalution-with-lowest-avg-loss :pipe-fn)
           (merge fitted-ctx
                  {:metamorph/data new-ds
                   :metamorph/mode :transform}))
          (:metamorph/data)
          (ds-mod/column-values->categorical target-column)
          seq)))

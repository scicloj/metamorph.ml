(ns scicloj.metamorph.ml
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.ml.loss :as loss]
            [clojure.tools.logging :as log]
            [tech.v3.ml :as ml]
            [tech.v3.ml.gridsearch :as ml-gs]
            [pppmap.core :as ppp]
            [tablecloth.api.split :as split]
            ))



(defn calc-ctx-with-loss [pipeline-fn loss-fn train-ds test-ds]
  (let [fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
        predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data test-ds}) )
        predictions (:metamorph/data predicted-ctx)
        target-colname (first (ds/column-names (cf/target (:metamorph/data fitted-ctx) )))
        true-target (get-in predicted-ctx [:target-ds target-colname])
        _ (errors/when-not-error true-target "Pipeline context need to have the true prediction target as a dataset at key :target-ds")
        loss (loss-fn (predictions target-colname)
                      true-target)]
    {:fitted-ctx fitted-ctx
     :prediction-ctx predicted-ctx
     :loss loss}))


(defn evaluate-model [ds pipe-create-fn pipe-options
                      split-type split-options
                      loss-fn
                      {:keys [
                              n-gridsearch
                              n-result-models
                              ]
                       :or {
                            n-gridsearch 75
                            n-result-models 5}
                       }]
    (let [train-split-seq (split/split ds split-type split-options)
          gs-seq (take n-gridsearch (ml-gs/sobol-gridsearch pipe-options))

          evaluations
          (flatten
           (ppp/pmap-with-progress
            "gridsearch"
            (fn [options]
              (let [pipe-fn (pipe-create-fn options)]

                (map
                 (fn [{:keys [train test]}]
                   (assoc (calc-ctx-with-loss pipe-fn loss-fn train test)
                          :options options
                          :pipe-fn pipe-fn))
                 train-split-seq)))
            gs-seq))]
      (->>
       evaluations
       (sort-by :avg-loss)
       (take n-result-models)
       )
      ))





        ;; train-split-seq (split/split ds split-type split-options)
        ;; gs-seq (take n-gridsearch (ml-gs/sobol-gridsearch pipe-options))
        ;; pipe-fn-seq (map pipe-create-fn gs-seq )

(defn evaluate-model [pipe-fn-seq
                      train-test-split-seq
                      loss-fn]
  (->>
   (for [pipe-fn pipe-fn-seq train-test-split train-test-split-seq]
     (let [{:keys [train test]} train-test-split]
       (assoc (calc-ctx-with-loss pipe-fn loss-fn train test)
                :pipe-fn pipe-fn)))
   flatten
   (sort-by :loss)
   ))


(defn train-split
  "Train a model splitting the dataset using tech.v3.dataset.modelling/train-test-split
  and then calculate the loss using loss-fn.  Loss is added to the model map under :loss.
  * `loss-fn` defaults to loss/mae if target column is not categorical else defaults to
  loss/classification-loss."
  ([dataset pipeline-fn options loss-fn]
   (let [{:keys [train-ds test-ds]} (ds-mod/train-test-split dataset options)]
     (calc-ctx-with-loss pipeline-fn loss-fn train-ds test-ds)
     )) )


(defn do-k-fold [pipeline-fn loss-fn ds-seq]
  (let [models (mapv (fn [{:keys [train-ds test-ds]}]
                       (calc-ctx-with-loss pipeline-fn loss-fn train-ds test-ds))
                     ds-seq)
        loss-vec (mapv :loss models)
        _ (println loss-vec)

        {
         min-loss :min
         max-loss :max
         avg-loss :mean}
        (dfn/descriptive-statistics [:min :max :mean] loss-vec)
        min-model-idx (argops/argmin loss-vec)]
    (assoc (models min-model-idx)
           :min-loss min-loss
           :max-loss max-loss
           :avg-loss avg-loss
           :n-k-folds (count ds-seq))))




(defn train-k-fold
  "Train a model across k-fold datasets using tech.v3.dataset.modelling/k-fold-dataset
  and then calculate the min,max,and avg across results using loss-fn.  Adds
  :n-k-folds, :min-loss, :max-loss, :avg-loss and :loss (min-loss) to the
  model with the lowest loss.
  * `n-k-folds` defaults to 5.
  * `loss-fn` defaults to loss/mae if target column is not categorical else defaults to
     loss/classification-loss."

  ([dataset pipeline-fn n-k-folds loss-fn]
   (do-k-fold pipeline-fn loss-fn
              (ds-mod/k-fold-datasets dataset n-k-folds)
              ))
  ([dataset pipeline-fn n-k-folds]
   (train-k-fold pipeline-fn dataset  n-k-folds (ml/default-loss-fn dataset)))
  ([dataset pipeline-fn]
   (train-k-fold  pipeline-fn dataset 5 (ml/default-loss-fn dataset))))


(defn- pprint-to-string [o]
  (let [out (java.io.StringWriter.)]
    (clojure.pprint/pprint o out)
    (.toString out)))

(defn- safe-do-k-fold
  [pipeline-fn loss-fn ds-seq]
  (try
    (do-k-fold pipeline-fn loss-fn ds-seq)
    (catch Exception e
      (log/error e "Exception caught during grid search "))))



(defn train-auto-gridsearch
  "Train a model gridsearching across the options map.  The gridsearch map is built by
  merging the model's hyperparameter definitions into the options map.  If the sobol
  sequence returned has only one element a warning is issued.  Note this returns a
  sequence of models as opposed to a single model.
  * Searches across k-fold datasets if n-k-folds is > 1.  n-k-folds defaults to 5.
  * Searches (in parallel) through n-gridsearch option maps created via
    sobol-gridsearch.
  * Returns n-result-models (defaults to 5) sorted by avg-loss.
  * loss-fn can be provided or is the loss-fn returned via default-loss-fn."
  ([dataset pipeline-create-fn options {:keys [n-k-folds
                            n-gridsearch
                            n-result-models
                            loss-fn]
                     :or {n-k-folds 5
                          n-gridsearch 75
                          n-result-models 5}
                     :as gridsearch-options}]

   (let [gs-seq (take n-gridsearch (ml-gs/sobol-gridsearch options))
         _ (when (== 1 (count gs-seq))
             (log/warn "Did not find any gridsearch axis in options map"))
         ds-seq (ds-mod/k-fold-datasets dataset n-k-folds gridsearch-options)]
     (->> gs-seq
          (ppp/ppmap-with-progress "gridsearch" 1 #(safe-do-k-fold (pipeline-create-fn  %) loss-fn ds-seq))
          (sort-by :avg-loss)
          (take n-result-models)
          )))
  ([dataset pipeline-create-fn  options]
   (train-auto-gridsearch dataset pipeline-create-fn options nil)))

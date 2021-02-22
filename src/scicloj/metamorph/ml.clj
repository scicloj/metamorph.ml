(ns scicloj.metamorph.ml
  (:require [tech.v3.dataset :as ds]
           [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.datatype.errors :as errors]))

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



(defn evaluate-pipelines [pipe-fn-seq
                      train-test-split-seq
                      loss-fn]
  (->>
   (for [pipe-fn pipe-fn-seq ]
     (let [split-eval-result
           (for [train-test-split train-test-split-seq]
             (let [{:keys [train test]} train-test-split]
               (assoc (calc-ctx-with-loss pipe-fn loss-fn train test)
                      :pipe-fn pipe-fn)))

           loss-vec (mapv :loss split-eval-result)
           {min-loss :min
            max-loss :max
            avg-loss :mean} (dfn/descriptive-statistics [:min :max :mean] loss-vec)

           ]
       (map
        #(assoc %
                :min-loss min-loss
                :max-loss max-loss
                :avg-loss avg-loss)
        split-eval-result)))
   flatten))

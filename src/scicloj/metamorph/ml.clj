(ns scicloj.metamorph.ml
  (:require [tech.v3.dataset :as ds]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.datatype.errors :as errors]
            [pppmap.core :as ppp]
            ))

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

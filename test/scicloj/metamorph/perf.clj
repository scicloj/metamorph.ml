(ns scicloj.metamorph.perf
  (:require
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.ml.smile.classification]
   [tech.v3.dataset.metamorph :as ds-mm]
   [tech.v3.dataset.column-filters :as cf]
   [tablecloth.api :as tc]
   [clj-memory-meter.core :as mm]))


(defn measure [{:keys [n-pipes n-folds return-best-pipeline-only return-best-crossvalidation-only dissoc] :as opts}]
  (println "opts: " opts)
  (let [ds (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
        pipe-fn
        (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         {:metamorph/id :model}
         (ml/model {:model-type :smile.classification/random-forest}))

        train-split-seq (tc/split->seq ds :holdout {:repeats n-folds})
        pipe-fn-seq (repeat n-pipes pipe-fn)

        _ (println "dissoc: " dissoc)
        _ (println "seq: " (if (true? dissoc)  scicloj.metamorph.ml/default-result-dissoc-in-seq []))
        tune-options
        {
         :result-dissoc-in-seq (if dissoc  scicloj.metamorph.ml/default-result-dissoc-in-seq [])

         :return-best-pipeline-only return-best-pipeline-only
         :return-best-crossvalidation-only return-best-crossvalidation-only}

        _ (println "tune-opstions: " tune-options)
        _ (println "class : " (class dissoc))
        evaluations
        (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-accuracy :accuracy tune-options)]
    
    (mm/measure evaluations)))

(comment
  (measure {:n-pipes 1
            :n-folds 1
            :return-best-pipeline-only false
            :return-best-crossvalidation-only false
            :dissoc true}))


(defn run [args]
  (println "args: " args)
  (println "memory used:" (measure args)))
  

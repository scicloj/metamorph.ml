(ns scicloj.metamorph.to-disk-perf
  (:require
   [scicloj.ml.smile.classification]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   ))
  

(comment

  (def  ds (rdatasets/datasets-iris))
    (defn make-pipe-fn [options]

      (morph/pipeline
       (fn [ctx]
         (assoc ctx :pipe-options options))
       (ds-mm/set-inference-target :species)
       (ds-mm/categorical->number cf/categorical)
       (ml/model options)))

    (def gs-options
      {:trees (gs/categorical [10 50 100 500])
       :split-rule (gs/categorical [:gini :entropy])
       :model-type :smile.classification/random-forest})

    (def pipe-fn-seq
      (->>
       (gs/sobol-gridsearch gs-options)
       (take 100)
       (map make-pipe-fn)))

    (def train-split-seq (tc/split->seq ds))


    (def results
      (ml/evaluate-pipelines pipe-fn-seq train-split-seq  loss/classification-accuracy :accuracy
                             {:map-fn :pmap
                              :evaluation-handler-fn nippy-handler
                              :result-dissoc-in-seq ml/result-dissoc-in-seq--remove-ctxs
                              :return-best-pipeline-only true
                              :return-best-crossvalidation-only true}))
    :ok)




(comment
  (->>
   (file-seq (io/file "/tmp"))
   (filter #(str/ends-with? (.getPath %) ".nippy"))
   (map str)
   (map nippy/thaw-from-file)
   (map #(hash-map :pipe-options (get-in % [:fit-ctx :pipe-options])
                   :metric (get-in % [:train-transform :metric])))
   (group-by :pipe-options)
   (map (fn [[k v]]
          (def k k)
          (def v v)
          (hash-map
           :mean (stats/mean (map :metric v))
           :opts k)))
   (tc/dataset)
   println)


  (first results)
  (count (first results))

  :ok)

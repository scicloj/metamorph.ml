(ns scicloj.metamorph.ml.impl.hyper-opt
  (:require
   [clojure.string :as str]
   [pppmap.core :as ppp]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml.evaluation-handler :as eval-handler]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.datatype.functional :as dfn]
   [scicloj.metamorph.ml.column-metric :as col-metric]))


(defn- strict-type-check [trueth-col predictions-col]
  (when (not (=
              (-> trueth-col meta :datatype)
              (-> predictions-col meta :datatype)))))
;; (println (format
;;           "trueth-col and prediction-col do not have same datatype. trueth-col: %s prediction-col: %s"
;;           trueth-col predictions-col))




(defn- check-categorical-maps [trueth-ds prediction-ds target-column-name]
  (let [predict-cat-map (-> prediction-ds (get  target-column-name) meta :categorical-map)
        trueth-cat-map (-> trueth-ds (get  target-column-name) meta :categorical-map)]
    (when (not (= trueth-cat-map predict-cat-map)))))
;; (println
;;  "trueth-ds and prediction-ds do not have same categorical-map for target-column '%s'. trueth-ds-cat-map: %s prediction-ds-cat-map: %s"
;;  target-column-name (into {} trueth-cat-map) (into {} predict-cat-map))


(defn- score [predictions-ds trueth-ds target-column-name metric-def other-metrics]
  (def metric-def metric-def)
  (def other-metrics other-metrics)

  (let [predictions-col (get (ds-cat/reverse-map-categorical-xforms predictions-ds)
                             target-column-name)
        trueth-col (get (ds-cat/reverse-map-categorical-xforms trueth-ds)
                        target-column-name)
        metric-fn-or-kw  (:metric metric-def)

        _ (def trueth-col trueth-col)
        _ (def predictions-col predictions-col)
        _ (def metric-fn-or-kw metric-fn-or-kw)
        _ (def trueth-ds trueth-ds)
        _ (def predictions-ds predictions-ds)

        _ (strict-type-check trueth-col predictions-col)
        metric-value
        (if (fn? metric-fn-or-kw)
          (metric-fn-or-kw trueth-col predictions-col)
          (col-metric/classification-metric
           trueth-ds
           predictions-ds 
           metric-fn-or-kw 
           (:averaging metric-def)
           ))

        other-metrics-result
        (map
         (fn [{:keys [metric-fn] :as m}]
           (assoc m
                  :metric-fn (:metric-fn metric-fn)
                  :metric ((:metric-fn metric-fn) trueth-col predictions-col)))
         other-metrics)]
    {:metric metric-value
     :other-metrics-result other-metrics-result}))



(defn- supervised-eval-pipeline [pipeline-fn fitted-ctx metric-def ds other-metrics]

  (let [start-transform (System/currentTimeMillis)
        predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data ds}))
        end-transform (System/currentTimeMillis)

        predictions-ds (cf/prediction (:metamorph/data predicted-ctx))

        _ (errors/when-not-error predictions-ds "No column in prediction result was marked as 'prediction' ")
        _ (errors/when-not-error (:model predicted-ctx) "Pipelines need to have the 'model' op with id :model")
        trueth-ds (get-in predicted-ctx [:model :scicloj.metamorph.ml/target-ds])
        _ (errors/when-not-error trueth-ds (str  "Pipeline context need to have the true prediction target as a dataset at key path: "
                                                 :model :scicloj.metamorph.ml/target-ds " Maybe a `scicloj.metamorph.ml/model` step is missing in the pipeline."))

        target-column-names (ds/column-names trueth-ds)
        _ (errors/when-not-error (= 1 (count target-column-names)) "Only 1 target column is supported")


        target-column-name (first target-column-names)

        _ (errors/when-not-error (get predictions-ds target-column-name) (format "Prediction dataset need to have column name: %s " target-column-name))
        _ (check-categorical-maps trueth-ds predictions-ds target-column-name)


        scores (score predictions-ds trueth-ds target-column-name metric-def other-metrics)


        eval-result
        {:other-metrics (:other-metrics-result scores)
         :timing (- end-transform start-transform)
         :ctx predicted-ctx
         :probability-distribution (cf/probability-distribution (:metamorph/data predicted-ctx))
         :metric (:metric scores)}]
    eval-result))





(defn- eval-pipeline [pipeline-fn fitted-ctx metric ds other-metrics]

  (if  (-> fitted-ctx :model :scicloj.metamorph.ml/unsupervised?)
    {:other-metrics []
     :timing 0
     :ctx {}
     :metric (metric fitted-ctx)}

    (supervised-eval-pipeline pipeline-fn fitted-ctx metric ds other-metrics)))


(defn- calc-metric [pipeline-fn metric-def train-ds test-ds tune-options]
  (try
    (let [start-fit (System/currentTimeMillis)
          fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
          end-fit (System/currentTimeMillis)

          ;; TODO: double cec this, ensembles do not have it so far in "fit"
          #_(errors/when-not-error (:model fitted-ctx) "Pipeline contexts under evaluation need to have the model operation with id :model")


          eval-pipeline-result-train (eval-pipeline pipeline-fn fitted-ctx metric-def train-ds (:other-metrics tune-options))
          eval-pipeline-result-test (if (-> fitted-ctx :model :scicloj.metamorph.ml/unsupervised?)
                                      {:other-metrics []
                                       :timing 0
                                       :ctx fitted-ctx
                                       :metric 0}
                                      (eval-pipeline pipeline-fn fitted-ctx metric-def test-ds (:other-metrics tune-options)))]



      {:fit-ctx  fitted-ctx
       :timing-fit (- end-fit start-fit)
       :train-transform eval-pipeline-result-train
       :test-transform eval-pipeline-result-test})


    (catch Exception e
      (throw e)
      (do
        (println e)
        {:fit-ctx nil
         :transform-ctx nil
         :metric nil}))))


(defn- format-fn-sources [fn-sources]
  (->> fn-sources

       (filter #(let [v (val %)
                      code-source (:code-source v)
                      code-source-local (:code-local-source v)]
                  (or code-source code-source-local)))

       (map (fn [[k v]]
              {k
               (let [str-code
                     (str (:code-source v) (:code-local-source v))]
                 (if-not (str/blank? str-code)
                   {:source-str str-code
                    :source-form (read-string str-code)}
                   ""))}))
       (apply merge)))


(defn- get-nice-source-info [pipeline-decl pipeline-fns-ns pipeline-fns-source-file]
  (when (and  (some? pipeline-fns-ns) (some? pipeline-decl))
    (let [source-information (scicloj.metamorph.ml.evaluation-handler/get-source-information
                              pipeline-decl
                              pipeline-fns-ns
                              pipeline-fns-source-file)]
      (update source-information :fn-sources format-fn-sources))))


(defn- evaluate-one-pipeline [pipeline-decl-or-fn train-test-split-seq metric-def tune-options]

  (let [pipeline-fn (if (fn? pipeline-decl-or-fn)
                      pipeline-decl-or-fn
                      (mm/->pipeline pipeline-decl-or-fn))
        pipeline-decl (when (sequential? pipeline-decl-or-fn)
                        pipeline-decl-or-fn)
        
        loss-or-accuracy (:loss-or-accuracy metric-def)

        split-eval-results
        (->>
         (for [train-test-split train-test-split-seq]
           (let [{:keys [train test split-uid]} train-test-split
                 complete-result
                 (assoc (calc-metric pipeline-fn metric-def train test tune-options)
                        :split-uid split-uid
                        :loss-or-accuracy loss-or-accuracy
                        :metric-fn (:metric metric-def)
                        :pipe-decl pipeline-decl
                        :pipe-fn pipeline-fn
                        :source-information
                        (get-nice-source-info pipeline-decl
                                              (get-in tune-options [:attach-fn-sources :ns])
                                              (get-in tune-options [:attach-fn-sources :pipe-fns-clj-file])))

                 reduced-result ((tune-options :evaluation-handler-fn) complete-result)]
             reduced-result)))




        metric-vec-test (mapv #(get-in % [:test-transform :metric]) split-eval-results)
        metric-vec-train (mapv #(get-in % [:train-transform :metric]) split-eval-results)

        metric-vec-stats-test (dfn/descriptive-statistics metric-vec-test [:min :max :mean])
        metric-vec-stats-train (dfn/descriptive-statistics metric-vec-train [:min :max :mean])


        evaluations
        (->>
         (map
          #(-> %
               (update  :train-transform (fn [m] (merge m metric-vec-stats-train)))
               (update  :test-transform (fn [m] (merge m metric-vec-stats-test))))


          split-eval-results)
         (sort-by (comp :metric :test-transform) <))

        result
        (if (tune-options :return-best-crossvalidation-only)
          (case loss-or-accuracy
            :loss (->> evaluations  (take 1))
            :accuracy (->> evaluations  (take-last 1)))
          (case loss-or-accuracy
            :loss evaluations
            :accuracy (-> evaluations  reverse)))]
    result))







(defn optimize-hyperparameter

  ([pipeline-fn-or-decl-seq train-test-split-seq metric-def options]
   (let [used-options (merge {:map-fn :map
                              :return-best-pipeline-only true
                              :return-best-crossvalidation-only true
                              :evaluation-handler-fn eval-handler/default-result-dissoc-in-fn
                              :ppmap-grain-size 10}


                             options)
         map-fn
         (case (used-options :map-fn)
           :ppmap (partial ppp/ppmap-with-progress "ppmap: evaluate pipelines" (used-options :ppmap-grain-size))
           :pmap (partial ppp/pmap-with-progress "pmap: evaluate pipelines")
           :map (partial ppp/map-with-progress "map: evaluate pipelines")
           :mapv mapv)


         pipeline-evals
         (map-fn
          (fn [pipeline-fn-or-decl]
            (evaluate-one-pipeline
             pipeline-fn-or-decl
             train-test-split-seq
             metric-def
             used-options))
          pipeline-fn-or-decl-seq)

         pipeline-eval-means
         (->>
          (mapv
           (fn [pipe-eval]
             {:pipe-mean
              (dfn/mean
               (mapv (comp :metric :train-transform) pipe-eval))
              :pipe-eval pipe-eval})
           pipeline-evals)
          (sort-by :pipe-mean))


         result-pipeline-evals
         (if (used-options :return-best-pipeline-only)
           (case (:loss-or-accuracy metric-def)
             :loss     (->> pipeline-eval-means  (take 1) (map :pipe-eval))
             :accuracy (->> pipeline-eval-means  (take-last 1) (map :pipe-eval)))
           (case (:loss-or-accuracy metric-def)
             :loss     (->> pipeline-eval-means  (map :pipe-eval))
             :accuracy (->> pipeline-eval-means  reverse (mapv :pipe-eval))))]


     result-pipeline-evals))

  ([pipeline-fn-seq train-test-split-seq metric]
   (optimize-hyperparameter pipeline-fn-seq train-test-split-seq metric {})))


(ns scicloj.metamorph.ml
  (:require
   [clojure.set :as set]
   [clojure.string :as str]
   [pppmap.core :as ppp]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml.ensemble]
   [scicloj.metamorph.ml.evaluation-handler :as eval-handler]
   [scicloj.metamorph.ml.loss :as loss]
   [malli.core :as m]
   [malli.error :as me]
   [scicloj.metamorph.ml.malli :as malli]
   [scicloj.metamorph.ml.tidy-models :as tidy]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.datatype.export-symbols :as exporter]
   [tech.v3.datatype.functional :as dfn]
   [tech.v3.datatype :as dt])
  (:import
   java.util.UUID))


(exporter/export-symbols scicloj.metamorph.ml.ensemble ensemble-pipe)

(def train-predict-cache
  "Controls , if train/predict invocations are cached.
   if 'use-cache' is true, the get-fn and set-fn functions ar called accorddngly"
  (atom {:use-cache false
         :get-fn (fn [key] nil)
         :set-fn (fn [key value] nil)}))


(defn- get-categorical-maps [ds]
  (->> (ds/column-names ds)
       (map #(list % (-> ds (get %) meta :categorical-map)))
       (remove #(nil? (second %)))
       (map #(hash-map (first %) (second %)))
       (apply merge)))



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



(defn score [predictions-ds trueth-ds target-column-name metric-fn other-metrics]
  (let [predictions-col (get (ds-cat/reverse-map-categorical-xforms predictions-ds)
                             target-column-name)
        trueth-col (get (ds-cat/reverse-map-categorical-xforms trueth-ds)
                        target-column-name)

        _ (strict-type-check trueth-col predictions-col)
        metric (metric-fn trueth-col predictions-col)

        other-metrics-result
        (map
         (fn [{:keys [metric-fn] :as m}]
           (assoc m
                  :metric (metric-fn trueth-col predictions-col)))
         other-metrics)]
    {:metric metric
     :other-metrics-result other-metrics-result}))



(defn- supervised-eval-pipeline [pipeline-fn fitted-ctx metric-fn ds other-metrics]

  (let [start-transform (System/currentTimeMillis)
        predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data ds}))
        end-transform (System/currentTimeMillis)

        predictions-ds (cf/prediction (:metamorph/data predicted-ctx))

        _ (errors/when-not-error predictions-ds "No column in prediction result was marked as 'prediction' ")
        _ (errors/when-not-error (:model predicted-ctx) "Pipelines need to have the 'model' op with id :model")
        trueth-ds (get-in predicted-ctx [:model ::target-ds])
        _ (errors/when-not-error trueth-ds (str  "Pipeline context need to have the true prediction target as a dataset at key path: "
                                                 :model ::target-ds " Maybe a `scicloj.metamorph.ml/model` step is missing in the pipeline."))

        target-column-names (ds/column-names trueth-ds)
        _ (errors/when-not-error (= 1 (count target-column-names)) "Only 1 target column is supported")


        target-column-name (first target-column-names)

        _ (errors/when-not-error (get predictions-ds target-column-name) (format "Prediction dataset need to have column name: %s " target-column-name))
        _ (check-categorical-maps trueth-ds predictions-ds target-column-name)


        scores (score predictions-ds trueth-ds target-column-name metric-fn other-metrics)


        eval-result
        {:other-metrics (:other-metrics-result scores)
         :timing (- end-transform start-transform)
         :ctx predicted-ctx
         :probability-distribution (cf/probability-distribution (:metamorph/data predicted-ctx))
         :metric (:metric scores)}]
    eval-result))


(defn- eval-pipeline [pipeline-fn fitted-ctx metric-fn ds other-metrics]

  (if  (-> fitted-ctx :model ::unsupervised?)
    {:other-metrics []
     :timing 0
     :ctx {}
     :metric (metric-fn fitted-ctx)}

    (supervised-eval-pipeline pipeline-fn fitted-ctx metric-fn ds other-metrics)))


(defn- calc-metric [pipeline-fn metric-fn train-ds test-ds tune-options]
  (try
    (let [start-fit (System/currentTimeMillis)
          fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
          end-fit (System/currentTimeMillis)

          ;; TODO: double cec this, ensembles do not have it so far in "fit"
          #_(errors/when-not-error (:model fitted-ctx) "Pipeline contexts under evaluation need to have the model operation with id :model")


          eval-pipeline-result-train (eval-pipeline pipeline-fn fitted-ctx metric-fn train-ds (:other-metrics tune-options))
          eval-pipeline-result-test (if (-> fitted-ctx :model ::unsupervised?)
                                      {:other-metrics []
                                       :timing 0
                                       :ctx fitted-ctx
                                       :metric 0}
                                      (eval-pipeline pipeline-fn fitted-ctx metric-fn test-ds (:other-metrics tune-options)))]



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



(defn- evaluate-one-pipeline [pipeline-decl-or-fn train-test-split-seq metric-fn loss-or-accuracy tune-options]

  (let [pipeline-fn (if (fn? pipeline-decl-or-fn)
                      pipeline-decl-or-fn
                      (mm/->pipeline pipeline-decl-or-fn))
        pipeline-decl (when (sequential? pipeline-decl-or-fn)
                        pipeline-decl-or-fn)

        split-eval-results
        (->>
         (for [train-test-split train-test-split-seq]
           (let [{:keys [train test split-uid]} train-test-split
                 complete-result
                 (assoc (calc-metric pipeline-fn metric-fn train test tune-options)
                        :split-uid split-uid
                        :loss-or-accuracy loss-or-accuracy
                        :metric-fn metric-fn
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








(defn evaluate-pipelines
  "Evaluates the performance of a seq of metamorph pipelines, which are suposed to have a model as last step under key :model,
  which behaves correctly  in mode :fit and  :transform. The function `scicloj.metamorph.ml/model` is such function behaving correctly.
  
   This function calculates the accuracy or loss, given as `metric-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits
  given in  `train-test-split-seq`.

   It runs the pipelines  in mode  :fit and in mode :transform for each pipeline-fn in `pipe-fn-seq` for each split in `train-test-split-seq`.

   The function returns a seq of seqs of evaluation results per pipe-fn per train-test split.
   Each of the evaluation results is a context map, which is specified in the malli schema attached to this function.

   * `pipeline-fn-or-decl-seq` need to be  sequence of pipeline functions or pipline declarations which follow the metamorph approach.
      These type of functions get produced typically by calling `scicloj.metamorph/pipeline`. Documentation is here:

   * `train-test-split-seq` need to be a sequence of maps containing the  train and test dataset (being tech.ml.dataset) at keys :train and :test.
    `tablecloth.api/split->seq` produces such splits. Supervised models require both keys (:train and :test), while unsupervised models only use :train

   * `metric-fn` Metric function to use. Typically comming from `tech.v3.ml.loss`. For supervised models the metric-fn receives the trueth
      and predicted values and should return a single double number.  The metric fns receives a a seq *without* categorical maps. These
      get reverse-applied to the prediction , if present, before passing the values to the metriic fn.
      For unsupervised models the function receives the fitted ctx
      and should return a singel double number as well. This metric will be used to sort and eventualy filter the result, depending on the options
      (:return-best-pipeline-only   and :return-best-crossvalidation-only). The notion of `best` comes from metric-fn combined with loss-and-accuracy
  

   * `loss-or-accuracy` If the metric-fn is a loss or accuracy calculation. Can be :loss or :accuracy. Decided the notion of `best` model.
      In case of :loss pipelines with lower metric are better, in case of :accuracy pipelines with higher value are better.

  * `options` map controls some mainly performance related parameters. These function can potentialy result in a large ammount of data,
    able to bring the JVM into out-of-memory. We can control how many details the function returns by the following parameter: 
     The default are quite aggresive in removing details, and this can be tweaked further into more or less details via:
     


       * `:return-best-pipeline-only` - Only return information of the best performing pipeline. Default is true.
       * `:return-best-crossvalidation-only` - Only return information of the best crossvalidation (per pipeline returned). Default is `true`.
       * `:map-fn` - Controls parallelism, so if we use map (:map) , pmap (:pmap) or :mapv to map over different pipelines. Default is `:map`
       * `:evaluation-handler-fn` - Gets called once with the complete result of an individual pipeline evaluation.
           It can be used to adapt the data returned for each evaluation and / or to make side effects using
           the evaluatio data.
           The result of this function is taken as evaluation result. It need to  contain as a minumum this 2 key paths:
           [:train-transform :metric]
           [:test-transform :metric]
           All other evalution data can be removed, if desired.

           It can be used for side effects as well, like experiment tracking on disk.
           The passed in evaluation result is a map with all information on the current evaluation, including the datasets used.

           The default handler function is:  `scicloj.metamorph.ml/default-result-dissoc--in-fn` which removes the often large
           model object and the training data.
           `identity` can be use to get all evaluation data.
           `scicloj.metamorph.ml/result-dissoc-in-seq--all` reduces even more agressively.


  
       * `:other-metrics` Specifies other metrices to be calculated during evaluation

   This function expects as well the ground truth of the target variable into
   a specific key in the context at key `:model :scicloj.metamorph.ml/target-ds`
   See here for the simplest way to set this up: https://github.com/behrica/metamorph.ml/blob/main/README.md
   The function [[scicloj.ml.metamorph/model]] does this correctly.
  "

  {:malli/schema
   [:function
    {:registry
     {::options [:or empty? [:map
                             [:return-best-pipeline-only {:optional true} boolean?]
                             [:return-best-crossvalidation-only {:optional true} boolean?]
                             [:map-fn {:optional true} [:enum :map :pmap :mapv :ppmap]]
                             [:ppmap-grain-size {:optional true} int?]
                             [:evaluation-handler-fn {:optional true} fn?]
                             [:other-metrics {:optional true} [:sequential [:map
                                                                            [:name keyword?]
                                                                            [:metric-fn fn?]]]]
                             [:attach-fn-sources {:optional true} [:map [:ns any?]
                                                                   [:pipe-fns-clj-file string?]]]]]
      ::evaluation-result
      [:sequential
       [:sequential
        [:map {:closed true}
         [:split-uid [:maybe string?]]
         [:fit-ctx [:map [:metamorph/mode [:enum :fit :transform]]]]
         [:timing-fit int?]

         [:train-transform [:map {:closed true}
                            [:other-metrics [:sequential [:map {:closed true}
                                                          [:name keyword?]
                                                          [:metric-fn fn?]
                                                          [:metric float?]]]]
                            [:timing int?]
                            [:metric float?]
                            [:probability-distribution  [:maybe [:fn dataset?]]]
                            [:min float?]
                            [:mean float?]
                            [:max float?]
                            [:ctx map?]]]
         [:test-transform [:map {:closed true}
                           [:other-metrics [:sequential [:map {:closed true}
                                                         [:name keyword?]
                                                         [:metric-fn fn?]
                                                         [:metric float?]]]]
                           [:timing int?]
                           [:metric float?]
                           [:probability-distribution  [:maybe [:fn dataset?]]]
                           [:min float?]
                           [:mean float?]
                           [:max float?]
                           [:ctx map?]]]
         [:loss-or-accuracy [:enum :accuracy :loss]]
         [:metric-fn fn?]
         [:pipe-decl [:maybe sequential?]]
         [:pipe-fn fn?]
         [:source-information [:maybe [:map [:classpath [:sequential string?]]
                                       [:fn-sources [:map-of :qualified-symbol [:map [:source-form any?]
                                                                                [:source-str string?]]]]]]]]]]}}

    [:=>
     [:cat
      [:sequential [:or vector? fn?]]
      [:sequential [:map {:closed true}
                    [:split-uid {:optional true} string?]
                    [:train [:fn dataset?]]
                    [:test  {:optional true} [:fn dataset?]]]]
      fn?
      [:enum :accuracy :loss]]

     ::evaluation-result]
    [:=>
     [:cat
      [:sequential [:or vector? fn?]]
      [:sequential [:map {:closed true}
                    [:split-uid {:optional true} string?]
                    [:train [:fn dataset?]]
                    [:test {:optional true} [:fn dataset?]]]]
      fn?
      [:enum :accuracy :loss]
      ::options]

     ::evaluation-result]]}
  ;;

  ([pipeline-fn-or-decl-seq train-test-split-seq metric-fn loss-or-accuracy options]
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
             metric-fn
             loss-or-accuracy
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
           (case loss-or-accuracy
             :loss     (->> pipeline-eval-means  (take 1) (map :pipe-eval))
             :accuracy (->> pipeline-eval-means  (take-last 1) (map :pipe-eval)))
           (case loss-or-accuracy
             :loss     (->> pipeline-eval-means  (map :pipe-eval))
             :accuracy (->> pipeline-eval-means  reverse (mapv :pipe-eval))))]


     result-pipeline-evals))

  ([pipeline-fn-seq train-test-split-seq metric-fn loss-or-accuracy]
   (evaluate-pipelines pipeline-fn-seq train-test-split-seq metric-fn loss-or-accuracy {})))



(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an
  options map and returns a model.  A model is something that, combined with a dataset,
  produces a inferred dataset."
  {:malli/schema [:=> [:cat :keyword fn? fn? [:map
                                              [:hyperparameters {:optional true} [:maybe map?]]
                                              [:thaw-fn {:optional true} fn?]
                                              [:explain-fn {:optional true} fn?]
                                              [:loglik-fn {:optional true} fn?]
                                              [:tidy-fn {:optional true} fn?]
                                              [:augment-fn {:optional true} fn?]
                                              [:glance-fn {:optional true} fn?]
                                              [:options {:optional true} vector?]
                                              [:documentation {:optional true} [:map
                                                                                [:javadoc {:optional true} [:maybe string?]]
                                                                                [:user-guide {:optional true} [:maybe string?]]
                                                                                [:code-example {:optional true} [:maybe string?]]]]
                                              [:unsupervised? {:optional true} boolean?]]]
                  :keyword]}

  [model-kwd train-fn predict-fn {:keys [hyperparameters
                                         thaw-fn
                                         explain-fn
                                         loglik-fn
                                         tidy-fn
                                         glance-fn
                                         augment-fn
                                         options
                                         documentation
                                         unsupervised?]
                                  :as opts}]

  (println "Register model: " model-kwd)

  (malli/model-options->full-schema opts) ;; throws on invalid malli schema for options


  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn
                                             :loglik-fn loglik-fn
                                             :glance-fn glance-fn
                                             :tidy-fn tidy-fn
                                             :augment-fn augment-fn
                                             :options options
                                             :unsupervised? unsupervised?
                                             :documentation documentation})


  :ok)


(defn model-definition-names
  "Return a list of all registered model defintion names."
  []
  (keys @model-definitions*))


(defn options->model-def
  "Return the model definition that corresponse to the :model-type option"
  [options]
  {:pre [(contains? options :model-type)]}
  (if-let [model-def (get @model-definitions* (:model-type options))]
    model-def
    (errors/throwf "Failed to find model %s.  Is a require missing?" (:model-type options))))


(defn hyperparameters
  "Get the hyperparameters for this model definition"
  [model-kwd]
  (:hyperparameters (options->model-def {:model-type model-kwd})))

(defn- validate-options [model-options options]
  (let [options-schema (malli/model-options->full-schema model-options)]

    (when (not (m/validate options-schema options))
      (throw (ex-info "Invalid options: "
                      (->
                       (m/explain options-schema options)
                       (me/humanize)))))))

(defn- assert-categorical-consistency [dataset]
  (-> ;; should not throw
   (ds-cat/reverse-map-categorical-xforms dataset)
   (ds/categorical->number (ds/column-names dataset)))
  (let [distinc-datatypes
        (->> dataset ds-cat/dataset->categorical-maps
             (map #(->> % :lookup-table vals (map dt/datatype)))
             flatten
             distinct)]
    (assert (contains? #{0 1} 
               (count distinc-datatypes)) (str "Non uniform cat map " (->> dataset ds-cat/dataset->categorical-maps vec)))
    (assert
     (or
      (empty? distinc-datatypes)
      (contains? #{:int :int32 :int64}
                 (first distinc-datatypes)))
     (str "Non :int cat map " (->> dataset ds-cat/dataset->categorical-maps vec))))
  )






(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns` - vector of column names.
  * `:target-columns` - vector of column names.
  * `:target-datatypes` - map of target columns names -> target columns type 
  * `:target-categorical-maps` - the categorical maps of the target columns, if present 
   
 A well behaving model implementaion should use 
   :target-column  
   :target-datatypes 
   :target-categorical-maps  
   
   to construct its prediction dataset so that its matches with the train data target column.
   "
  {:malli/schema [:=> [:cat [:fn dataset?] map?]
                  [map?]]}
  [dataset options]

  (assert-categorical-consistency dataset)


  (let [model-options (options->model-def options)
        _ (when (some? (:options model-options))
            (validate-options model-options options))

        combined-hash (when (:use-cache @train-predict-cache)
                        (str  (hash dataset) "___"  (hash options)))

        cached (when combined-hash ((:get-fn @train-predict-cache) combined-hash))]

    (if cached
      cached
      (let [{:keys [train-fn unsupervised?]} model-options
            feature-ds (cf/feature  dataset)
            _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                     "No features provided")
            target-ds (if unsupervised?
                        nil
                        (do
                          (errors/when-not-error (> (ds/row-count (cf/target dataset)) 0) "No target columns provided, see tech.v3.dataset.modelling/set-inference-target")
                          (cf/target dataset)))
            model-data (train-fn feature-ds target-ds
                                 options)
        ;; _ (errors/when-not-error (:model-as-bytes model-data)  "train-fn need to return a map with key :model-as-bytes")
            targets-datatypes
            (zipmap
             (keys target-ds)
             (->>
              (vals target-ds)
              (map meta)
              (map :datatype)))
            cat-maps (ds-mod/dataset->categorical-xforms target-ds)

            model
            (merge
             {:model-data model-data
              :options options
              :train-input-hash combined-hash
              :id (UUID/randomUUID)
              :feature-columns (vec (ds/column-names feature-ds))
              :target-columns (vec (ds/column-names target-ds))
              :target-datatypes targets-datatypes}
             (when-not (== 0 (count cat-maps))
               {:target-categorical-maps cat-maps}))]
        (when combined-hash
          ((:set-fn @train-predict-cache) combined-hash model))

        model))))




(defn thaw-model
  "Thaw a model.  Model's returned from train may be 'frozen' meaning a 'thaw'
  operation is needed in order to use the model.  This happens for you during predict
  but you may also cached the 'thawed' model on the model map under the
  ':thawed-model'  keyword in order to do fast predictions on small datasets."
  {:malli/schema [:function
                  [:=> [:cat [:map [:model-data any?]] map?] map?]
                  [:=> [:cat [:map [:model-data any?]]] map?]]}



  ([model {:keys [thaw-fn] :as opts}]
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


(defn- warn-inconsistent-maps [model pred-ds]
  ;; TODO revise
  ;;https://github.com/scicloj/metamorph.ml/issues/35


  (let [target-cat-maps-from-train (-> model :target-categorical-maps)
        target-cat-maps-from-predict (-> pred-ds get-categorical-maps)
        simple-predicted-values (-> pred-ds cf/prediction (get (first (keys target-cat-maps-from-predict))) seq)
        inverse-predicted-map (-> target-cat-maps-from-predict vals first :lookup-table set/map-invert)]
    (when  (not (= target-cat-maps-from-predict target-cat-maps-from-train))
      
      (throw (Exception.
              (format
               "target categorical maps do not match between train an predict. \n train: %s \n predict: %s "
               target-cat-maps-from-train target-cat-maps-from-predict)))
      
      )
    
    (when (not (every? some?
                       (map inverse-predicted-map
                            (distinct simple-predicted-values))))
      (throw (Exception. 
              (format
               "Some predicted values are not in prediction categorical map. Maybe invalid predict fn.
                            predicted values: %s
                            categorical map from predict: %s 
                            categorical map from train %s
                    
                    "
               (vec (distinct simple-predicted-values))
               (-> target-cat-maps-from-predict)
               (-> target-cat-maps-from-train)
               )))
      
      )))



(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the
    target
  * For classification, a dataset is returned with a float64 column for each target
    value and values that describe the probability distribution.
   
  Each implementing model should construct its prediction in a shape expressed by
   :target-column  
   :target-datatypes 
   :target-categorical-maps  

   it is receiving.

   Any implementing model need to behave symetric between the 'datatype in the target columns
   of training data' and the 'datatype of the prediction columns`
   A model can decide to not accept certain dataypes in the target columns of training data.
   (and fail with exception). But any model should try to minimize this and accept for categorical data:

   - all numeric types ( :int32, :int64, :float32, :float64)
   - string
   - categorical maps

   It NEED to be symetric, and return the same datatype in prediction as it receives in training:
   numeric in train -> same numeric in predict
   string in train -> string in predict
   categorical map in train -> equivalent categorical map in predict
   
   ml/train passes the needed information of the train target column to the model implementaion to do this.

   "
  {:malli/schema [:=> [:cat [:fn dataset?]
                       [:map [:options map?]
                        [:feature-columns sequential?]
                        [:target-columns sequential?]]]

                  [map?]]}

  [dataset {:keys [feature-columns options train-input-hash]
            :as model}]
  (assert-categorical-consistency dataset)
  (let [predict-hash (when (:use-cache @train-predict-cache) (str train-input-hash "--" (hash dataset)))
        cached (when predict-hash ((:get-fn @train-predict-cache) predict-hash))

        pred-ds
        (if cached
          cached

          (let [{:keys [predict-fn] :as model-def} (options->model-def options)
                feature-ds (ds/select-columns dataset feature-columns)
                thawed-model (thaw-model model model-def)
                pred-ds (predict-fn feature-ds
                                    thawed-model
                                    model)]
            (warn-inconsistent-maps model pred-ds)

            (when predict-hash
              ((:set-fn @train-predict-cache) predict-hash pred-ds))

            pred-ds))]
    pred-ds))



(defn loglik [model y yhat]

  (let [loglik-fn
        (get
         (options->model-def (:options model))
         :loglik-fn)]
    (loglik-fn y yhat)))

(defn tidy
  "summarizes information about model components.
  Returns a dataset with rows from this list:
 https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-tidy.edn

  No other row names should be used.
 Each model will only return a small subset of possible rows.
 The list of allowed row names might change over time.

 A model might not implement this function, and then an empty dataset will be returned.

  "
  [model]
  (let [tidy-fn
        (get
         (options->model-def (:options model))
         :tidy-fn)]

    (if tidy-fn
      (tidy/validate-tidy-ds
       (tidy-fn model))
      (ds/->dataset {}))))



(defn glance
  "Gives a glance on the model, returning a dataset with model information
  about the entire model.

  Potential row names are these:
  https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-glance.edn

 No other row names should be used.
 Each model will only return a small subset of possible rows.
 The list of allowed row names might change over time.

 A model might not implement this function, and then an empty dataset will be returned.
 "
  [model]
  (let [glance-fn
        (get
         (options->model-def (:options model))
         :glance-fn)]
    (if glance-fn
      (tidy/validate-glance-ds
       (glance-fn model))
      (ds/->dataset {}))))



(defn augment
  "
  Adds informations about observations to a dataset

  Potential row names are these:
  https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-augment.edn

 No other row names should be used.
 Each model will only return a small subset of possible rows.

  A model might not implement this function, and then the dataset is
  returned unchanged.

"
  [model data]

  (let [augment-fn
        (get
         (options->model-def (:options model))
         :augment-fn)]
    (if augment-fn
      (tidy/validate-augment-ds
       (augment-fn model data)
       data)
      data)))



(defn explain
  "Explain (if possible) an ml model.  A model explanation is a model-specific map
  of data that usually indicates some level of mapping between features and importance"
  {:malli/schema [:=> [:cat map? [:* any?]]
                  [map?]]}
  [model & [options]]
  (let [{:keys [explain-fn] :as model-def}
        (options->model-def (:options model))]
    (when explain-fn
      (explain-fn (thaw-model model model-def) model options))))



(defn default-loss-fn
  "Given a datset which must have exactly 1 inference target column return a default
  loss fn. If column is categorical, loss is tech.v3.ml.loss/classification-loss, else
  the loss is tech.v3.ml.loss/mae (mean average error)."
  {:malli/schema [:=> [:cat [:fn dataset?]]
                  [fn?]]}
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
  "Executes a machine learning model in train/predict (depending on :mode)
  from the `metamorph.ml` model registry.

  The model is passed between both invocation via the shared context ctx in a
  key (a step indentifier) which is passed in key `:metamorph/id` and guarantied to be unique for each
  pipeline step.

  The function writes and reads into this common context key.

  Options:
  - `:model-type` - Keyword for the model to use

  Further options get passed to `train` functions and are model specific.

  See here for an overview for the models build into scicloj.ml:


  https://scicloj.github.io/scicloj.ml-tutorials/userguide-models.html

  Other libraries might contribute other models,
  which are documented as part of the library.


  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Calls `scicloj.metamorph.ml/train` using data in `:metamorph/data` and `options`and stores trained model in ctx under key in `:metamorph/id`
  Behaviour in mode :transform         | Reads trained model from ctx and calls `scicloj.metamorph.ml/predict` with the model in $id and data in `:metamorph/data`
  Reads keys from ctx                  | In mode `:transform` : Reads trained model to use for prediction from key in `:metamorph/id`.
  Writes keys to ctx                   | In mode `:fit` : Stores trained model in key $id and writes feature-ds and target-ds before prediction into ctx at `:scicloj.metamorph.ml/feature-ds` /`:scicloj.metamorph.ml/target-ds`




  See as well:

  * `scicloj.metamorph.ml/train`
  * `scicloj.metamorph.ml/predict`

  "
  {:malli/schema [:=> [:cat map?] [map?]]}
  [options]

  (malli/instrument-mm
   (fn [{:metamorph/keys [id data mode] :as ctx}]
     (case mode
       :fit (assoc ctx
                   id (assoc (train data options)
                             ::unsupervised? (get (options->model-def options) :unsupervised? false)))

       :transform  (if (get-in ctx [id ::unsupervised?])
                     ctx
                     (-> ctx
                         (update
                          id
                          assoc
                          ::feature-ds (cf/feature data)
                          ::target-ds (cf/target data))
                         (assoc
                          :metamorph/data (predict data (get ctx id)))))))))


(malli/instrument-ns 'scicloj.metamorph.ml)


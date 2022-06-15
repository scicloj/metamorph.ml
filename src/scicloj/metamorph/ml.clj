(ns scicloj.metamorph.ml
  (:require [pppmap.core :as ppp]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset.column :as ds-col]
            [scicloj.metamorph.core]
            [scicloj.metamorph.ml.evaluation-handler]
            [scicloj.metamorph.ml.tools :refer [dissoc-in]]
            [tech.v3.datatype.export-symbols :as exporter]
            [tech.v3.dataset.impl.dataset :refer [dataset?]]
            [scicloj.metamorph.ml.malli :as malli]
            [malli.core :as m])
  (:import java.util.UUID))





(defn- supervised-eval-pipe [pipeline-fn fitted-ctx metric-fn ds other-metrices]

  (let [
        start-transform (System/currentTimeMillis)
        predicted-ctx (pipeline-fn (merge fitted-ctx {:metamorph/mode :transform  :metamorph/data ds}))
        end-transform (System/currentTimeMillis)

        predictions-ds (cf/prediction (:metamorph/data predicted-ctx))

        _ (errors/when-not-error predictions-ds "No column in prediction result was marked as 'prediction' ")
        trueth-ds (get-in predicted-ctx [:model ::target-ds])
        _ (errors/when-not-error trueth-ds (str  "Pipeline context need to have the true prediction target as a dataset at key path: "
                                                 :model ::target-ds " Maybe a `scicloj.metamorph.ml/model` step is missing in the pipeline."))


        target-column-names (ds/column-names trueth-ds)
        _ (errors/when-not-error (= 1 (count target-column-names)) "Only 1 target column is supported")

        predictions-col (get predictions-ds (first target-column-names))
        trueth-col      (get trueth-ds      (first target-column-names))


        metric (metric-fn (ds-col/to-double-array trueth-col)
                          (ds-col/to-double-array predictions-col))

        other-metrices-result
        (map
         (fn [{:keys [name metric-fn] :as m}]
           (assoc m
                  :metric (metric-fn
                           (ds-col/to-double-array trueth-col)
                           (ds-col/to-double-array predictions-col))))
         other-metrices)
        eval-result
        {:other-metrices other-metrices-result
         :timing (- end-transform start-transform)
         :ctx predicted-ctx
         :metric metric}]
    eval-result))


(defn- eval-pipe [pipeline-fn fitted-ctx metric-fn ds other-metrices]

  (if  (-> fitted-ctx :model ::unsupervised?)
    {:other-metrices []
     :timing 0
     :ctx {}
     :metric (metric-fn fitted-ctx)}

    (supervised-eval-pipe pipeline-fn fitted-ctx metric-fn ds other-metrices)))


(defn- calc-metric [pipeline-fn metric-fn train-ds test-ds tune-options]
  (try
    (let [
          start-fit (System/currentTimeMillis)
          fitted-ctx (pipeline-fn {:metamorph/mode :fit  :metamorph/data train-ds})
          end-fit (System/currentTimeMillis)
          _ (errors/when-not-error (:model fitted-ctx) "Pipeline contexts under evaluation need to have the model operation with id :model")


          eval-pipe-result-train (eval-pipe pipeline-fn fitted-ctx metric-fn train-ds (:other-metrices tune-options))
          eval-pipe-result-test (if (-> fitted-ctx :model ::unsupervised?)
                                  {:other-metrices []
                                   :timing 0
                                   :ctx fitted-ctx
                                   :metric 0}
                                  (eval-pipe pipeline-fn fitted-ctx metric-fn test-ds (:other-metrices tune-options)))]
          

          
         {:fit-ctx  fitted-ctx
           :timing-fit (- end-fit start-fit)
           :train-transform eval-pipe-result-train
           :test-transform eval-pipe-result-test})


    (catch Exception e
      (throw e)
      (do
        (println e)
        {:fit-ctx nil
         :transform-ctx nil
         :metric nil}))))

(defn reduce-result [r result-dissoc-in-seq]
  (reduce (fn [x y]
            (dissoc-in x y))
          r
          result-dissoc-in-seq))
          






(defn format-fn-sources [fn-sources]
  (->> fn-sources

       (filter #(let [v (val %)
                      code-source (:code-source v)
                      code-source-local (:code-local-source v)]
                  (or code-source code-source-local)))

       (map (fn [[k v]]
              {k
               (let [str-code
                     (str (:code-source v) (:code-local-source v))]
                 (if-not (clojure.string/blank? str-code)
                   {:source-str str-code
                    :source-form (read-string str-code)}
                   ""))}))
       (apply merge)))



(defn get-nice-source-info [pipeline-decl pipe-fns-ns pipe-fns-source-file]
  (when (and  (some? pipe-fns-ns) (some? pipeline-decl))
    (let [source-information (scicloj.metamorph.ml.evaluation-handler/get-source-information
                              pipeline-decl
                              pipe-fns-ns
                              pipe-fns-source-file)]
      (update source-information :fn-sources format-fn-sources))))



(defn- evaluate-one-pipeline [pipeline-decl-or-fn train-test-split-seq metric-fn loss-or-accuracy tune-options]

  (let [

        pipe-fn (if (fn? pipeline-decl-or-fn)
                  pipeline-decl-or-fn
                  (mm/->pipeline pipeline-decl-or-fn))
        pipeline-decl (when (sequential? pipeline-decl-or-fn)
                        pipeline-decl-or-fn)

        split-eval-results
        (->>
         (for [train-test-split train-test-split-seq]
           (let [{:keys [train test]} train-test-split
                 complete-result
                 (assoc (calc-metric pipe-fn metric-fn train test tune-options)
                        :loss-or-accuracy loss-or-accuracy
                        :metric-fn metric-fn
                        :pipe-decl pipeline-decl
                        :pipe-fn pipe-fn
                        :source-information
                        (get-nice-source-info pipeline-decl
                                              (get-in tune-options [:attach-fn-sources :ns])
                                              (get-in tune-options [:attach-fn-sources :pipe-fns-clj-file])))

                 reduced-result ((tune-options :evaluation-handler-fn) complete-result)]
             reduced-result)))
             



        metric-vec-test (mapv #(get-in % [:test-transform :metric]) split-eval-results)
        metric-vec-train (mapv #(get-in % [:train-transform :metric]) split-eval-results)

        metric-vec-stats-test (dfn/descriptive-statistics [:min :max :mean] metric-vec-test)
        metric-vec-stats-train (dfn/descriptive-statistics [:min :max :mean] metric-vec-train)


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


      
    

(def default-result-dissoc-in-seq
  [[:fit-ctx :metamorph/data]

   [:train-transform :ctx :metamorph/data]

   [:train-transform :ctx :model :scicloj.metamorph.ml/target-ds]
   [:train-transform :ctx :model :scicloj.metamorph.ml/feature-ds]

   [:test-transform :ctx :metamorph/data]

   [:test-transform :ctx :model :scicloj.metamorph.ml/target-ds]
   [:test-transform :ctx :model :scicloj.metamorph.ml/feature-ds]
   ;;  scicloj.ml.smile specific
   [:train-transform :ctx :model :model-data :model-as-bytes]
   [:train-transform :ctx :model :model-data :smile-df-used]
   [:test-transform :ctx :model :model-data :model-as-bytes]
   [:test-transform :ctx :model :model-data :smile-df-used]])


(defn default-result-dissoc-in-fn [result]
  (reduce-result result default-result-dissoc-in-seq))


(def result-dissoc-in-seq--ctxs
  [[:fit-ctx]
   [:train-transform :ctx]
   [:test-transform :ctx]])

(defn result-dissoc-in-seq-ctx-fn [result]
  (reduce-result result result-dissoc-in-seq--ctxs))


(def result-dissoc-in-seq--all
  [[:metric-fn]
   [:fit-ctx]
   [:train-transform :ctx]
   [:train-transform :other-metrices]
   [:train-transform :timing]
   [:test-transform :ctx]
   [:test-transform :other-metrices]
   [:test-transform :timing]
   [:pipe-decl]
   [:pipe-fn]
   [:timing-fit]
   [:loss-or-accuracy]
   [:source-information]])

(defn result-dissoc-in-seq--all-fn [result]
  (reduce-result result result-dissoc-in-seq--all))







(defn evaluate-pipelines
  "Evaluates the performance of a seq of metamorph pipelines, which are suposed to have a model as last step under key :model,
  which behaves correctly  in mode :fit and  :transform. The function `scicloj.metamorph.ml/model` is such function behaving correctly.
  
   This function calculates the accuracy or loss, given as `metric-fn` of each pipeline in `pipeline-fn-seq` using all the train-test splits
  given in  `train-test-split-seq`.

   It runs the pipelines  in mode  :fit and in mode :transform for each pipeline-fn in `pipe-fn-seq` for each split in `train-test-split-seq`.

   The function returns a seq of seqs of evaluation results per pipe-fn per train-test split.
   Each of teh evaluation results is a context map, which is specified in the malli schema attached to this function. 

   * `pipe-fn-or-decl-seq` need to be  sequence of pipeline functions or pipline declarations which follow the metamorph approach.
      These type of functions get produced typically by calling `scicloj.metamorph/pipeline`. Documentation is here:

   * `train-test-split-seq` need to be a sequence of maps containing the  train and test dataset (being tech.ml.dataset) at keys :train and :test.
    `tableclot.api/split->seq` produces such splits. Supervised models require both keys (:train and :test), while unsupervised models only use :train

   * `metric-fn` Metric function to use. Typically comming from `tech.v3.ml.loss`. For supervised models the metric-fn receives the trueth
      and predicted vales as double arrays and should return a single double number.  For unsupervised models he function receives the fitted ctx
      and should return a singel double number as well. This metric will be used to sort and eventualy filter the result, depending on the options
      (:return-best-pipeline-only   and :return-best-crossvalidation-only). The notion of `best` comes from metric-fn combined with loss-and-accuracy
  

   * `loss-or-accuracy` If the metric-fn is a loss or accuracy calculation. Can be :loss or :accuracy. Decided the notion of `best` model.
      In case of :loss pipelines with lower metric are better, in case of :accuracy pipelines with higher value are better.

  * `options` map controls some mainly performance related parameters. These function can potentialy result in a large ammount of data,
    able to bring the JVM into out-of-memory. We can control how many details the function returns by the following parameter: 
     The default are quite aggresive in removing details, and this can be tweaked further into more or less details via:
     


       * `:return-best-pipeline-only` - Only return information of the best performing pipeline. Default is true.
       * `:return-best-crossvalidation-only` - Only return information of the best crossvalidation (per pipeline returned). Default is true.
       * `:map-fn` - Controls parallelism, so if we use map (:map) , pmap (:pmap) or :mapv to map over different pipelines. Default :pmap
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


  
       * `:other-metrices` Specifies other metrices to be calculated during evaluation

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
                             [:map-fn {:optional true} [:enum :map :pmap :mapv]]
                             [:evaluation-handler-fn {:optional true} fn?]
                             [:other-metrices {:optional true} [:sequential [:map
                                                                             [:name keyword?]
                                                                             [:metric-fn fn?]]]]
                             [:attach-fn-sources {:optional true} [:map [:ns any?]
                                                                   [:pipe-fns-clj-file string?]]]]]
      ::evaluation-result
      [:sequential
       [:sequential
        [:map {:closed true}
         [:fit-ctx [:map [:metamorph/mode [:enum :fit :transform]]]]
         [:timing-fit int?]

         [:train-transform [:map {:closed true}
                            [:other-metrices [:sequential [:map {:closed true}
                                                           [:name keyword?]
                                                           [:metric-fn fn?]
                                                           [:metric float?]]]]
                            [:timing int?]
                            [:metric float?]
                            [:min float?]
                            [:mean float?]
                            [:max float?]
                            [:ctx map?]]]
         [:test-transform [:map {:closed true}
                           [:other-metrices [:sequential [:map {:closed true}
                                                          [:name keyword?]
                                                          [:metric-fn fn?]
                                                          [:metric float?]]]]
                           [:timing int?]
                           [:metric float?]
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
      [:sequential [:map {:closed true} [:train [:fn dataset?]] [:test {:optional true} [:fn dataset?]]]]
      fn?
      [:enum :accuracy :loss]]

     ::evaluation-result]
    [:=>
     [:cat
      [:sequential [:or vector? fn?]]
      [:sequential [:map {:closed true} [:train [:fn dataset?]] [:test {:optional true} [:fn dataset?]]]]
      fn?
      [:enum :accuracy :loss]
      ::options]

     ::evaluation-result]]}
  ;;

  ([pipe-fn-or-decl-seq train-test-split-seq metric-fn loss-or-accuracy options]
   (let [used-options (merge {:map-fn :map
                              :return-best-pipeline-only true
                              :return-best-crossvalidation-only true
                              :evaluation-handler-fn default-result-dissoc-in-fn}
                         
                             options)
         map-fn
         (case (used-options :map-fn)
           :pmap (partial ppp/pmap-with-progress "pmap: evaluate pipelines ")
           :map (partial ppp/map-with-progress "map: evaluate pipelines")
           :mapv mapv)


         pipe-evals
         (map-fn
          (fn [pipe-fn-or-decl]
            (evaluate-one-pipeline
             pipe-fn-or-decl
             train-test-split-seq
             metric-fn
             loss-or-accuracy
             used-options))
          pipe-fn-or-decl-seq)

         pipe-eval-means
         (->>
          (mapv
           (fn [pipe-eval]
             {:pipe-mean
              (dfn/mean
               (mapv (comp :metric :train-transform) pipe-eval))
              :pipe-eval pipe-eval})
           pipe-evals)
          (sort-by :pipe-mean))


         result-pipe-evals
         (if (used-options :return-best-pipeline-only)
           (case loss-or-accuracy
             :loss     (->> pipe-eval-means  (take 1) (map :pipe-eval))
             :accuracy (->> pipe-eval-means  (take-last 1) (map :pipe-eval)))
           (case loss-or-accuracy
             :loss     (->> pipe-eval-means  (map :pipe-eval))
             :accuracy (->> pipe-eval-means  reverse (mapv :pipe-eval))))]


     result-pipe-evals))

  ([pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy]
   (evaluate-pipelines pipe-fn-seq train-test-split-seq metric-fn loss-or-accuracy {})))



(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an
  options map and returns a model.  A model is something that, combined with a dataset,
  produces a inferred dataset."
  [model-kwd train-fn predict-fn {:keys [hyperparameters
                                         thaw-fn
                                         explain-fn
                                         options
                                         documentation
                                         unsupervised?]}]

  (println "Register model: " model-kwd)
  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn
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


(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns` - vector of column names.
  * `:target-columns` - vector of column names."
  {:malli/schema [:=> [:cat [:fn dataset?] map?]
                  [map?]]}
  [dataset options]
  (let [{:keys [train-fn unsupervised?]} (options->model-def options)
        feature-ds (cf/feature  dataset)
        _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                 "No features provided")
        target-ds (if unsupervised?
                    nil
                    (do
                      (errors/when-not-error (> (ds/row-count (cf/target dataset)) 0) "No target columns provided, see tech.v3.dataset.modelling/set-inference-target")
                      (cf/target dataset)))

                     
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
  operation is needed in order to use the model.  This happens for you during predict
  but you may also cached the 'thawed' model on the model map under the
  ':thawed-model'  keyword in order to do fast predictions on small datasets."
  {:malli/schema [:function
                  [:=> [:cat [:map [:model-data any?]] map?] map?]
                  [:=> [:cat [:map [:model-data any?]]] map?]]}



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




(defn lookup-tables-consistent? [train-lookup-table prediction-lookup-table]
  ;; simplification
  ;; TODO find better way
  (= train-lookup-table prediction-lookup-table))

(defn validate-lookup-tables [model predict-ds-classification target-col]
  (let [
        train-lookup-table (-> model :target-categorical-maps (get target-col) :lookup-table)

        prediction-lookup-table (-> predict-ds-classification (get target-col) meta :categorical-map :lookup-table)]
    ;;  check consistency of the lookup tables
    ;;  having this violated, likley mean that the model implementation did something wrong

    (errors/when-not-error (lookup-tables-consistent? train-lookup-table prediction-lookup-table)

                           (str  "The lookup tables of the train-target column and prediction lable column are not consistent: "
                                 train-lookup-table " vs. " prediction-lookup-table))))


(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the
    target
  * For classification, a dataset is returned with a float64 column for each target
    value and values that describe the probability distribution."
  {:malli/schema [:=> [:cat [:fn dataset?]
                       [:map [:options map?]
                        [:feature-columns sequential?]
                        [:target-columns sequential?]]]
                       

                  [map?]]}
  [dataset model]
  (let [{:keys [predict-fn] :as model-def} (options->model-def (:options model))
        feature-ds (ds/select-columns dataset (:feature-columns model))
        label-columns (:target-columns model)
        thawed-model (thaw-model model model-def)
        target-col (first label-columns)
        pred-ds (predict-fn feature-ds
                            thawed-model
                            model)]
    pred-ds))




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



(comment
  (require '[malli.dev.pretty :as pretty])
  (m/explain
   [:cat {:registry {::blub string?}} ::blub ::blub]
   ["a" "b"])

  (mi/unstrument!)


  (require '[malli.dev :as dev])

  (dev/start! {:report (pretty/reporter)})

  :ok)


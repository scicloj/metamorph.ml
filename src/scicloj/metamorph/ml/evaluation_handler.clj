(ns scicloj.metamorph.ml.evaluation-handler
  (:require
   [clojure.java.classpath]
   [clojure.repl :as repl]
   [clojure.tools.reader :as tr]
   [clojure.tools.reader.reader-types :as rts]
   [clojure.walk :as walk]
   [scicloj.metamorph.ml.tools :refer [dissoc-in multi-dissoc-in pp-str reduce-result]]
   [taoensso.nippy :as nippy]))

(defn- file->topforms-with-metadata [path]
  (->> path
       slurp
       rts/source-logging-push-back-reader
       repeat
       (map #(tr/read % false :EOF))
       (take-while (partial not= :EOF))))

(defn- resolve-keyword
  "Interpret keyword or list as a symbol and try to resolve it."
  [k pipe-ns]
  (if (or  (sequential? k) (map? k))
    k
    (-> (if-let [n (namespace k)] ;; namespaced?
          (let [sn (symbol n)
                n (str (get (ns-aliases pipe-ns) sn sn))] ;; try to find namespace in aliases
            (symbol n (name k))) ;; create proper symbol with fixed namespace
          (symbol (str pipe-ns) (name k))) ;; no namespace case
        (resolve))))




(defn- fns->code-list [pipeline-source-file]
  (->>
   (file->topforms-with-metadata pipeline-source-file)

   (map (fn [form]
          (when (sequential? form)
            (hash-map :top-level (first form)
                      :form form
                      :form-str (pp-str form)))))
   (filter some?)
   (filter #(= 'defn (:top-level %)))
   (map #(assoc % :fn-name (second (:form  %))))))


(defn- fns->code-map [pipeline-source-file]
  (let [fns-code-list (fns->code-list pipeline-source-file)]
    (zipmap
     (map :fn-name fns-code-list)
     fns-code-list)))

(defn- fn-symbol->code [fn-symbol pipe-ns pipeline-source-file]
  (:form-str
   (get (fns->code-map pipeline-source-file)
        (:name
         (meta
          (resolve-keyword  fn-symbol pipe-ns))))))


(defn- get-code [symbol pipe-ns pipeline-source-file]
  (let [source (repl/source-fn symbol)
        orig-source (some-> symbol
                         resolve
                         meta
                         :orig
                         clojure.repl/source-fn)]


    {:code-source (if orig-source orig-source source)
     :code-local-source (fn-symbol->code symbol pipe-ns pipeline-source-file)}))

(defn- get-classpath []
  (->>
   (clojure.java.classpath/classpath)
   (map #(.getPath %))))

(defn- get-fn-sources [qualified-pipe-decl pipe-ns pipeline-source-file]
  (let [codes (atom {})]
    (walk/postwalk (fn [keyword]
                             (when (keyword? keyword)
                               (let [symbol (symbol keyword)]
                                 (swap! codes #(assoc % symbol
                                                      (get-code symbol pipe-ns pipeline-source-file)))))
                             keyword)
                           qualified-pipe-decl)
    @codes))

(defn get-source-information
  "Creates metadata about a pipeline including function sources and classpath.

  `qualified-pipe-decl` - Pipeline declaration with fully-qualified keywords
  `pipe-ns` - Namespace symbol for the pipeline
  `pipeline-source-file` - Path to the source file containing the pipeline

  Returns a map with `:fn-sources` (source code of all pipeline functions) and
  `:classpath` (JVM classpath at evaluation time)."
  [qualified-pipe-decl pipe-ns pipeline-source-file]
  {:fn-sources (get-fn-sources qualified-pipe-decl pipe-ns pipeline-source-file)
   :classpath (get-classpath)})

(defn example-nippy-handler
  "Creates an evaluation result handler that serializes results to Nippy format.

  `files` - Atom to track generated file paths
  `output-dir` - Directory path for output files
  `result-reduce-fn` - Function to apply to results after serialization

  Returns a handler function that removes non-freezable data (`:pipe-fn`, `:metric-fn`),
  serializes the result to a UUID-named .nippy file, and applies `result-reduce-fn`.

  See also: `scicloj.metamorph.ml/evaluate-pipelines`"
  [files output-dir result-reduce-fn]
  (fn [result]
    (let [
          freezable-result
          (-> result
              (multi-dissoc-in
               [
                [:pipe-fn]
                [:metric-fn]]))

          temp-file (str output-dir "/" ( java.util.UUID/randomUUID) ".nippy")
          _ (swap! files #(conj % temp-file))]
      (nippy/freeze-to-file temp-file freezable-result)
      (result-reduce-fn result))))

(defn qualify-keywords
  "Converts unqualified keywords in a pipeline declaration to fully-qualified form.

  `pipe-decl` - Pipeline declaration (nested data structure)
  `pipe-ns` - Namespace symbol for keyword resolution

  Returns the pipeline declaration with all resolvable keywords converted to
  namespace-qualified keywords (e.g., `:fn-name` becomes `:my.ns/fn-name`).

  Used to make pipeline declarations portable across namespaces."
  [pipe-decl pipe-ns]
  (walk/postwalk (fn [form]
                           ;; (println form)
                           (if-let [resolved (resolve-keyword form pipe-ns)]
                             (do
                               (if (var? resolved)
                                 (let [v (var-get resolved)
                                       k (keyword
                                          (-> resolved meta :ns str)
                                          (-> resolved meta :name str))]
                                   k)

                                 resolved))


                             form))

                         pipe-decl))


(defn qualify-pipelines
  "Qualifies all keywords in a sequence of pipeline declarations.

  `pipe-decls` - Sequence of pipeline declarations
  `pipe-ns` - Namespace symbol for keyword resolution

  Returns a vector of qualified pipeline declarations. Applies `qualify-keywords`
  to each pipeline in the sequence.

  See also: `scicloj.metamorph.ml.evaluation-handler/qualify-keywords`"
  [pipe-decls pipe-ns]
  (mapv #(qualify-keywords % pipe-ns) pipe-decls))


(def default-result-dissoc-in-seq
  "Default sequence of key paths to remove from evaluation results.

  Removes large data objects (datasets, model internals) while preserving metrics,
  options, and essential model metadata. Removes:

  * Dataset objects from fit/train/test contexts
  * Target and feature datasets from model
  * Smile-specific model internals (serialized bytes, dataframes)

  Used by `default-result-dissoc-in-fn` to clean results for storage/analysis."
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




(def result-dissoc-in-seq--ctxs
  "Sequence of paths to remove all context objects from evaluation results.

  Removes fit, train, and test contexts entirely, keeping only metrics and metadata.
  More aggressive cleanup than `default-result-dissoc-in-seq`.

  Used by `result-dissoc-in-seq-ctx-fn`."
  [[:fit-ctx]
   [:train-transform :ctx]
   [:test-transform :ctx]])


(def result-dissoc-in-seq--all
  "Maximum cleanup: removes almost everything except core metrics and model type.

  Removes contexts, timing data, probability distributions, pipeline declarations,
  functions, and source information. Keeps only the essential metric values and
  model type.

  Most aggressive cleanup option. Used by `result-dissoc-in-seq--all-fn`."
  [[:metric-fn]
   [:fit-ctx]
   [:train-transform :ctx]
   [:train-transform :other-metrics]
   [:train-transform :timing]
   [:train-transform :probability-distribution]
   [:test-transform :ctx]
   [:test-transform :other-metrics]
   [:test-transform :timing]
   [:test-transform :probability-distribution]
   [:pipe-decl]
   [:pipe-fn]
   [:timing-fit]
   [:loss-or-accuracy]
   [:source-information]])



(defn result-dissoc-in-seq--all-fn
  "evaluation-handler-fn which removes all :ctx"
  [result]
  (reduce-result result result-dissoc-in-seq--all))

(defn default-result-dissoc-in-fn
  "default evaluation-handler-fn"
  [result]
  (reduce-result result default-result-dissoc-in-seq))

(defn result-dissoc-in-seq-ctx-fn
  "evaluation-handler-fn which removes all :ctx"
  [result]
  (reduce-result result result-dissoc-in-seq--ctxs))

(defn select-paths
  "Extracts specific nested paths from a map into a new map.

  `m` - Source map
  `paths` - Sequence of key path vectors (e.g., `[[:a :b] [:c :d]]`)

  Returns a new map containing only the specified paths with their values.
  Paths with nil values are omitted from the result.

  Example: `(select-paths {:a {:b 1} :c 2} [[:a :b]]) => {:a {:b 1}}`"
  [m paths]
  (reduce (fn [acc path]
            (let [v (get-in m path)]
              (if v
                (assoc-in acc path v)
                acc)))
          {}
          paths))  


(defn metrics-and-model-keep-fn
  "evaluation-handler-fn which keeps only train-metric, test-metric and 
   the fitted model map, which contains as well the model object as byte array
   (amon other things)"
  [result]
  (select-paths result
                [[:train-transform :metric]
                 [:test-transform :metric]
                 [:fit-ctx :model]]))



(defn metrics-and-options-keep-fn
  "evaluation-handler-fn which keeps only train-metric, test-metric and and the options"
  [result]
  (select-paths result
                [[:train-transform :metric]
                 [:test-transform :metric]
                 [:fit-ctx :model :options]]))


(defn metrics-keep-fn
  "evaluation-handler-fn which keeps only train-metric, test-metric"
  [result]
  (select-paths result
                [[:train-transform :metric]
                 [:test-transform :metric]]))


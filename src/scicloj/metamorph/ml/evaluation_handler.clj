(ns scicloj.metamorph.ml.evaluation-handler
  (:require
   [scicloj.metamorph.ml.tools :refer [dissoc-in pp-str multi-dissoc-in]]
   [clojure.tools.reader :as tr]
   [clojure.tools.reader.reader-types :as rts]
   [clojure.java.classpath]
   [clojure.repl :as repl]
   [taoensso.nippy :as nippy])
  (:import [java.io File]))

(defn file->topforms-with-metadata [path]
  (->> path
       slurp
       rts/source-logging-push-back-reader
       repeat
       (map #(tr/read % false :EOF))
       (take-while (partial not= :EOF))))

(defn resolve-keyword
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




(defn fns->code-list [pipeline-source-file]
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


(defn fns->code-map [pipeline-source-file]
  (let [fns-code-list (fns->code-list pipeline-source-file)]
    (zipmap
     (map :fn-name fns-code-list)
     fns-code-list)))

(defn fn-symbol->code [fn-symbol pipe-ns pipeline-source-file]
  (:form-str
   (get (fns->code-map pipeline-source-file)
        (:name
         (meta
          (resolve-keyword  fn-symbol pipe-ns))))))


(defn get-code [symbol pipe-ns pipeline-source-file]
  ;; (def symbol symbol)
  (let [source (repl/source-fn symbol)
        orig-source (some-> symbol
                         resolve
                         meta
                         :orig
                         clojure.repl/source-fn)]

    ;; resolve
    ;; #(.toSymbol %)

    {:code-source (if orig-source orig-source source)
     :code-local-source (fn-symbol->code symbol pipe-ns pipeline-source-file)}))

(defn get-classpath []
  (->>
   (clojure.java.classpath/classpath)
   (map #(.getPath %))))

(defn get-fn-sources [qualified-pipe-decl pipe-ns pipeline-source-file]
  (let [codes (atom {})]
    (clojure.walk/postwalk (fn [keyword]
                             (when (keyword? keyword)
                               (let [symbol (symbol keyword)]
                                 (swap! codes #(assoc % symbol
                                                      (get-code symbol pipe-ns pipeline-source-file)))))
                             keyword)
                           qualified-pipe-decl)
    @codes))

(defn get-source-information [qualified-pipe-decl pipe-ns pipeline-source-file]
  {:fn-sources (get-fn-sources qualified-pipe-decl pipe-ns pipeline-source-file)
   :classpath (get-classpath)})

(defn example-nippy-handler [files output-dir result-reduce-fn]
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

(defn qualify-keywords [pipe-decl pipe-ns]
  (clojure.walk/postwalk (fn [form]
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


(defn qualify-pipelines [pipe-decls pipe-ns]
  (mapv #(qualify-keywords % pipe-ns) pipe-decls))


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




(def result-dissoc-in-seq--ctxs
  [[:fit-ctx]
   [:train-transform :ctx]
   [:test-transform :ctx]])


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

(defn- reduce-result [r result-dissoc-in-seq]
  (reduce (fn [x y]
            (dissoc-in x y))
          r
          result-dissoc-in-seq))


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


(defn metrics-and-model-keep-fn
  "evaluation-handler-fn which keeps only train-metric, test-metric and and the fitted model"
  [result]
  {:train-transform {:metric (get-in result [:train-transform :metric])}
   :test-transform {:metric (get-in result [:test-transform :metric])}
   :fit-ctx {:model (get-in result [:fit-ctx :model])}})

(defn metrics-and-options-keep-fn
  "evaluation-handler-fn which keeps only train-metric, test-metric and and the options"
  [result]
  {:train-transform {:metric (get-in result [:train-transform :metric])}
   :test-transform {:metric (get-in result [:test-transform :metric])}
   :fit-ctx {:model {:options (get-in result [:fit-ctx :model :options])}}})

(comment
  (def x (nippy/thaw-from-file "/tmp/f29648a6-9a73-4cb4-a7df-27e868b5bccf.nippy")))

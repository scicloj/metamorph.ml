(ns scicloj.metamorph.persistence-tools
  (:require  [clojure.test :as t]
             [clojure.tools.reader :as tr]
             [clojure.tools.reader.reader-types :as rts]
             [clojure.java.classpath]
             [clojure.repl]))
   

(defn keys-in
    "Returns a sequence of all key paths in a given map using DFS walk."
    [m]
    (letfn [(children [node]
              (let [v (get-in m node)]
                (if (map? v)
                  (map (fn [x] (conj node x)) (keys v))
                  [])))
            (branch? [node] (-> (children node) seq boolean))]
      (->> (keys m)
           (map vector)
           (mapcat #(tree-seq branch? children %)))))


(defn find-model-data [m]
  (->>
   (keys-in m)
   (filter #(= :model-data (last %)))))


(defn pp-str [x]
  (with-out-str (clojure.pprint/pprint x)))


(defn file->topforms-with-metadata [path]
  (->> path
       slurp
       rts/source-logging-push-back-reader
       repeat
       (map #(tr/read % false :EOF))
       (take-while (partial not= :EOF))))

(defn fns->code-list []
  (->>
   (file->topforms-with-metadata "/home/carsten/Dropbox/sources/metamorph.ml/test/scicloj/metamorph/ml_test.clj")
   (map (fn [form]
          (hash-map :top-level (first form)
                    :form form
                    :form-str (pp-str form))))
   (filter #(= 'defn (:top-level %)))
   (map #(assoc % :fn-name (second (:form  %))))))


(defn fns->code-map []
  (let [fns-code-list (fns->code-list)]
    (zipmap
     (map :fn-name fns-code-list)
     fns-code-list)))

(defn resolve-keyword
  "Interpret keyword or list as a symbol and try to resolve it."
  [k pipe-ns]
  (def k k)
  (def pipe-ns pipe-ns)
  (if (or  (sequential? k) (map? k))
    k
    (-> (if-let [n (namespace k)] ;; namespaced?
          (let [sn (symbol n)
                n (str (get (ns-aliases pipe-ns) sn sn))] ;; try to find namespace in aliases
            (symbol n (name k))) ;; create proper symbol with fixed namespace
          (symbol (str pipe-ns) (name k))) ;; no namespace case
        (resolve))))

(defn fn-symbol->code [fn-symbol pipe-ns]
  (:form-str
   (get (fns->code-map)
        (:name
         (meta
          (resolve-keyword  fn-symbol pipe-ns))))))


(defn qualify-keywords [pipe-decl pipe-ns]
  (clojure.walk/postwalk (fn [form]
                           ;; (println form)
                           (if-let [resolved (resolve-keyword form pipe-ns)]
                             (do (def resolved resolved)
                                 (if (var? resolved)
                                   (let [v (var-get resolved)
                                         k (keyword
                                            (-> resolved meta :ns str)
                                            (-> resolved meta :name str))]
                                     k)

                                   resolved))

                                 
                             form))

                         pipe-decl))

(defn get-code [symbol]
  {:code-source (clojure.repl/source-fn symbol)
   :code-local-source (fn-symbol->code symbol *ns*)})




;; (fn-symbol->code (symbol :scicloj.metamorph.ml-test/do-xxx) *ns*)

(defn get-fn-sources [qualified-pipe-decl]
  (let [codes (atom {})]
    (clojure.walk/postwalk (fn [keyword]
                             (when (keyword? keyword)
                               (let [symbol (symbol keyword)]
                                 (swap! codes #(assoc % symbol
                                                      (get-code symbol)))))
                             keyword)
                           qualified-pipe-decl)
    @codes))

(defn get-classpath []
  (->>
     (clojure.java.classpath/classpath)
     (map #(.getPath %))))


(defn get-source-information [qualified-pipe-decl]
  {:fn-sources (get-fn-sources qualified-pipe-decl)
   :classpath (get-classpath)})
  



(defn qualify-pipelines [pipe-decls pipe-ns]
  (mapv #(qualify-keywords % pipe-ns) pipe-decls))

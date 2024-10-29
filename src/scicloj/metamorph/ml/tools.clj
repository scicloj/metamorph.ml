(ns scicloj.metamorph.ml.tools
  (:require
   [clojure.pprint :as pprint]) 
  (:import
   [it.unimi.dsi.fastutil.objects Object2LongLinkedOpenHashMap]))

(set! *warn-on-reflection* true)

(defn- maybe-dissoc [x k]
  (if (associative? x)
    (dissoc x k)
    x))

(defn- maybe-empty? [x]
  (and (sequential? x) (empty? x)))

(defn dissoc-in
  "Dissociate a value in a nested assocative structure, identified by a sequence
  of keys. Any collections left empty by the operation will be dissociated from
  their containing structures."
  [m ks]
  (if-let [[k & ks] (seq ks)]
    (if (seq ks)
      (let [v (dissoc-in (get m k) ks)]
        (if (maybe-empty? v)
          (maybe-dissoc m k)
          (assoc m k v)))
      (maybe-dissoc m k))
    m))


(defn multi-dissoc-in [m kss]
  (reduce (fn [x y]
            (dissoc-in x y))
          m
          kss))


(defn pp-str [x]
  (with-out-str (pprint/pprint x)))



(def time-format  (java.text.SimpleDateFormat. "HH:mm:ss.SSSS"))
(def prevoius-debug-time (atom (java.time.LocalTime/now)))
(defn debug [& s]
  (let [duration
        (.toSeconds
         (java.time.Duration/between
          @prevoius-debug-time
          (java.time.LocalTime/now)))]

    (reset! prevoius-debug-time (java.time.LocalTime/now))
    (println (format "  (%s) " duration))
    (apply print (.format ^java.text.SimpleDateFormat time-format
                          (java.util.Date.)) " - " s)))



(defn process-file [lines-source 
                    lines-seq-fn
                    line-func
                    line-acc
                     max-lines skip-lines]
  (if (instance? java.io.Reader lines-source)
    (with-open [rdr ^java.io.Reader lines-source]
      (->> rdr
           lines-seq-fn
           (drop skip-lines)
           (take max-lines)
           (reduce line-func line-acc)))
    
    (->> lines-source
         lines-seq-fn
         (drop skip-lines)
         (take max-lines)
         (reduce line-func line-acc)))
  )

(defn put-retrieve-token! [^Object2LongLinkedOpenHashMap token-lookup-table ^String token]
  (if (.containsKey token-lookup-table token)
    (.get token-lookup-table token)
    (let [next-token (.size token-lookup-table)]
      (.put token-lookup-table ^String
            token next-token)
      next-token)))

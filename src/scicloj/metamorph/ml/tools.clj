(ns scicloj.metamorph.ml.tools
  (:require
   [clojure.pprint :as pprint]
   ) 
  (:import
   [java.util Map]))

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

(def token-of-unknown "[UNKNOWN]")
(def token-idx-for-unknown (int 0))


(defn- put-and-rethrow [^Map token-lookup-table
                       ^String token
                       next-token-id
                       ]
  (try
    (.put token-lookup-table token (int next-token-id))
    next-token-id
    (catch UnsupportedOperationException e
      (throw (Exception. "token->int-map is immutable. You can set :new-token-behaviour to :as-unknown to 
                          use the special token [UNKNOWN] for an new token or to :store and provide a mutable 
                          map in :token->int-map" e)))))

(defn- save-put [^Map token-lookup-table
                 ^String token
                 next-token-id
                 new-token-behaviour]
  
  (case new-token-behaviour
   :store (put-and-rethrow token-lookup-table token next-token-id)
   :fail (throw ( Exception. (str "token not in token->int-map: " token "This exception can be avoided by setting `:new-token-behaviour` to :as-unknown or :store" )))
   :as-unknown (int token-idx-for-unknown) 

  )
  )

(defn get-put-token [^Map token-lookup-table new-token-behaviour ^String token ]
  ;;(println :table-size (.size token-lookup-table) :get-put-token token)
  (let [v (.get token-lookup-table token)]
    (if (some? v)
      v
      (let [next-token-id (.size token-lookup-table)]

        (save-put token-lookup-table ^String token next-token-id
                  new-token-behaviour)))
  ))

(comment
  (def m (java.util.HashMap.))
  
  (def u (java.util.Collections/unmodifiableMap m))
  
  (save-put u "meme" 3 :assign-unknn)
  )


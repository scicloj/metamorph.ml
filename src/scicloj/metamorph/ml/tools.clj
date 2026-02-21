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


(defn multi-dissoc-in
  "Removes multiple nested key paths from a map.

  `m` - Map to dissociate from
  `kss` - Sequence of key path vectors to remove

  Returns the map with all specified paths removed. Applies `dissoc-in` to each
  path in sequence. Empty collections left by removals are automatically dissociated.

  Example: `(multi-dissoc-in {:a {:b 1 :c 2}} [[:a :b] [:a :c]]) => {}`"
  [m kss]
  (reduce (fn [x y]
            (dissoc-in x y))
          m
          kss))


(defn pp-str
  "Pretty-prints a value to a string.

  `x` - Value to pretty-print

  Returns a string containing the pretty-printed representation using
  `clojure.pprint/pprint`. Useful for readable logging and debugging output."
  [x]
  (with-out-str (pprint/pprint x)))



(def time-format
  "SimpleDateFormat for debug output (HH:mm:ss.SSSS format)."
  (java.text.SimpleDateFormat. "HH:mm:ss.SSSS"))

(def prevoius-debug-time
  "Atom tracking the last debug timestamp for duration calculations."
  (atom (java.time.LocalTime/now)))

(defn debug
  "Prints formatted debug messages with timestamp and elapsed time.

  `s` - Variable number of arguments to print

  Prints current time, elapsed seconds since last debug call, and the message.
  Resets the internal timer on each call.

  Example output: `  (5) HH:mm:ss.SSSS - Debug message here`"
  [& s]
  (let [duration
        (.toSeconds
         (java.time.Duration/between
          @prevoius-debug-time
          (java.time.LocalTime/now)))]

    (reset! prevoius-debug-time (java.time.LocalTime/now))
    (println (format "  (%s) " duration))
    (apply print (.format ^java.text.SimpleDateFormat time-format
                          (java.util.Date.)) " - " s)))



(defn process-file
  "Generic file line processor with skip/take support.

  `lines-source` - java.io.Reader or any sequence source
  `lines-seq-fn` - Function to convert source to line sequence
  `line-func` - Reducer function `(fn [acc line] ...)`
  `line-acc` - Initial accumulator value
  `max-lines` - Maximum number of lines to process
  `skip-lines` - Number of lines to skip at start

  Returns the final accumulator after processing. Automatically manages Reader
  lifecycle with `with-open` for Reader sources."
  [lines-source
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
         (reduce line-func line-acc))))

(def token-of-unknown
  "String constant representing unknown tokens in categorical encoding.

  Value: `\"[UNKNOWN]\"`. Used as a placeholder when encountering tokens not
  in the training vocabulary, depending on `:new-token-behaviour` setting."
  "[UNKNOWN]")

(def token-idx-for-unknown
  "Integer index assigned to unknown tokens in categorical encoding.

  Value: `0`. Used when `:new-token-behaviour` is `:as-unknown` to map all
  out-of-vocabulary tokens to this index."
  (int 0))


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

(defn get-put-token
  "Gets token ID from lookup table or assigns new ID based on strategy.

  `token-lookup-table` - Mutable Map from tokens (String) to IDs (int)
  `new-token-behaviour` - Strategy for unknown tokens: `:store`, `:fail`, or `:as-unknown`
  `token` - String token to look up or add

  Returns the token's integer ID. Behavior for new tokens:

  * `:store` - Adds token to table with next ID (requires mutable map)
  * `:fail` - Throws exception
  * `:as-unknown` - Returns `token-idx-for-unknown` (0)

  Used internally for categorical and text encoding."
  [^Map token-lookup-table new-token-behaviour ^String token]
  ;;(println :table-size (.size token-lookup-table) :get-put-token token)
  (let [v (.get token-lookup-table token)]
    (if (some? v)
      v
      (let [next-token-id (.size token-lookup-table)]

        (save-put token-lookup-table ^String token next-token-id
                  new-token-behaviour)))))

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

(defn reduce-result
  "Removes multiple nested paths from a result map.

  `r` - Result map
  `result-dissoc-in-seq` - Sequence of key path vectors to remove

  Returns the result map with all specified paths removed using `dissoc-in`.
  Alias for `multi-dissoc-in` used in evaluation result handling.

  See also: `multi-dissoc-in`, `scicloj.metamorph.ml.evaluation-handler/default-result-dissoc-in-seq`"
  [r result-dissoc-in-seq]
  (reduce (fn [x y]
            (dissoc-in x y))
          r
          result-dissoc-in-seq))



(comment
  (def m (java.util.HashMap.))
  
  (def u (java.util.Collections/unmodifiableMap m))
  
  (save-put u "meme" 3 :assign-unknn)
  )


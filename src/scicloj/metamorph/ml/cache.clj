(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [taoensso.carmine :as car]
            [potemkin]))

(defn ->cache-file [dir k]
  (io/file (format "%s/%s.nippy" dir k)))

(defn- stream-bytes [is]
  (let [baos (java.io.ByteArrayOutputStream.)]
    (io/copy is baos)
    (.toByteArray baos)))

;; A map which "freezes" its values via nippy, and delegates to a set of functions to persist this
;; however needed
(potemkin/def-map-type NippyMap [write-bytes-fn read-bytes-fn delete-key-fn keys-fn]

  (get [_ k default-value]
       (let [bytes (read-bytes-fn k)]
         (if (nil? bytes)
           default-value
           (if (not (bytes? bytes))   ;; carmine auto-thaws
               bytes                  ;
               (nippy/thaw bytes {:serializable-allowlist #{"*"}})))))

  (assoc [_ k v]
         (write-bytes-fn k (nippy/freeze v))
         (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (dissoc [_ k]
          (delete-key-fn k)
          (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (keys [_]
        (keys-fn))


  (meta [_]
        (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (with-meta [_ mta]
    (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn)))





(defn string->bytes-map-factory
  "Creates a map (String -> bytes) implementation. Can be used as base for
  clojure.core.cachw.wrapped caches to be passed to function `ml/model`

  It takes 4 method to control what persistence means:
  write-bytes-fn - [key bytes]  -> nil
  read-bytes-fn  - [key]        -> byte array
  delete-key-fn  - [key]        -> nil
  keys-fn        - []           -> seq of Strings

  The `key` passed is URL and filename safe. It's a hash to the inputs of train / predict
 The `bytes` is a byte array, and represnts the nippied train/predict result
  "

  [write-bytes-fn read-bytes-fn delete-key-fn keys-fn]
  (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))


(defn- scan-all [wcar-opts]
  (loop [cursor "0"
         keys []]

    (let [res (car/wcar wcar-opts (car/scan cursor))
          new-keys (concat (second res) keys)]
      (if (= "0"  (first res))
        new-keys
        (recur (first res) new-keys)))))


(defn redis-persisted-map-factory
  "Creates a map (String -> bytes)  which is persisted in redis via Carmine.
  `wcar-opts` are teh Carmine connection options to redis
  See `presisted-map-factory`"
  [wcar-opts]

 
  (string->bytes-map-factory

   (fn [k bytes]
     (car/wcar wcar-opts (car/set k bytes)))

   (fn [k]
     (car/wcar wcar-opts (car/get k)))

   (fn [k]
     (car/wcar wcar-opts (car/del k)))

   (fn [] (scan-all wcar-opts))))



(defn fs-persisted-map-factory
  "Creates a map (String -> bytes)  which is persisted in local file system.
  `dir` is the loation of the store.
  See `persisted-map-factory`
  "
  [dir]

  (string->bytes-map-factory

   (fn [k bytes]
     (println :write k)
     (io/copy bytes (->cache-file dir k)))

   (fn [k]
     (let [file (->cache-file dir k)]
       (when (.exists file)
         (stream-bytes (io/input-stream file)))))

   (fn [k] (.delete (->cache-file dir k)))

   (fn []
     (->> (io/file dir)
          (file-seq)
          (filter #(.isFile %))
          (map #(-> (.getName %) (str/replace ".nippy" "")))))))

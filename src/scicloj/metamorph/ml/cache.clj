(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [potemkin]
            [konserve.core :as k]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml :as ml]))


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
       (println :get :k k)

       (let [bytes (read-bytes-fn k)]
         (if (nil? bytes)
           default-value
           (nippy/thaw bytes))))

  (assoc [_ k v]
         (println :assoc :k k)
         (write-bytes-fn k (nippy/freeze v))
         (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (dissoc [_ k]
          (println :dissoc)
          (delete-key-fn k)
          (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (keys [_]
        (println :keys)
        (keys-fn))


  (meta [_]
        (println :meta)
        (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (with-meta [_ mta]
    (println :with-meta)
    (NippyMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn)))





(defn persisted-map-factory
  "Creates a map (String -> bytes) implementation which get persisted. Can be used as base for
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




(defn fs-persisted-map-factory
  "Creates a map (String -> bytes)  which is persisted in local file system.
  `dir` is the loation of the store.
  See `presisted-map-factory`
  "

  [dir]

  (persisted-map-factory

   (fn [k bytes]
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

(defn- get-binary [store k]

  (k/bget store k
          (fn [{is :input-stream}]
            (nippy/thaw (stream-bytes is)
                        {:serializable-allowlist #{"*"}}))
              ;; "java.util.Properties"
              ;; "smile.data.DataFrameImpl"
              ;; "smile.data.formula.Formula"
          {:sync? true}))

(defn caching-train [store dataset options]

  (let [hash-ds (str (hash dataset))
        k {:op :train
           :options options
           :hash-ds hash-ds}]
    (if (k/exists? store k {:sync? true})
      (get-binary store k)
      (let [train-result (ml/train dataset options)
            wrapped {:train-result-wrapper train-result
                     :hash-train-inputs (str (hash k))}]
        (k/bassoc store k (nippy/freeze wrapped)
                  {:sync? true})
        wrapped))))
        
        
(defn- dissoc-in
  [m [k & ks]]
  (if-not ks
    (dissoc m k)
    (assoc m k (dissoc-in (m k) ks))))


(defn caching-predict [store dataset wrapped-model]
  (let [k {:op :predict
           :hash-train-inputs (:hash-train-inputs wrapped-model)
           :hash-ds (str (hash dataset))}
        model (:train-result-wrapper wrapped-model)]
    (if (k/exists? store k {:sync? true})
      (get-binary store k)
      (let [predict-result (ml/predict dataset model)]
        (k/bassoc store k (nippy/freeze predict-result) {:sync? true})

        predict-result))))

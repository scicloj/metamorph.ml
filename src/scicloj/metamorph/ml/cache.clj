(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]
            [konserve.core :as k]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml :as ml]
            [buddy.core.hash :as hash]
            [buddy.core.codecs :as codecs]))


(defn- stream-bytes [is]
  (let [baos (java.io.ByteArrayOutputStream.)]
    (io/copy is baos)
    (.toByteArray baos)))

(defn- get-binary [store k]
  (k/bget store k
          (fn [{is :input-stream}]
            (nippy/thaw (stream-bytes is)
                        {:serializable-allowlist #{"*"}}))
              ;; "java.util.Properties"
              ;; "smile.data.DataFrameImpl"
              ;; "smile.data.formula.Formula"
          {:sync? true}))

(defn caching-train-konserve
  "A variant of ml/train which caches training results.
   It constructs a `hash` out of:
      * hash of dataset
      * hash of options

   and checks if present in 'store'. I fpresent it retunrs teh result,
  otherwise it class ml/train and stores resukt in store under the key `hash`
  "
  [store dataset options]
  (let [hash-ds (str (hash dataset))
        k {:op :train
           :options options
           :hash-ds hash-ds}]
    (if (k/exists? store k {:sync? true})
      (get-binary store k)
      (let [train-result (ml/train dataset options)
            wrapped {:model-wrapper train-result
                     :hash-train-inputs (str (hash k))}]
        (k/bassoc store k (nippy/freeze wrapped)
                  {:sync? true})
        wrapped))))
        
        
(defn caching-predict-konserve [store dataset wrapped-model]
  (let [k {:op :predict
           :hash-train-inputs (:hash-train-inputs wrapped-model)
           :hash-ds (str (hash dataset))}
        model (:model-wrapper wrapped-model)]
    (if (k/exists? store k {:sync? true})
      (get-binary store k)
      (let [predict-result (ml/predict dataset model)]
        (k/bassoc store k (nippy/freeze predict-result) {:sync? true})

        predict-result))))


(defn k->sha256 [k]
  (-> k
      str
      (hash/sha256)
      (codecs/bytes->hex)))



(defn caching-train-nippy
  "A variant of ml/train which caches training results.
   It constructs a `hash` out of:
      * hash of dataset
      * hash of options

   and checks if present in 'store'. I fpresent it retunrs teh result,
  otherwise it class ml/train and stores resukt in store under the key `hash`
  "
  [dir dataset options]
  (let [hash-ds (str (hash dataset))
        k {:op :train
           :options options
           :hash-ds hash-ds}
        cache-file (io/file (format "%s/%s.nippy" dir (k->sha256  k)))]
    (if (.exists cache-file)
      (nippy/thaw-from-file cache-file {:serializable-allowlist #{"*"}})
      (let [train-result (ml/train dataset options)
            wrapped {:model-wrapper train-result
                     :hash-train-inputs (str (hash k))}]
        (io/copy (nippy/freeze wrapped) cache-file)
        wrapped))))


(defn caching-predict-nippy [dir dataset wrapped-model]
  (let [k {:op :predict
           :hash-train-inputs (:hash-train-inputs wrapped-model)
           :hash-ds (str (hash dataset))}
        model (:model-wrapper wrapped-model)
        cache-file (io/file (format "%s/%s.nippy" dir  (k->sha256 k)))]
    (if (.exists cache-file)
      (nippy/thaw-from-file cache-file {:serializable-allowlist #{"*"}})
      (let [predict-result (ml/predict dataset model)]
        (nippy/freeze-to-file cache-file predict-result)
       predict-result))))

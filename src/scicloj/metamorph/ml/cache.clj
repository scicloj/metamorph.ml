(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]

            [konserve.core :as k]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml :as ml]))


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
           :hash-train-input (:hash-train-inputs wrapped-model)

           :hash-ds (str (hash dataset))}
        model (:train-result-wrapper wrapped-model)]
    (if (k/exists? store k {:sync? true})
      (get-binary store k)
      (let [predict-result (ml/predict dataset model)]
        (k/bassoc store k (nippy/freeze predict-result) {:sync? true})

        predict-result))))

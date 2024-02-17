(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]
            [konserve.core :as k]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml :as ml]
            [buddy.core.hash :as hash]
            [clojure.core.cache :as cache]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [clojure.core.cache.wrapped :as wcache]
            [tech.v3.dataset.modelling :as ds-mod]
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


(def dir "/tmp/store")

(defn item->key [item]

  (if (= :train (:op item))
    (k->sha256
     {:op :train
      :options (:options item)
      :hash-ds (hash (:dataset item))})
    (k->sha256
     {:op :predict
      :hash-train-inputs (:hash-train-inputs (:model-wrapper item))
      :hash-ds (hash (:dataset item))})))



(defn item->file [dir item]
  (io/file (format "%s/%s.nippy" dir (item->key item))))




(cache/defcache MyCache [cache]
  cache/CacheProtocol
  (lookup [cache e]

          (when (cache/has? cache e)
            (nippy/thaw-from-file (item->file dir e))))

  (lookup [cache e not-found]
          (if (cache/has? cache e)
            (nippy/thaw-from-file (item->file dir e))
            not-found))


  (has? [_ e]
        ;; (println :has? e)
        (.exists (item->file dir e)))
  (hit [cache e]
       (nippy/thaw-from-file (item->file dir e)))


  (miss [cache e ret]
        (def e e)
        (def ret ret)
        (if (= :train (:op e))
          (let [train-result (ml/train (:dataset e)
                                       (:options e))
                wrapped {:model-wrapper train-result
                         :hash-train-inputs (item->key e)}]
            (nippy/freeze-to-file (item->file dir e) wrapped)
            wrapped)
          (let [model (:model-wrapper e)
                predict-result (ml/predict
                                (:dataset e)
                                model)]
            (nippy/freeze-to-file (item->file dir e) predict-result)
           predict-result))))


        
        



(def iris
  (->
   (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})
   (ds/categorical->number [:species])
   (ds-mod/set-inference-target :species)))


(def c (MyCache. {}))



(defn caching-train-ccc
  [dataset options]
  (cache/through c {:op :train
                    :dataset dataset
                    :options options}))



(defn caching-predict-ccc [dataset wrapped-model]
  (cache/through c (merge  {:op :predict
                            :dataset dataset}
                           wrapped-model)))
                            



(comment

  {:op nil
   :dataset nil
   :options nil
   :wrapped-model nil}


  (cache/lookup c :e "hello")
  (cache/through c {:op :train
                    :dataset iris
                    :options {:model-type :slow-model}})


  (cache/lookup c {:op :train
                   :dataset iris
                   :options {:model-type :slow-model}}))

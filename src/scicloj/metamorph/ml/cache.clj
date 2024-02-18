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
            [buddy.core.codecs :as codecs]
            [clojure.string :as str]
            [potemkin]))




(defn- k->sha256 [k]
  (-> k
      str
      (hash/sha256)
      (codecs/bytes->hex)))


(defn- stream-bytes [is]
    (let [baos (java.io.ByteArrayOutputStream.)]
      (io/copy is baos)
      (.toByteArray baos)))



(defn caching-train
  [wcache dataset options]
  (let [k (k->sha256
           {:op :train
            :dataset dataset
            :options options})]
    (wcache/lookup-or-miss
     wcache
     k
     (fn [item]

       (let [train-result (ml/train
                           dataset
                           options)

             wrapped {:model-wrapper train-result
                      :hash-train-inputs (str (hash k))}
             bytes (nippy/freeze wrapped)]
         (println :bytes bytes)
         wrapped)))))


(defn caching-predict [wcache dataset wrapped-model]
  (let [k
        (k->sha256

         {:op :predict
          :hash-train-inputs (:hash-train-inputs wrapped-model)
          :hash-ds (str (hash dataset))})
        model (:model-wrapper wrapped-model)]


    (wcache/lookup-or-miss
     wcache
     k
     (fn [item]

       (let [predict-result
             (ml/predict dataset model)
             bytes (nippy/freeze predict-result)]
         (println :bytes bytes)
         predict-result)))))




(defn- item->key [item]


  (if (= :train (:op item))
    (k->sha256
     {:op :train
      :options (:options item)
      :hash-ds (hash (:dataset item))})
    (k->sha256
     {:op :predict
      :hash-train-inputs (:hash-train-inputs (:model-wrapper item))
      :hash-ds (hash (:dataset item))})))



(defn- item->file [dir item]
  (io/file (format "%s/%s.nippy" dir (item->key item))))



(potemkin/def-map-type DiskMap [write-bytes-fn read-bytes-fn delete-key-fn keys-fn]

  (get [_ k default-value]
       (println :get :k k)

       (let [bytes (read-bytes-fn k)]
         (if (nil? bytes)
           default-value
           (nippy/thaw bytes))))


  (assoc [_ k v]
         (println :assoc :k k)
         (write-bytes-fn k (nippy/freeze v))
         (DiskMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (dissoc [_ k]
          (println :dissoc)
          (delete-key-fn k)

          (DiskMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))


  (keys [_]
        (println :keys)
        (keys-fn))
        


  (meta [_]
        (println :meta)
        (DiskMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn))

  (with-meta [_ mta]
    (println :with-meta)
    (DiskMap. write-bytes-fn read-bytes-fn delete-key-fn keys-fn)))




(defn fs-persisted-map-factory [dir]
  (DiskMap. (fn [k bytes]
              (io/copy
               bytes
               (io/file (format "%s/%s.nippy" dir k))))
            (fn [k]
              (let [file (io/file (format "%s/%s.nippy"
                                          dir k))]
                (when (.exists file)

                  (stream-bytes (io/input-stream file)))))
            (fn [k] (.delete (io/file (format "%s/%s.nippy" dir k))))
            (fn [] (->>

                    (file-seq (io/file dir))
                    (filter #(.isFile %))
                    (map #(-> (.getName %) (str/replace ".nippy" "")))))))




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



(comment
  (defn caching-train-ccc
    [cache dataset options]
    (println :train)
    (cache/through cache {:op :train
                          :dataset dataset
                          :options options}))



  (defn caching-predict-ccc [cache dataset wrapped-model]
    (println :predict)
    (def wrapped-model wrapped-model)
    (cache/through cache (merge  {:op :predict
                                  :dataset dataset}
                                 wrapped-model)))



  (defn bytes-storage-cache-factory
    ([store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn]
     (BytesStorageCache. {} store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn))


    ([dir] (bytes-storage-cache-factory (fn [key bytes]
                                          (println :store-bytes :key key)
                                          (io/copy bytes (io/file (format "%s/%s.nippy" dir key))))
                                        (fn [key]
                                          (println :retrieve-bytes :key key)
                                          (stream-bytes (io/input-stream (format "%s/%s.nippy" dir key))))
                                        (fn [key]
                                          ;; (def dir dir)
                                          ;; (def key key)
                                          (println :exists :key key)
                                          (.exists (io/file (format "%s/%s.nippy" dir key))))
                                        (fn [key]
                                          (println :delete-bytes :key key)
                                          (.delete (io/file (format "%s/%s.nippy" dir key)))))))

  (cache/defcache BytesStorageCache [cache store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn]
    cache/CacheProtocol
    (lookup [_ e
             (println :lookup-1

                      (when (exists-bytes-fn (item->key e))
                       (retrieve-bytes-fn (item->key e))))])
    ;; (nippy/thaw-from-file (item->file dir e))


    (lookup [_ e not-found
             (println :lookup-2
                      (if (exists-bytes-fn (item->key e))
                       (retrieve-bytes-fn (item->key e)
                        not-found)))])


    (has? [_ e
           (println :has?
                    (exists-bytes-fn (item->key e)))])

    (hit [_ e
          (println :hit
                   (-> e
                      item->key
                      retrieve-bytes-fn
                      (nippy/thaw {:serializable-allowlist #{"*"}}))
                   cache)])

    ;; (nippy/thaw-from-file (item->file dir e))



    (miss [_ e ret
           (println :miss
                    ;; (def e e)
                    ;; (def ret ret)
                    ;; (println :e e)
                    ;; (println :ret ret)
                    (let [k (item->key e)]
                     val
                     (if (= :train (:op e))
                      (let [train-result (ml/train (:dataset e)
                                                   (:options e))])
                      wrapped {:model-wrapper train-result}
                      :hash-train-inputs (item->key e)


                      (store-bytes-fn  k (nippy/freeze wrapped)
                               wrapped

                               ;; (BytesStorageCache. cache store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn)

                               ;; (nippy/freeze-to-file (item->file dir e) wrapped)


                               (let [model (:model-wrapper e)]


                                    predict-result (ml/predict)
                                       (:dataset e)
                                       model


                                       (store-bytes-fn k (nippy/freeze predict-result))
                                       predict-result
                                       (BytesStorageCache. (assoc cache k nil) store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn))))))])



    ;;





    (evict [_ key
            (println :evict
                     (delete-bytes-fn key)
                     (BytesStorageCache. cache store-bytes-fn retrieve-bytes-fn exists-bytes-fn delete-bytes-fn))]))




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

 (defn caching-train-nippy
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

 :ok)

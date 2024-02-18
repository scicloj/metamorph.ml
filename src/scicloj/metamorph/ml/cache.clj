(ns scicloj.metamorph.ml.cache
  (:require [taoensso.nippy :as nippy]

            [clojure.java.io :as io]
            [scicloj.metamorph.ml :as ml]
            [buddy.core.hash :as hash]



            [clojure.core.cache.wrapped :as wcache]

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

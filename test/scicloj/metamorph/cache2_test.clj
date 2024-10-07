(ns scicloj.metamorph.cache2-test

  (:require  [clojure.test :as t]
             [scicloj.metamorph.ml :as ml]
             [scicloj.metamorph.ml.toydata :as data]
             [tech.v3.dataset :as ds]
             [taoensso.nippy :as nippy]
             [clojure.core.async :refer [<!!]]
             [konserve.core :as k] 
             [konserve.protocols :refer [PStoreSerializer]]
             [scicloj.ml.smile.classification]
             [clojure.java.io :as io]
             [konserve.filestore :refer [connect-fs-store]]
             [konserve.protocols :as kp]
             [ bites.core :as bites]
             [clj-commons.byte-streams :as bs])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))



(defn slurp-bytes
  "Slurp the bytes from a slurpable thing"
  [x]
  (with-open [out (java.io.ByteArrayOutputStream.)]
    (clojure.java.io/copy (clojure.java.io/input-stream x) out)
    (.toByteArray out)))

(def dataset (data/iris-ds))
(def options {:model-type :smile.classification/logistic-regression})



(defn hash-train-args-as-string [dataset options]
  ;; todo good enough ?
  (str
   (hash {:ds dataset
          :options options}))
  )


(defn hash-predict-args-as-string [dataset model]
  ;; todo good enough ?
  (str
   (hash {:ds dataset
          :model (dissoc model :id )})))


(def store (connect-fs-store "/tmp/store" 
                             
                             ;:default-serializer :NippySerializer
                             ;:serializers {:NippySerializer (->NippySerializer)} 
                             
                             :opts {:sync? true
                                    }))
(def trained-model
  (ml/train dataset options))

(clojure.reflect/reflect smile-model-0)
(.w smile-model-0)

(def trained-model-bytes
  (nippy/freeze trained-model))

(def smile-model-0 (ml/thaw-model trained-model))
(def smile-model-1 (ml/thaw-model trained-model))

(= smile-model-0 smile-model-1)

(hash
 (bites/to-bytes smile-model-1))

(def train-arg-hash (hash-train-args-as-string dataset options))
train-arg-hash
;;=> "-661618312"

(k/dissoc store train-arg-hash {:sync? true})
;; binay
(k/bassoc store train-arg-hash trained-model-bytes {:sync? true})
(k/exists? store train-arg-hash {:sync? true})

(def m (k/bget store train-arg-hash
               (fn locked-cb [{is :input-stream}]
                 (println :is is)
                 (nippy/thaw (slurp-bytes is)))
               {:sync? true}))

(def predict-args-hash (hash-predict-args-as-string 
                        dataset 
                        trained-model))

predict-args-hash
;;=> "-1883638799"

(ml/predict dataset m)



(comment

  (defrecord NippySerializer []
    PStoreSerializer
    (-deserialize [this read-handlers input-stream]
      (when-not (empty? @read-handlers)
        (throw (ex-info "Read handlers not supported yet." {:type :handlers-not-supported-yet})))
      (nippy/thaw input-stream))
    (-serialize [this output-stream write-handlers val]
      (println :serialized-called)
      (when-not (empty? @write-handlers)
        (throw (ex-info "Write handlers not supported yet." {:type :handlers-not-supported-yet})))
             ;(let [out  ( java.io.ByteArrayOutputStream.)])
      (io/copy (nippy/freeze val) output-stream)
      (.close output-stream)


      (def n (->NippySerializer))))
  
  (k/dissoc store train-arg-hash)
  (k/exists? store train-arg-hash {:sync? true})
  
  (k/assoc store train-arg-hash trained-model {:sync? true})

  
  )  

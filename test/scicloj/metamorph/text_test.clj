(ns scicloj.metamorph.text-test
  (:require [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [scicloj.metamorph.ml.csr :as csr]
            [scicloj.metamorph.ml.text :as text]
            [clojure.data.csv :as csv]
            [scicloj.ml.xgboost :as xgboost]
            [scicloj.metamorph.ml.loss :as loss]
  (:import [ml.dmlc.xgboost4j.java XGBoost]
           [ml.dmlc.xgboost4j.java DMatrix DMatrix$SparseType]))



(def result
  (text/->tidy-text "test/data/reviews.csv"
               (fn [line] 
                 (let [splitted (first
                                 (csv/read-csv line))]
                   (vector 
                    (first splitted)
                    (dec (Integer/parseInt (second splitted)))
                           ))
                 ) 
               :max-lines 10000
               :skip-lines 1     
                    
               ))

(def ds (:ds result))
;(def st (:st result))



(def rnd-indexes
  (-> (range 1000) shuffle))

(def rnd-indexes-train
  (take 800 rnd-indexes))

(def rnd-indexes-test
  (take-last 200 rnd-indexes))

(def ds-train
  (tc/left-join (tc/dataset {:document rnd-indexes-train}) ds [:document]))

(def ds-test
  (tc/left-join (tc/dataset {:document rnd-indexes-test}) ds [:document]))


(def bow-train
  (-> ds-train
      text/->term-frequency
      text/add-word-idx))

(def zero-baseddocs-map-train
  (zipmap
   (-> bow-train :document distinct)
   (range)))

(def bow-train-zeroed
  (-> bow-train
      (tc/add-or-replace-column
       :document
       #(map zero-baseddocs-map-train (:document %)))))


(def bow-test
  (-> ds-test
      text/->term-frequency
      text/add-word-idx))

(def zero-baseddocs-map-test
  (zipmap
   (-> bow-test :document distinct)
   (range)))

(def bow-test-zeroed
  (-> bow-test
      (tc/add-or-replace-column
       :document
       #(map zero-baseddocs-map-test (:document %)))))



(def sparse-features-train
  (-> bow-train-zeroed
      (tc/select-columns [:document :word-idx :tf])
      (tc/rows)))


(def sparse-features-test
  (-> bow-test-zeroed
      (tc/select-columns [:document :word-idx :tf])
      (tc/rows)))

;(def n-rows (inc (apply tcc/max (bow :document))))
(def n-col-train (inc (apply max  (bow-train-zeroed :word-idx))))
(def n-col-test (inc (apply max  (bow-test-zeroed :word-idx))))

(def csr-train
  (csr/->csr sparse-features-train))

(def csr-test
  (csr/->csr sparse-features-test))



(def labels-train
  (->
   bow-train-zeroed
   (tc/group-by :document)
   (tc/aggregate #(-> % :label first))
   (tc/column "summary")))

(def labels-test
  
  (->
   bow-test-zeroed
   (tc/group-by :document)
   (tc/aggregate #(-> % :label first))
   (tc/column "summary")))


(def m-train
  (DMatrix.
   (long-array (:row-pointers csr-train))
   (int-array (:column-indices csr-train))
   (float-array (:values csr-train))
   DMatrix$SparseType/CSR
   n-col-train))
(.setLabel m-train (float-array labels-train))

(def m-test
  (DMatrix.
   (long-array (:row-pointers csr-test))
   (int-array (:column-indices csr-test))
   (float-array (:values csr-test))
   DMatrix$SparseType/CSR
   n-col-test))
(.setLabel m-test (float-array labels-test))




(def model
  (xgboost/train-from-dmatrix
   m
   ["word"]
   ["label"]
   {:num-class 5}
   {}
   "multi:softmax"))



(def booster
  (XGBoost/loadModel
   (java.io.ByteArrayInputStream. (:model-data model))))

(def predition
  (.predict booster m-test))

(def predition
  (map #(int (first %)) predition))

(loss/classification-accuracy
 (vec predition)
 (vec labels-test))
;;=> 0.973

;;------------------------------


(def result
  (text/->tidy-text "test/data/small_text.csv"
                    (fn [line]
                      (let [splitted (first
                                      (csv/read-csv line))]
                        (vector
                         (first splitted)
                         (dec (Integer/parseInt (second splitted))))))
                    :max-lines 10000
                    :skip-lines 1))


(def ds (:ds result))
(def st (:st result))

ds
;;=> _unnamed [12 4]:
;;   
;;   | :word | :word-index | :document | :label |
;;   |-------|------------:|----------:|-------:|
;;   |     I |           0 |         0 |      0 |
;;   |  like |           1 |         0 |      0 |
;;   |  fish |           2 |         0 |      0 |
;;   |   and |           3 |         0 |      0 |
;;   |   you |           4 |         0 |      0 |
;;   |   the |           5 |         0 |      0 |
;;   |  fish |           6 |         0 |      0 |
;;   |    Do |           0 |         1 |      1 |
;;   |   you |           1 |         1 |      1 |
;;   |  like |           2 |         1 |      1 |
;;   |    me |           3 |         1 |      1 |
;;   |     ? |           4 |         1 |      1 |
;;   

(def bow
  (-> ds
      text/->term-frequency
      text/add-word-idx))

bow
;;=> _unnamed [11 5]:
;;   
;;   | :word | :document | :label | :tf | :word-idx |
;;   |-------|----------:|-------:|----:|----------:|
;;   |     I |         0 |      0 |   1 |         1 |
;;   |  like |         0 |      0 |   1 |         2 |
;;   |  fish |         0 |      0 |   2 |         3 |
;;   |   and |         0 |      0 |   1 |         4 |
;;   |   you |         0 |      0 |   1 |         5 |
;;   |   the |         0 |      0 |   1 |         6 |
;;   |    Do |         1 |      1 |   1 |         7 |
;;   |   you |         1 |      1 |   1 |         5 |
;;   |  like |         1 |      1 |   1 |         2 |
;;   |    me |         1 |      1 |   1 |         8 |
;;   |     ? |         1 |      1 |   1 |         9 |
;;   

st


(def sparse-features
  (-> bow
      (tc/select-columns [:document :word-idx :tf])
      (tc/rows)))
     ;;=> [[0 1 1] [0 2 1] [0 3 2] [0 4 1] [0 5 1] [0 6 1] [1 7 1] [1 5 1] [1 2 1] [1 8 1] [1 9 1]]


(def n-rows (inc (apply tcc/max (bow :document))))
n-rows
;;=> 2

(def n-col (inc (apply max  (bow :word-idx))))
n-col
;;=> 10

(def csr
  (csr/->csr sparse-features))

(def dense
  (csr/->dense csr n-rows n-col))
;;     0 1 2 3 4 5 6 7 8 9   
;;=> ((0 1 1 2 1 1 1 0 0 0)     ; I like fish fish and you the
;;    (0 0 1 0 0 1 0 1 1 1))    ; like you do me ? 

bow
;;=> _unnamed [11 5]:
;;   
;;   | :word | :document | :label | :tf | :word-idx |
;;   |-------|----------:|-------:|----:|----------:|
;;   |     I |         0 |      0 |   1 |         1 |
;;   |  like |         0 |      0 |   1 |         2 |
;;   |  fish |         0 |      0 |   2 |         3 |
;;   |   and |         0 |      0 |   1 |         4 |
;;   |   you |         0 |      0 |   1 |         5 |
;;   |   the |         0 |      0 |   1 |         6 |
;;   |    Do |         1 |      1 |   1 |         7 |
;;   |   you |         1 |      1 |   1 |         5 |
;;   |  like |         1 |      1 |   1 |         2 |
;;   |    me |         1 |      1 |   1 |         8 |
;;   |     ? |         1 |      1 |   1 |         9 |
;;   


(def labels
  (->
   bow
   (tc/group-by :document)
   (tc/aggregate #(-> % :label first))
   (tc/column "summary")))
labels
;;=> #tech.v3.dataset.column<int64>[2]
;;   summary
;;   [0, 1]



(def m
  (DMatrix.
   (long-array (:row-pointers csr))
   (int-array (:column-indices csr))
   (float-array (:values csr))
   DMatrix$SparseType/CSR
   n-col))
(.setLabel m (float-array labels))



(def model
  (xgboost/train-from-dmatrix
   m
   ["word"]
   ["label"]
   {:num-class 2}
   {}
   "multi:softprob"))


(def booster
  (XGBoost/loadModel
   (java.io.ByteArrayInputStream. (:model-data model))))
(def predition
  (.predict booster m))

predition



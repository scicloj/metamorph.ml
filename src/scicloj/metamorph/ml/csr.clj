(ns scicloj.metamorph.ml.csr
  (:require [fastmath.matrix :as m]
            ))

(defn- add-to-csr [csr row col value]
  (if (zero? value)
    csr
    (let [new-values (conj (:values csr) value)
          new-column-indices (conj (:column-indices csr) col)
          new-row-pointers (if (<= (count (:row-pointers csr)) row)

                             (conj (:row-pointers csr) (dec (count new-values)))
                             (:row-pointers csr))]
      {:values new-values
       :column-indices new-column-indices
       :row-pointers new-row-pointers})))

(defn ->csr [r-c-vs]
  (->
   (reduce

    (fn [csr [row col value]]
      (add-to-csr csr row col value))
    {:values []
     :column-indices []
     :row-pointers [0]}
    r-c-vs)

   (#(assoc % :row-pointers (conj (:row-pointers %)
                                  (count (:values %)))))))


(defn- first-non-nil-or-0 [s]
  (or
   (first (filter some? s))
   0))


(defn ->dense [csr rows cols]
  (for [i (range rows)]
    (let [row-start (nth (:row-pointers csr) i)
          row-end   (nth (:row-pointers csr) (inc i))]
      (for [j (range cols)]
        (first-non-nil-or-0
         (for [k (range row-start row-end)]
           (if (= (nth (:column-indices csr) k) j)
             (nth (:values csr) k)
             nil)))))))








(m/rows->RealMatrix (->dense {:values [10 20 30 40 50 60 70 80]
                              :column-indices [0  1  1  3  2  3  4  5]
                              :row-pointers [0  2  4  7  8]}
                             4 6))

(comment
  (import '[ml.dmlc.xgboost4j.java DMatrix DMatrix$SparseType])
  (DMatrix.
   (long-array (:row-pointers csr))
   (int-array (:column-indices csr))
   (float-array (:values csr))
   DMatrix$SparseType/CSR)
  )
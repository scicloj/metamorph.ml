(ns scicloj.metamorph.ml.create-term-idf-map
  (:require [clj-memory-meter.core :as mm]
            [criterium.core :as criterium]
            [ham-fisted.api :as hf]
            [ham-fisted.reduce :as hf-reduce]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [scicloj.metamorph.ml.text]
            [tech.v3.datatype :as dt]
            [tech.v3.dataset.reductions :as ds-reduce]
  (:import [java.io BufferedReader]))

(defn create-term->idf-map-5  [df]
  (let [N
        (float
         (tc/row-count
          (tc/unique-by df :document)))]
    (-> df
        (ds/unique-by #(vector (% :term-idx) (% :document)))
        (#(ds-reduce/group-by-column-agg
           :term-idx
           {:idf
            (ds-reduce/reducer :document
                               (constantly 0)
                               (fn [acc ^long v]
                                 (+ (long acc) 1))
                               +
                               (fn [^double v] (Math/log10 (/ N v))))}
           %)))))



(defn create-term->idf-map-3 [df]
  (hf/persistent!
   (let [N
         (float
          (tc/row-count
           (tc/unique-by df :document)))
         rows-by-term-idx
         (ds/group-by-column->indexes df :term-idx)]
     (hf-reduce/preduce
      (fn []  (hf/mut-long-hashtable-map))
      (fn [m [term-idx row-indices]]
        ;;(println :reduce term-idx)
        (let [documents
              (distinct (ds/select-rows (:document df) row-indices))
              idf (Math/log10 (/ N (count documents)))]
          (hf/assoc!  m  term-idx idf)))
      (fn [m-1 m-2]
        (hf/map-union + m-1 m-2))
      {;:cat-parallelism :elem-wise
       ;:min-n 1
       }

      rows-by-term-idx))))



(defn create-term->idf-map-2 [df]

  (let [keys-values
        (let [N
              (float
               (tc/row-count
                (tc/unique-by df :document)))
              rows-by-term-idx
              (ds/group-by-column->indexes df :term-idx)]
          (hf-reduce/preduce
           (fn [] {:term-idx (dt/make-list :int32)
                   :idf        (dt/make-list :float32)})
           (fn [m [term-idx row-indices]]
             (let [documents
                   (distinct (ds/select-rows (:document df) row-indices))
                   idf (Math/log10 (/ N (count documents)))]
               (.add (:term-idx m) term-idx)
               (.add (:idf m) idf))
             m)
           (fn [m-1 m-2]
             (.addAllReducible
              (:term-idx m-1)
              (:term-idx m-2))
             (.addAllReducible
              (:idf m-1)
              (:idf m-2))
             m-1)
           rows-by-term-idx))]

    (zipmap (:term-idx keys-values)
            (:idf keys-values))))

(defn create-term->idf-map-1 [df]

  (let [N
        (float
         (tc/row-count
          (tc/unique-by df :document)))]
    (hf-reduce/preduce
     (fn [] (hf/mut-map))
     (fn [m [term-idx row-indices]]
       (let [documents
             (hf/float-array-list
              (distinct (ds/select-rows (:document df) row-indices)))]
         (hf/assoc! m term-idx (Math/log10 (/ N (hf/constant-count documents))))))
     (fn [m-1 m-2]
       (hf/merge m-1 m-2))

     (ds/group-by-column->indexes df :term-idx))))





(comment
  (require '[criterium.core :as criterium])
  (def ds
    (ds/->dataset
     [{:term-idx 1, :term-pos 0, :document 0, :label 0} {:term-idx 2, :term-pos 1, :document 0, :label 0} {:term-idx 3, :term-pos 2, :document 0, :label 0} {:term-idx 3, :term-pos 3, :document 0, :label 0} {:term-idx 4, :term-pos 4, :document 0, :label 0} {:term-idx 1, :term-pos 0, :document 1, :label 1} {:term-idx 2, :term-pos 1, :document 1, :label 1} {:term-idx 5, :term-pos 2, :document 1, :label 1} {:term-idx 5, :term-pos 3, :document 1, :label 1} {:term-idx 6, :term-pos 4, :document 1, :label 1} {:term-idx 6, :term-pos 5, :document 1, :label 1} {:term-idx 6, :term-pos 6, :document 1, :label 1}]))


  (ds/group-by-column->indexes ds :term-idx)





  (def m-1 (create-term->idf-map-1 ds))
  (def m-2 (create-term->idf-map-2 ds))
  (def m-3 (create-term->idf-map-3 ds))
  (def m-4 (scicloj.metamorph.ml.text/create-term->idf-map-4 ds))

  (class m-1)
  (class m-2)
  (class m-3)

  ;;Execution time mean : 25.098424 ns
  (criterium/quick-bench
   (get m-1 3))

  ;;Execution time mean : 52.328324 ns
  (criterium/quick-bench
   (get m-2 3))

  ;;Execution time mean : 7.326608 ns
  (criterium/quick-bench
   (get m-3 3))

  ;execution time mean : 59.036666 µs
  (criterium/quick-bench
   (create-term->idf-map-1 ds))

  ;Execution time mean : 132.785200 µs
  (criterium/quick-bench
   (create-term->idf-map-2 ds))

  ;;Execution time mean : 55.492165 µs
  (criterium/quick-bench
   (create-term->idf-map-3 ds))

  ;;Execution time mean : 82.837948 µs
  (criterium/quick-bench
   (scicloj.metamorph.ml.text/create-term->idf-map-4 ds))


  (mm/measure m-1)
  ;;=> "608 B"
  (mm/measure m-2)
  ;;=> "288 B"
  (mm/measure m-3 :debug true)
  ;;=> "560 B"
  (mm/measure m-4 :debug true)
  ;;=> "1.1 KiB"
  )

(comment
  (require '[clojure.java.io :as io]
           '[clojure.string :as str]
           '[criterium.core :as criterium])
  (def df
    (:dataset
     (-> (scicloj.metamorph.ml.text/->tidy-text
          (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
          (fn [line] [line
                      (rand-int 6)])
          #(str/split % #" ")
          :max-lines 10000
          :skip-lines 1
          :datatype-document :int32
          :datatype-term-pos :int32
          :datatype-metas    :int8))))



  (time (def m-1 (create-term->idf-map-1 df)))
  (time (def m-2 (create-term->idf-map-2 df)))
  (time (def m-3 (create-term->idf-map-3 df)))
  (time (def m-4 (scicloj.metamorph.ml.text/create-term->idf-map-4 df)))
  (time (def m-5 (create-term->idf-map-5 df)))


  (tc/order-by m-5 :term-idx)
  (mm/measure m-1)
  ;;=> "10.4 MiB"

  (mm/measure m-2)
  ;;=> "7.8 MiB"

  (mm/measure m-3)
  ;;=> "8.5 MiB"

  (mm/measure m-4)
  ;;=> "959.6 KiB"


  (mm/measure m-5)
  ;;=> "2 MB""

  (criterium/quick-bench (get m-1 5))
; Execution time mean : 44.493164 ns

  (criterium/quick-bench (get m-2 5))
  ;execution time mean : 54.264319 ns

  (criterium/quick-bench (get m-3 5))
  ;Execution time mean : 25.109670 ns

  (def term-index (hf/->random-access (first m-4)))
  (def idf  (hf/->random-access (second m-4)))

  (criterium/quick-bench
   (->> 5
        (hf/binary-search term-index)
        (nth idf)))
  ;; 326 ns

  (def container
    (dt/make-container :int32  (repeat
                                (inc (apply max term-index))
                                -1)))
  (run!
   (fn [[i v]]
     (dt/set-value! container v i))
   (map vector (range) (first m-4)))


  (def reader
    (-> container
        dt/as-reader))



  (criterium/quick-bench
   (->> 5
        (nth reader)
        (nth idf)))
 ;122 ns
  )

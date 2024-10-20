(ns scicloj.metamorph.ml.text
  (:require [clj-memory-meter.core :as mm]
            [criterium.core :as criterium]
            [ham-fisted.api :as hf]
            [ham-fisted.reduce :as hf-reduce]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [ tech.v3.dataset.reductions :as ds-reduce]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.functional :as func]
            [tech.v3.parallel.for :as for])
  (:import [java.io BufferedReader]))



(defn- process-file [reader line-func
                     line-acc
                     max-lines skip-lines]
  (with-open [rdr (BufferedReader. reader)]
    (reduce line-func line-acc
            (take max-lines
                  (drop skip-lines
                        (line-seq rdr))))))


(defn- process-line [string-table line-split-fn text-tokenizer-fn acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        index-count (count tokens)]
    (.addAllReducible string-table tokens)
    (let [meta-list (:meta-list acc)
          index-list (:index-list acc)
          _ (.add meta-list meta)
          _ (.add index-list index-count)]
      (when (zero? (rem (dt/ecount index-list) 10000))
        (println (dt/ecount index-list)))
      acc)))

(def time-format (java.text.SimpleDateFormat. "HH:mm:ss.SSSS"))
(def prevoius-debug-time (atom (java.time.LocalTime/now)))
(defn debug [& s]
  (let [duration
        (.toSeconds
         (java.time.Duration/between
          @prevoius-debug-time
          (java.time.LocalTime/now)))]

    (reset! prevoius-debug-time (java.time.LocalTime/now))
    (println (format "  (%s) " duration))
    (apply print (.format time-format (java.util.Date.)) " - " s)))


(defn- make-col-container [map-fn container-type res-dataype  container-size datas]
  (let [container (dt/make-container container-type res-dataype container-size)
        metas
        (apply dt/emap map-fn res-dataype datas)]

    (dt/coalesce-blocks! container metas)))

(defn- make-metas-col-container [index-and-lable-lists col-size datatype]
  (make-col-container
   (fn [index meta]
     (dt/const-reader meta index))
   :jvm-heap
   datatype
   col-size
   [(:index-list index-and-lable-lists)
    (:meta-list index-and-lable-lists)]))


(defn- make-document-col-container [index-and-lable-lists col-size datatype container-type]
  (make-col-container
   (fn [idx count]
     (dt/const-reader idx count))
   container-type
   datatype
   col-size
   [(range)
    (:index-list index-and-lable-lists)]))


(defn- make-term-pos-col-container [index-and-lable-lists col-size datatype container-type]
  (make-col-container
   range
   container-type
   datatype
   col-size
   [(:index-list index-and-lable-lists)]))


(defn ->tidy-text
  "Reads, parses and tokenizes a text file into a tech.v3.dataset in the tidy-text format,
   so one word per row. 
   It does the parsing and conversion strictly line based, so it should work for large documents.

   Initial tests show that each byte of text size need one byte of heap.
   So a 8 GB text fil, can be sucessfully loaded when having at least 8 GB of heap for the JVM


   `line-split-fn` A fn which should seperate a single line of input in text and `other`
   Supposed to retrun a seq of size 2, where the first is teh 'text' of the line and `other` can be 
   anything (map, vector, scalar). It's value will be returned in column `meta` and is usppsoe dto be further processed
   `text-tokenizer-fn` A fuction which will be called for any `text` as obtained by `line-split-fn`
    It should split the text by word boundaries and return the obtained tokens as a seq of string.
    It can do text normalisation already.
   `skip-lines` Lines to skip at egining of file
   `max-lines` max lines to return
   "
  [reader line-split-fn
   text-tokenizer-fn

   & {:keys [skip-lines max-lines
             datatype-document
             datatype-term-pos
             datatype-metas
             container-type]
      :or {skip-lines  0
           datatype-document :int32
           datatype-term-pos :int16
           datatype-metas    :int8
           container-type    :jvm-heap
           max-lines Integer/MAX_VALUE}}]

  (let [term-index-string-table (st/string-table-from-strings [])
        _ (debug :parse)
        index-and-lable-lists
        (process-file reader
                      (partial process-line term-index-string-table line-split-fn text-tokenizer-fn)
                      {:meta-list (dt/make-list :object)
                       :index-list (dyn-int-list/dynamic-int-list)}
                      max-lines skip-lines)





        col-size (func/reduce-+ (:index-list index-and-lable-lists))
        _ (debug :count-index-aad-label-lists (count (:index-list index-and-lable-lists)))
        _ (debug :make-document-col-container)
        document-index (make-document-col-container index-and-lable-lists col-size datatype-document container-type)


        _ (debug :make-term-pos-col-container)
        term-pos (make-term-pos-col-container index-and-lable-lists col-size datatype-term-pos container-type)


        _ (debug :make-metas-col-container)
        metas (make-metas-col-container index-and-lable-lists col-size datatype-metas)

        _ (debug :measure-term-index-st (mm/measure (.data term-index-string-table)))
        _ (debug :measure-term-pos (mm/measure term-pos))
        _ (debug :measure-document-idx (mm/measure document-index))
        _ (debug :measure-metas (mm/measure metas))

        col-term-index (ds/new-column :term-idx  (st/indices term-index-string-table) {} [])
        col-term-pos (ds/new-column :term-pos  term-pos {} [])
        col-document (ds/new-column :document document-index {} [])
        col-meta (ds/new-column :meta metas {} [])
        ds
        (ds/new-dataset
         [col-term-index col-term-pos col-document col-meta])]

    (debug :string-table-count (count term-index-string-table))
    (debug :string-table-vocab-size (count (st/int->string term-index-string-table)))
    (debug :measure-term-index-string-table (mm/measure term-index-string-table))


    (debug :measure-col-term-index (mm/measure col-term-index))
    (debug :measure-col-term-pos (mm/measure col-term-pos))
    (debug :measure-col-document-idx (mm/measure col-document))
    (debug :measure-col-metas (mm/measure col-meta))
    (debug :measure-ds (mm/measure ds))


    {:dataset ds
     :int->str (st/int->string term-index-string-table)}))

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


(defn create-term->idf-map-4 [df]
  (let [N
        (float
         (tc/row-count
          (tc/unique-by df :document)))
        rows-by-term-index
        (ds/group-by-column->indexes df :term-idx)
        term-indices (hf/->random-access (keys rows-by-term-index))]

    (for/indexed-map-reduce
     (count term-indices)

     (fn [start-idx group-len]

       (let [idf-container  (dt/->writer (dt/make-container :float32 group-len))
             termindex-container (dt/->writer (dt/make-container :int32 group-len))]
         (run!
          (fn [idx]
            (let [term-index (nth term-indices idx)
                  row-indices (get  rows-by-term-index term-index)
                  documents (distinct (ds/select-rows (:document df) row-indices))
                  idf (Math/log10 (/ N (count documents)))]
              (dt/set-value! idf-container (- idx start-idx) idf)
              (dt/set-value! termindex-container (- idx start-idx) term-index)))
          (range  start-idx (+ start-idx group-len)))
         [termindex-container idf-container]))
     (fn [seq]
       (println :n-seq (count seq))
       (let [dst-termindex (dt/make-container :int32 (count rows-by-term-index))
             dst-idf (dt/make-container :float (count rows-by-term-index))]

         (dt/concat-buffers
          dst-termindex
          (map first seq))
         (dt/coalesce-blocks!
          dst-idf
          (map second seq))
         [dst-termindex dst-idf])
       (let [dst-termindex (dt/concat-buffers (map first seq))
             dst-idf (dt/concat-buffers (map second seq))]
         [dst-termindex dst-idf])))))




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


(defn ->tfidf [text]
  (let [term->idf (create-term->idf-map-4 text)
        term-index (hf/->random-access (first term->idf))
        idf  (hf/->random-access (second term->idf))
        container
        (dt/make-container :int32  (repeat
                                    (inc (apply max term-index))
                                    -1))
        _ (run!
           (fn [[i v]]
             (dt/set-value! container v i))
           (map vector (range) term-index))

        idf-reader
        (-> container
            dt/as-reader)

        term-idx->idf
        (fn [term-index]
          (->> term-index
               (nth idf-reader)
               (nth idf)))]

    (->>
     (hf-reduce/preduce (fn [] (hf/object-array-list))
                        (fn [l [document-idx row-indices]]
                          (let [term-idxs
                                (ds/select-rows (:term-idx text) row-indices)
                                freqs (hf/frequencies term-idxs)
                                n-terms (hf/constant-count row-indices)
                                term-count (hf/int-array-list  (hf/vals freqs))
                                tfs (hf/float-array-list (func// term-count (float n-terms)))
                                terms (hf/int-array-list (hf/keys freqs))
                                idfs (hf/float-array-list (hf/mapv term-idx->idf terms))
                                document (hf/int-array-list (hf/repeat (hf/constant-count freqs) document-idx))]

                            ;; (def row-indices row-indices)
                            ;; (def n-terms n-terms)
                            ;(def term-idxs term-idxs)
                            ;; (def freqs freqs)
                            ;; (def term-count term-count)
                            ;(def terms terms)
                            ;(def idfs idfs)
                            ;; (def tfs tfs)

                            (-> l
                                (hf/conj!
                                 (hf/hash-map
                                  :document document
                                  :term-idx terms
                                  :tf tfs
                                  :term-count term-count
                                  ;:n-terms (hf/int-array-list (hf/repeat (hf/constant-count freqs) n-terms))
                                  ;:idfs idfs
                                  :tfidf (hf/float-array-list (map * tfs idfs)))))))
                        (fn [list-1 list-2]
                          (hf/add-all! list-1 list-2))
                        (ds/group-by-column->indexes text :document))
     (hf/union-reduce-maps
      (fn [l-1 l-2]
        (hf/add-all! l-1 l-2)))
     (ds/->>dataset))))


(comment
  (require '[criterium.core :as criterium])
  (def ds
    (ds/->dataset
     [{:term-idx 1, :term-pos 0, :document 0, :label 0} {:term-idx 2, :term-pos 1, :document 0, :label 0} {:term-idx 3, :term-pos 2, :document 0, :label 0} {:term-idx 3, :term-pos 3, :document 0, :label 0} {:term-idx 4, :term-pos 4, :document 0, :label 0} {:term-idx 1, :term-pos 0, :document 1, :label 1} {:term-idx 2, :term-pos 1, :document 1, :label 1} {:term-idx 5, :term-pos 2, :document 1, :label 1} {:term-idx 5, :term-pos 3, :document 1, :label 1} {:term-idx 6, :term-pos 4, :document 1, :label 1} {:term-idx 6, :term-pos 5, :document 1, :label 1} {:term-idx 6, :term-pos 6, :document 1, :label 1}]))


  (ds/group-by-column->indexes ds :term-idx)





  (def m-1 (create-term->idf-map-1 ds))
  (def m-2 (create-term->idf-map-2 ds))
  (def m-3 (create-term->idf-map-3 ds))
  (def m-4 (create-term->idf-map-4 ds))

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
   (create-term->idf-map-4 ds))


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
     (-> (->tidy-text
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
  (time (def m-4 (create-term->idf-map-4 df)))
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
  

  
 

(ns scicloj.metamorph.ml.text
  (:require [clj-memory-meter.core :as mm]
            [ham-fisted.api :as hf]
            [ham-fisted.reduce :as hf-reduce]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.base :as ds-base]
            [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.functional :as func]
            [tech.v3.datatype.functional :as fun])
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
          _ (.add index-list index-count)
]
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
    (apply print (.format time-format (java.util.Date.)) " - " s))
  )


(defn- make-col-container [map-fn container-type res-dataype  container-size datas]
  (let [container (dt/make-container container-type res-dataype container-size)
        metas
        (apply dt/emap map-fn res-dataype datas)]
    
    (dt/coalesce-blocks! container metas))
  )

(defn- make-metas-col-container [index-and-lable-lists col-size datatype ]
  (make-col-container
   (fn [index meta]
     (dt/const-reader meta index))
   :jvm-heap
   datatype
   col-size
   [(:index-list index-and-lable-lists)
    (:meta-list index-and-lable-lists)
    ]
   ))


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

  (let [
        term-index-string-table (st/string-table-from-strings [])
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
     :int->str (st/int->string term-index-string-table)})
  )


(defn ->term-frequency [tidy-text-ds]
  (let [N
        (tc/row-count
         (tc/unique-by tidy-text-ds :document))

        n-tokens-per-document
        (-> tidy-text-ds
            (tc/group-by :document)
            (tc/aggregate {:n-terms-per-document ds/row-count})
            (tc/rename-columns {:$group-name :document}))
        idf
        (-> tidy-text-ds
            (tc/unique-by [:term-idx :document])
            (tc/group-by  [:term-idx])
            (tc/aggregate #(Math/log10 (/ N (tc/row-count %))))
            (tc/rename-columns {"summary" :idf}))]

    (-> tidy-text-ds
        (tc/left-join n-tokens-per-document [:document])
        (tc/left-join idf [:term-idx])
        (tc/group-by  [:term-idx :document :label])
        (tc/aggregate
         (fn [ds-per-token]
           (let [term-count (ds/row-count ds-per-token)
                 tf (float (/ term-count (first (:n-terms-per-document ds-per-token))))
                 idf (first (:idf ds-per-token))
                 tf-idf (* tf idf)]
             (hash-map
              :term-count term-count
              :tf tf
              :idf idf
              :tfidf tf-idf))))
        (tc/rename-columns {:summary-term-count :term-count
                            :summary-tf :tf
                            :summary-idf :idf
                            :summary-tfidf :tfidf})
        )))


(defn create-term->idf-map [df]
  (let [N 
        (double
         (tc/row-count
          (tc/unique-by df :document)))]
    (hf-reduce/preduce
     (fn [] (hf/mut-map))
     (fn [m [term-idx row-indices]]
       (let [documents 
             (hf/double-array-list
              (distinct (ds/select-rows (:document df) row-indices)))]
         (hf/assoc! m term-idx (Math/log10 (/ N (hf/constant-count documents))))))
     (fn [m-1 m-2]
       (hf/merge m-1 m-2))

     (ds/group-by-column->indexes df :term-idx))))


(defn ->tfidf [text]
  (let [term->idf-map
        (create-term->idf-map text)]
    (->>
     (hf-reduce/preduce (fn [] (hf/object-array-list))
                        (fn [l [document-idx row-indices]]
                          (let [term-idxs
                                (ds/select-rows (:term-idx text) row-indices)
                                freqs (hf/frequencies term-idxs)
                                n-terms (hf/constant-count row-indices)
                                term-count (hf/int-array-list (hf/vals freqs))
                                tfs (hf/double-array-list (fun// term-count (float n-terms)))
                                terms (hf/int-array-list (hf/keys freqs))
                                idfs (hf/double-array-list (map term->idf-map terms))]

                            ;; (def row-indices row-indices)
                            ;; (def n-terms n-terms)
                            ;; (def term-idxs term-idxs)
                            ;; (def freqs freqs)
                            ;; (def term-count term-count)
                            ;; (def terms terms)
                            ;; (def idfs idfs)
                            ;; (def tfs tfs)

                            (-> l
                                (hf/conj!
                                 (hf/hash-map
                                  :document (hf/int-array-list (hf/repeat (hf/constant-count freqs) document-idx))
                                  :term-idx terms
                                  :tf tfs
                                  :term-count term-count
                                  :n-terms (hf/int-array-list (hf/repeat (hf/constant-count freqs) n-terms))
                                  :idfs idfs
                                  :tfidf (hf/double-array-list (map * tfs idfs)))))))
                        (fn [list-1 list-2]
                          (hf/add-all! list-1 list-2))
                        (ds/group-by-column->indexes text :document))
     (hf/union-reduce-maps
      (fn [l-1 l-2]
        (hf/add-all! l-1 l-2)))
     (ds/->>dataset))))




(comment

  (require '[tech.v3.dataset.impl.column :as col-impl])
  (def col
    (tech.v3.dataset/new-column :text
                                (tech.v3.dataset.string-table/string-table-from-strings ["hello" "world" "hello"])
                                {}
                                []))

  (-> col .data class)
   ;;=> tech.v3.dataset.string_table.StringTable
;;OK

  (-> col dt/clone .data class)

;;=> ham_fisted.ArrayLists$ObjectArraySubList
;; not OK

  (-> col dt/clone ds-base/column->string-table)

;;=> Execution error at tech.v3.dataset.base/column->string-table (base.clj:847).
;;   Column :text does not contain a string table
;;   

  (-> col dt/clone ds-base/ensure-column-string-table .data)
  ;;=> [1 2 1]
;;OK, but reparses the data

;;=> [1 2 1]

  (->
   (ds/new-dataset [col])
   :text
   .data
   .data))






(comment
  (require '[criterium.core :as criterim])


  )


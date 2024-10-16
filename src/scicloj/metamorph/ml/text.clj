(ns scicloj.metamorph.ml.text
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column :as col]
            [tech.v3.dataset.impl.column :as col-impl]
            [tech.v3.dataset.base :as ds-base]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [tech.v3.dataset.impl.dataset :as ds-impl]
            [tech.v3.datatype :as dtt]
            [clj-memory-meter.core :as mm]
            [pppmap.core :as ppp]
            [tech.v3.datatype.functional :as func]
            [tech.v3.datatype.functional :as func]
            [tech.v3.datatype.functional :as fun])
  (:import [java.io BufferedReader]))


(defn- process-file [reader line-func line-acc max-lines skip-lines]
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
    (let [new-acc (conj acc
                        {:index-count index-count
                         :meta meta})]
      (when (zero? (rem (count new-acc) 10000))
        (println (count new-acc)))
      new-acc)))

(def time-format (java.text.SimpleDateFormat. "HH:mm:ss.SSSS"))
(def prevoius-debug-time (atom (java.time.LocalTime/now)))
(defn debug [& s]
  (let [duration
        (.toSeconds
         (java.time.Duration/between
          @prevoius-debug-time
          (java.time.LocalTime/now)))]
    
    (reset! prevoius-debug-time (java.time.LocalTime/now))
    (apply println (format "(%s) - " duration)  (.format time-format (java.util.Date.)) " - " s))
  )

(defn make-document-col-container-1 [index-counts-and-label]
  (->
   (map
    (fn [idx count]
      (dt/const-reader idx count))
    (range)
    (map :index-count index-counts-and-label))
   (dt/concat-buffers))
)


(defn make-document-col-container-2 [index-counts-and-label]
  (let [counts
        (dt/emap
         (fn [idx count]
           (dt/const-reader idx count))
         :int64
         (range)
         (map :index-count index-counts-and-label))

        length
        (func/reduce-+
         (seq
          (dt/emap #(dt/get-value (dt/shape %) 0) :int64 counts)))]
    (first
     (dt/copy-raw->item! counts (dt/make-container :int64 length)))
))



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

   & {:keys [skip-lines max-lines]
      :or {skip-lines  0
           max-lines Integer/MAX_VALUE}}]

  (let [string-table (st/string-table-from-strings [])
        _ (debug :parse)
        index-counts-and-label
        (process-file reader
                      (partial process-line string-table line-split-fn text-tokenizer-fn)
                      [] max-lines skip-lines)




        _ (def index-counts-and-label index-counts-and-label)
        _ (debug :count-index-nad-labels (count index-counts-and-label))
        _ (debug :line-idx)

        line-idx (make-document-col-container-2 index-counts-and-label)
        

        _ (debug :word-pos)
        word-pos
        (flatten
         (map
          #(range (:index-count %))
          index-counts-and-label))

        _ (debug :label)
        metas
        (flatten
         (map
          #(repeat (:index-count %) (:meta %))
          index-counts-and-label))

        _ (def string-table string-table)
        _ (def metas metas)



        _ (debug :count-indices
                   (count (st/indices string-table)))



        _ (debug :measure-data (mm/measure (.data string-table)))
        _ (debug :measure-word-pos (mm/measure word-pos))
        _ (debug :measure-line-idx (mm/measure line-idx))
        _ (debug :measure-metas (mm/measure metas))

        _ (debug :make-col-term-index)
        col-term-index
        (col-impl/construct-column [] (.data string-table)
                                   {:datatype :int16
                                    :name :term-idx})
        _ (debug :make-col-term-pos)
        col-term-pos
        (col-impl/construct-column []
                                   (dtt/make-container :int16 word-pos)
                                   {:datatype :int16 :name :term-pos})

        _ (debug :make-col-document)
        col-document
        (col-impl/construct-column []
                                   (dtt/make-container :int16 line-idx)
                                   {:datatype :int16 :name :document})
        _ (debug :make-col-meta)
        col-meta
        (col-impl/construct-column []
                                   (dtt/make-container :int16 metas)
                                   {:datatype :int16 :name :meta})
        _ (debug :make-ds)
        ds
        (ds/new-dataset
         [col-term-index col-term-pos col-document col-meta])]

    
    ;(debug :col-term-index col-term-index)
    ;(debug :col-term-pos  col-term-pos)
    ;(debug :col-line-idx  col-document)
    ;(debug :col-metas col-meta)
    ;(debug :string-table string-table)
    (debug :string-table-count (count string-table))

    (debug :measure-col-term-index (mm/measure col-term-index))
    (debug :measure-col-term-pos (mm/measure col-term-pos))
    (debug :measure-col-line-idx (mm/measure col-document))
    (debug :measure-col-metas (mm/measure col-meta))
    (debug :measure-string-table (mm/measure string-table))
    (debug :measure-ds (mm/measure ds))


    {:dataset ds
     :int->str (st/int->string string-table)})
  )


(defn- add-word-idx [tidy-text-ds]
  (let [term-col-as-string-table (ds-base/column->string-table (:term tidy-text-ds))
        word->int-table
        (-> (st/get-str-table term-col-as-string-table) :str->int)]

    (-> tidy-text-ds
        (tc/add-column
         :term-idx
         (fn [ds]
           (map #(get word->int-table %)
                (:term ds)))))))



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
        ;add-word-idx
        ;(tc/drop-columns [:term])
        )))






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

  (def document-1 (make-document-col-container-1 index-counts-and-label))
  (def document-2 (make-document-col-container-2 index-counts-and-label))

  (count document-2)

  (= document-1 document-2)
  (criterim/quick-bench
   (make-document-col-container-1 index-counts-and-label))
  (criterim/quick-bench
   (make-document-col-container-2 index-counts-and-label))


  )


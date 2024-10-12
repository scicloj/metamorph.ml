(ns scicloj.metamorph.ml.text
  (:require [clojure.string :as str]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.base :as ds-base]
            [tech.v3.dataset.impl.column :as col-impl]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [tech.v3.dataset.base :as ds-base])
  (:import [java.io BufferedReader FileReader]))




(defn- process-file [reader line-func line-acc max-lines skip-lines]
  (with-open [rdr (BufferedReader. reader)]
    (reduce line-func line-acc 
            (take max-lines 
                  (drop skip-lines
                        (line-seq rdr))))))


(defn- process-line [string-table text-label-split-fn acc line]
  (let [[text meta](text-label-split-fn line) 
        words (str/split text #" ")
        index-count (count words)]
    (.addAllReducible string-table words)
    (let [new-acc (conj acc 
                        {:index-count index-count
                         :meta meta})]
      (when (zero? (rem (count new-acc) 10000))
        (println (count new-acc)))
      new-acc))
  )

(defn ->tidy-text 
  "Reads, parses and tokenizes a text file into a tech.v3.dataset in the tidy-text format,
   so one word per row. 
   It does the parsing and conversion strictly line based, so it should work for large documents.

   Initial tests show that each byte of text size need one byte of heap.
   So a 8 GB text fil, can be sucessfully loaded when having at least 8 GB of heap for the JVM


   `line-split-fn` A fn which should seperate a single line of input in text and `other`
   Supposed to retrun a seq of size 2, where the first is teh 'text' of the line and `other` can be 
   anything (map, vector, scalar). It's value will be returned in column `meta` and is usppsoe dto be further processed
   `skip-lines` Lines to skip at egining of file
   `max-lines` max lines to return
   "
  [reader line-split-fn 
                   & {:keys [skip-lines max-lines] 
                      :or {skip-lines  0
                           max-lines Integer/MAX_VALUE
                                                  }}

                   ]

  (let [string-table (st/string-table-from-strings [])
        index-counts-and-label
        (process-file reader
                      (partial process-line string-table line-split-fn)
                      [] max-lines skip-lines)




        line-idx
        (->
         (map-indexed
          (fn [idx count]
            (dt/const-reader idx count))
          (map :index-count index-counts-and-label))
         (dt/concat-buffers))

        word-pos
        (flatten
         (map
          #(range (:index-count %))
          index-counts-and-label))

        metas
        (flatten
         (map
          #(repeat (:index-count %) (:meta %))
          index-counts-and-label))

        ds
        (ds/new-dataset
         [(ds/new-column :word string-table
                         {}
                         [])
               ;(ds/new-column :word string-table  [])
          (ds/new-column :word-index word-pos nil [])
          (ds/new-column :document line-idx nil [])
          (ds/new-column :meta metas nil [])])]

    ds
    )
    ;; drops empty string 
    ;;https://clojurians.zulipchat.com/#narrow/stream/236259-tech.2Eml.2Edataset.2Edev/topic/is.20empty.20string.20a.20.22missing.22.20.3F
      

     
    )


(defn ->term-frequency [tidy-text-ds]
  (-> tidy-text-ds
      (tc/group-by  [:word :document :label])
      (tc/aggregate #(hash-map :tf (ds/row-count %)))
      (tc/rename-columns {:summary-tf :tf})))


(defn add-word-idx [tidy-text-ds]
  (let [word->int-table
        (zipmap
         (-> tidy-text-ds :word .data st/int->string)
         (range))]
    (-> tidy-text-ds
        (tc/add-column
         :word-idx
         #(map word->int-table (:word %))))))



(def col
  (tech.v3.dataset/new-column :text
                              (tech.v3.dataset.string-table/string-table-from-strings ["hello" "world" "hello"])
                              {}
                              []))
(-> col .data class) ;;=> tech.v3.dataset.string_table.StringTable
;;OK

(-> col dt/clone .data class) 
;;=> ham_fisted.ArrayLists$ObjectArraySubList
;; not OK

(-> col dt/clone ds-base/column->string-table)
;;=> Execution error at tech.v3.dataset.base/column->string-table (base.clj:847).
;;   Column :text does not contain a string table
;;   

(-> col dt/clone ds-base/ensure-column-string-table .data);;=> [1 2 1]
;;OK, but reparses the data

;;=> [1 2 1]

(->
 (ds/new-dataset [col])
 :text
 .data
 .data
 )




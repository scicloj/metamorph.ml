(ns scicloj.metamorph.ml.text
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.base :as ds-base]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt])
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
        index-counts-and-label
        (process-file reader
                      (partial process-line string-table line-split-fn text-tokenizer-fn)
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
         [(ds/new-column :term string-table
                         {}
                         [])
               ;(ds/new-column :word string-table  [])
          (ds/new-column :term-index word-pos nil [])
          (ds/new-column :document line-idx nil [])
          (ds/new-column :meta metas nil [])])]

    ds)
    ;; drops empty string 
    ;;https://clojurians.zulipchat.com/#narrow/stream/236259-tech.2Eml.2Edataset.2Edev/topic/is.20empty.20string.20a.20.22missing.22.20.3F
  )

(defn ->term-frequency-old [tidy-text-ds]
  (-> tidy-text-ds
      (tc/group-by  [:term :document :label])
      (tc/aggregate #(hash-map :tterm-count (ds/row-count %)))
      (tc/rename-columns {:summary-term-count :term-count})))



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
            (tc/unique-by [:term :document])
            (tc/group-by  [:term])
            (tc/aggregate #(Math/log10 (/ N (tc/row-count %))))
            (tc/rename-columns {"summary" :idf}))]

    (-> tidy-text-ds
        (tc/left-join n-tokens-per-document [:document])
        (tc/left-join idf [:term])
        (tc/group-by  [:term :document :label])
        (tc/aggregate
         (fn [ds-per-token]
           (let [token-count (ds/row-count ds-per-token)
                 tf (float (/ token-count (first (:n-terms-per-document ds-per-token))))
                 idf (first (:idf ds-per-token))
                 tf-idf (* tf idf)]
             (hash-map
              :term-count token-count
              :tf tf
              :idf idf
              :tfidf tf-idf))))
        (tc/rename-columns {:summary-term-count :term-count
                            :summary-tf :tf
                            :summary-idf :idf
                            :summary-tfidf :tfidf}))))





(defn add-word-idx [tidy-text-ds]
  (let [word->int-table
        (zipmap
         (-> tidy-text-ds :term .data st/int->string)
         (range))]
    (-> tidy-text-ds
        (tc/add-column
         :term-idx
         #(map word->int-table (:term %))))))



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




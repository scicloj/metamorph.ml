(ns scicloj.metamorph.ml.text
  (:require [clj-memory-meter.core :as mm]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [ham-fisted.api :as hf]
            [ham-fisted.lazy-noncaching :as lznc]
            [ham-fisted.reduce :as hf-reduce]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
            [tech.v3.dataset.reductions :as reductions]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.functional :as func])
  (:import [java.io BufferedReader]
           [java.util ArrayList]))



(defn- process-file [reader line-func
                     line-acc
                     max-lines skip-lines]
  (with-open [rdr (BufferedReader. reader)]
    (reduce line-func line-acc
            (take max-lines
                  (drop skip-lines
                        (line-seq rdr))))))

(defn process-line [string-table line-split-fn text-tokenizer-fn acc line]
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


(defn- fill-string-table-from-line [string-table line-split-fn text-tokenizer-fn acc line]
  (let [[text _] (line-split-fn line)
        tokens (text-tokenizer-fn text)
        ]
    (.addAllReducible string-table tokens)
    (when (zero? (rem acc 1000))
      (println 
        acc " : "
        :num-tokens (dt/ecount string-table) " - "
               :num-unique-tokens (dt/ecount (st/int->string string-table)))
      )
    (inc acc)))



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
   term-index-string-table

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

  (let [ _ (debug :parse)
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

(defn create-term->idf-map [tidy-text]

  (let [ N
        (->
         (reductions/aggregate
          {:count (reductions/count-distinct :document)}
          tidy-text)
         :count
         first)]


    (reductions/group-by-column-agg
     :term-idx
     {:idf (reductions/reducer :document
                               (fn [] (hf/mut-set))
                               (fn [acc ^long document]
                                 (.add acc document  )
                                 acc)
                               (fn [uniq-documents-1 uniq-documents-2]
                                 (hf/add-all! uniq-documents-1 uniq-documents-2))
                               (fn [uniq-documents] (float (Math/log10 (/ N (count uniq-documents))))))}
     tidy-text)))



(defn ->tfidf-2 [tidy-text]
  (let [term->idf (create-term->idf-map tidy-text)]
     term-idx->idf-map
    (Long2FloatLinkedOpenHashMap. (-> term->idf :term-idx dt/->long-array)
                                  (-> term->idf :idf dt/->float-array))

    )
  )

(defn ->tfidf [tidy-text]
  (let [term->idf (create-term->idf-map tidy-text)
        term-index (:term-idx term->idf)
        idf  (:idf term->idf)
        container
        (dt/make-container :native-heap :int32  (repeat
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

    (println :start-tfidf)

    (->>
     (hf-reduce/preduce (fn [] (hf/object-array-list))
                        (fn [l [document-idx row-indices]]
                          (let [term-idxs
                                (ds/select-rows (:term-idx tidy-text) row-indices)
                                freqs (hf/frequencies term-idxs)
                                n-terms (hf/constant-count row-indices)
                                term-count (hf/int-array-list  (hf/vals freqs))
                                tfs (hf/float-array-list (func// term-count (float n-terms)))
                                terms (hf/int-array-list (hf/keys freqs))
                                idfs (hf/float-array-list (hf/mapv term-idx->idf terms))
                                document (hf/int-array-list (hf/repeat (hf/constant-count freqs) document-idx))]

                            (-> l
                                (hf/conj!
                                 (hf/hash-map
                                  :document document
                                  :term-idx terms
                                  :tf tfs
                                  :term-count term-count
                                  :tfidf (hf/float-array-list (map * tfs idfs)))))))
                        (fn [list-1 list-2]
                          (println :reduce (count list-1) (count list-2))
                          (hf/add-all! list-1 list-2))
                        {:max-batch-size 10000
                         :parallelism 6263}
                        (ds/group-by-column->indexes tidy-text :document))
     (hf/union-reduce-maps
      (fn [l-1 l-2]
        (hf/add-all! l-1 l-2)))
     (ds/->>dataset))))




(comment 
  (import '[org.mapdb DBMaker])
  
  (def db
    (.. DBMaker 
      ;(tempFileDB) 
      ;memoryDB
      ;heapDB
        (fileDB "/tmp/mapdb.bin")
        fileMmapEnable
        make))
  

  (def db-map (.. db ( hashMap "map") createOrOpen))
  


  (def term-index-string-table
    (st/->StringTable 
     (hf/object-array-list)
     db-map
     (dyn-int-list/dynamic-int-list)
     ))
  



  (process-file (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
                (partial fill-string-table-from-line term-index-string-table
                         (fn [line] [line
                                     (rand-int 6)])
                         #(str/split % #" ")
                         )

                0
                100000 1)
  
  

  


  (-> term-index-string-table st/int->string (get 641057))
  )
  
 
(comment
  (require '[clojure.data.csv :as csv]
           '[tablecloth.api :as tc]
           '[ham-fisted.lazy-noncaching :as lznc]
           '[ham-fisted.mut-map :as mut-map]
           '[criterium.core :as criterium])
  (set! *warn-on-reflection* true)
  (import '[it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap]
          [java.util List])

  (defn- parse-review-line [line]
    (let [splitted (first
                    (csv/read-csv line))]
      [(first splitted)
       (dec (Integer/parseInt (second splitted)))]))

  (def df
    (-> (->tidy-text (io/reader "test/data/reviews.csv")
                     parse-review-line
                     #(str/split % #" ")
                     (st/make-string-table)
                     :max-lines 5
                     :skip-lines 1)

        :dataset
        (tc/drop-columns [:term-pos :meta])))


  (def df
    (tc/dataset [{:term-idx 1, :term-pos 0, :document 0, :label 0} {:term-idx 2, :term-pos 1, :document 0, :label 0} {:term-idx 3, :term-pos 2, :document 0, :label 0} {:term-idx 3, :term-pos 3, :document 0, :label 0} {:term-idx 4, :term-pos 4, :document 0, :label 0} {:term-idx 1, :term-pos 0, :document 1, :label 1} {:term-idx 2, :term-pos 1, :document 1, :label 1} {:term-idx 5, :term-pos 2, :document 1, :label 1} {:term-idx 5, :term-pos 3, :document 1, :label 1} {:term-idx 6, :term-pos 4, :document 1, :label 1} {:term-idx 6, :term-pos 5, :document 1, :label 1} {:term-idx 6, :term-pos 6, :document 1, :label 1}]))

  (def idfs
    (create-term->idf-map df))




  
  
  (def term-idx->idf-map
    (Long2FloatLinkedOpenHashMap. (-> idfs :term-idx dt/->long-array)
                                  (-> idfs :idf dt/->float-array)))

  
  (def tfidf-data

    (reductions/group-by-column-agg
     :document
     {:tfidf-cols (reductions/reducer
                   [:term-idx :document]
                   (fn [] {:term-counts (hf/mut-map)
                           :term-counter 0
                           :document -1})
                   (fn [acc ^long term-idx ^long document]
                     (println  :rfn :document  document)
                     (mut-map/compute! (:term-counts acc)
                                       term-idx
                                       (fn [_ v] (if (nil? v)  1 (inc v))))
                     {:term-counts (:term-counts acc)

                      :term-counter (inc (:term-counter acc))
                      :document document})
                   (fn [acc-1 acc-2]
                     (println :combine)
                     {:term-counts (hf/merge-with + (:term-counts acc-1) (:term-counts acc-2))

                      :term-counter (+ (:term-counter acc-1) (:term-counter acc-2))
                      :document (or (:document acc-1) (:document acc-2))})

                   (fn [{:keys [term-counts term-counter document]}]
                     (let [tf-idfs
                           (apply hf/merge
                                  (lznc/map
                                   (fn [[term-index count]]
                                     (let [tf (float (/ count term-counter))]
                                       (hash-map term-index
                                                 (hash-map :tf tf
                                                           :tfidf (* tf (get term-idx->idf-map term-index))))))

                                   term-counts))]

                       {:term-idx (dt/make-container  :int64 (hf/keys tf-idfs))
                        :tf (dt/make-container :float32 (hf/mapv :tf (hf/vals tf-idfs)))
                        :tfidf (dt/make-container :float32  (hf/mapv :tfidf (hf/vals tf-idfs)))
                        :document (dt/const-reader document (count tf-idfs))})))}
     df))


  
  (ds/new-dataset
   [
    (ds/new-column :document
                   (dt/concat-buffers
                    (lznc/map
                     :document
                     (-> tfidf-data :tfidf-cols)))
                   {} [])
    
    (ds/new-column :tfidf 
                   (dt/concat-buffers
                    (lznc/map
                     :tfidf
                     (-> tfidf-data :tfidf-cols)))
                   {} [])
    (ds/new-column :tf
                   (dt/concat-buffers
                    (lznc/map
                     :tf
                     (-> tfidf-data :tfidf-cols)))
                   {} [])
    (ds/new-column :term-idx
                   (dt/concat-buffers
                    (lznc/map
                     :term-idx
                     (-> tfidf-data :tfidf-cols)))
                   {} [])])


  (mapv
   (fn [document len]
     (dt/const-reader document len)
     )
   (:document df)
   (map 
    #(-> % :tfidf count)
    (-> tfidf-data :tfidf-cols)
    ))
    

  )


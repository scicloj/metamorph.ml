(ns scicloj.metamorph.ml.text
  (:require [clj-memory-meter.core :as mm]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [ham-fisted.api :as hf]
            [ham-fisted.lazy-noncaching :as lznc]
            [ham-fisted.mut-map :as mut-map]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.impl.column :as col-impl]
            [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
            [tech.v3.dataset.reductions :as reductions]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.functional :as func]
            [scicloj.metamorph.ml.toydata :as data])
  (:import [ham_fisted IMutList]
           [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2LongRBTreeMap Long2LongArrayMap]
           [java.io BufferedReader]
           [java.util List Set]))

(set! *warn-on-reflection* true)





(defn- process-file [reader line-func
                     line-acc
                     max-lines skip-lines]
  (with-open [rdr (BufferedReader. reader)]
    (reduce line-func line-acc
            (take max-lines
                  (drop skip-lines
                        (line-seq rdr))))))

(defn process-line [^IMutList string-table line-split-fn text-tokenizer-fn acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        index-count (count tokens)]
    (.addAllReducible  string-table tokens)
    (let [meta-list (:meta-list acc)
          index-list (:index-list acc)
          _ (.add ^List meta-list meta)
          _ (.add ^List index-list index-count)]
      (when (zero? (rem (dt/ecount index-list) 10000))
        (println (dt/ecount index-list)))
      acc)))


(defn- fill-string-table-from-line [^IMutList string-table line-split-fn text-tokenizer-fn acc line]
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
  (debug :create-term->idf-map)
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
                               (fn [^Set acc ^long document]
                                 (when (zero? (rem document 100000))
                                   (println :reduce-idf document))
                                 (.add acc document)
                                 acc)
                               (fn [uniq-documents-1 uniq-documents-2]
                                 (hf/add-all! uniq-documents-1 uniq-documents-2))
                               (fn [uniq-documents] (float (Math/log10 (/ N (count uniq-documents))))))}
     tidy-text)))


(defn ->column [col-name data-type tfidf-data key]
  (let [data
        (->>
         (lznc/map key
                   (get tfidf-data :tfidf-cols))
         (hf/apply-concat)
         (dt/make-container data-type))
       
        meta-data {:datatype data-type
                   :name col-name}]
    (col-impl/construct-column [] data meta-data)))


(defn ->tfidf [tidy-text]

  (let [idfs (create-term->idf-map tidy-text)

        _ (debug :term-idx->idf-map)
        term-idx->idf-map
        (Long2FloatLinkedOpenHashMap. (-> idfs :term-idx dt/->long-array)
                                      (-> idfs :idf dt/->float-array))


        _ (debug :tfidf-data)
        tfidf-data
        (reductions/group-by-column-agg
         :document

         {:tfidf-cols (reductions/reducer
                       [:document :term-idx]
                       (fn [] {:term-counts (hf/mut-map)
                               :term-counter 0
                               :document nil})
                       (fn [acc ^long document ^long term-idx]
                         (println :rfn :document document :term-idx term-idx)
                         (mut-map/compute! (:term-counts acc)
                                           term-idx
                                           (fn [_ v] (if (nil? v)  1 (inc v))))
                         {:term-counts (:term-counts acc)
                          :term-counter (inc (:term-counter acc))
                          :document document})
                       (fn [acc-1 acc-2]
                         (throw (Exception. "merge should not get called"))
                         )

                       (fn [{:keys [term-counts term-counter document]}]
                         (println :finalize :term-counts term-counts 
                                  :term-counter term-counter
                                  :document document)
                         (let [tf-idfs
                               (apply hf/merge
                                      (lznc/map
                                       (fn [[term-index count]]
                                         (let [tf (float (/ count term-counter))]
                                           (hash-map term-index
                                                     (hash-map :tf tf
                                                               :tfidf (* tf
                                                                         (get term-idx->idf-map term-index))))))

                                       term-counts))]


                           {:document (dt/->int-array (hf/repeat document (count term-counts)))
                            :term-idx (dt/->int-array  (hf/keys tf-idfs))
                            :term-count (dt/->int-array  (hf/vals term-counts))
                            :tf (dt/->double-array (hf/mapv :tf (hf/vals tf-idfs)))
                            :tfidf (dt/->double-array (hf/mapv :tfidf (hf/vals tf-idfs)))})))}
         tidy-text)]


  
    (ds/new-dataset
     [(->column :document :int32 tfidf-data :document)
      (->column :tfidf :float32 tfidf-data :tfidf)
      (->column :tf :float32 tfidf-data :tf)
      (->column :term-idx :int32 tfidf-data :term-idx)
      (->column :term-count :int32 tfidf-data :term-count)
      
])))

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

  (def tfidf
    (-> df
        ->tfidf))
  

  (-> tfidf :tfidf-cols first)
  ; :term-idx #array-buffer<int64>[68]
  ; :term-count #array-buffer<int64>[68]
  ; :tf #array-buffer<float32>[68]
  ;:tfidf #array-buffer<object>[68]
  
  (mm/measure (-> tfidf :tfidf-cols first) :debug true)
  
  
  ( count
   )
  (-> tfidf :tfidf )
  (mm/measure
   (-> tfidf :tfidf .data ) :debug true)
  )
  
(* 68 4)


(def m-1 (Long2LongRBTreeMap.))
(def m-2 (Long2LongArrayMap.))

(defn fill! [m]
  (run!
   (fn [[ k v]] 
     (.put m k v)
     )
   (map vector (range 1000) (reverse (range 1000)))))

(fill! m-1)
(fill! m-2)

(mm/measure m-1) 

(mm/measure m-2) 
(def it
  (.fastIterator
   (.long2LongEntrySet  m-2)))

(class (.getLongKey (.next it)))


(mm/measure
 (long-array [1 2 3 4])
 :debug true)

(mm/measure
 (object-array [1 2 3 4])
 :debug true)

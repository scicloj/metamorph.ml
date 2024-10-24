(ns scicloj.metamorph.ml.text
  (:require
   [clj-memory-meter.core :as mm]
   [ham-fisted.api :as hf]
   [ham-fisted.lazy-noncaching :as lznc]
   [ham-fisted.set :as hf-set]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
   [tech.v3.dataset.impl.column :as col-impl]
   [tech.v3.dataset.reductions :as reductions]
   [tech.v3.dataset.string-table :as st]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.functional :as func])
  (:import
   [ham_fisted IMutList]
   [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2IntOpenHashMap]
   [java.io BufferedReader]
   [java.util List]
   [org.mapdb DBMaker]))

(set! *warn-on-reflection* true)
;;(set! *unchecked-math* :warn-on-boxed)


(defn- process-file [reader line-func
                     line-acc
                     max-lines skip-lines]
  (with-open [rdr (BufferedReader. reader)]
    (reduce line-func line-acc
            (take max-lines
                  (drop skip-lines
                        (line-seq rdr))))))

(defn- put-retrieve-token! [token->long token]
  (if (contains? token->long token)
    (get token->long token)
    (let [next-token (hf/constant-count token->long)]
      (hf/assoc! token->long token next-token)
      next-token)))


(defn process-line [token->long line-split-fn text-tokenizer-fn acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        token-indices (map (partial put-retrieve-token! token->long) tokens)
        index-count (count tokens)
        meta-list (:meta-list acc)
        index-list (:index-list acc)
        term-list (:term-list acc)]
    
    (.add ^List meta-list meta)
    (.add ^List index-list index-count)
    (.addAll ^List term-list token-indices)

    (when (zero? (rem (dt/ecount index-list) 10000))
      (println (dt/ecount index-list)))
    acc))




(defn- fill-string-table-from-line! [^IMutList string-table line-split-fn text-tokenizer-fn acc line]
  (let [[text _] (line-split-fn line)
        tokens (text-tokenizer-fn text)]
    (.addAllReducible string-table tokens)
    (when (zero? (rem acc 1000))
      (println
       acc " : "
       :num-tokens (dt/ecount string-table) " - "
       :num-unique-tokens (dt/ecount (st/int->string string-table))))
    (inc acc)))





(defn heap-string-table []
  (st/make-string-table [])
  )

(defn mapdb-string-table [^org.mapdb.DB db]
  (st/->StringTable
   (hf/object-array-list)
   (.. db (hashMap "map") createOrOpen)
   (dyn-int-list/dynamic-int-list)))







(defn fill-string-table! [reader term-index-string-table
                           line-split-fn text-tokenizer-fn
                           max-lines skip-lines]


  (process-file reader
                (partial fill-string-table-from-line! term-index-string-table line-split-fn text-tokenizer-fn)
                0
                max-lines skip-lines))


(def time-format  (java.text.SimpleDateFormat. "HH:mm:ss.SSSS"))
(def prevoius-debug-time (atom (java.time.LocalTime/now)))
(defn debug [& s]
  (let [duration
        (.toSeconds
         (java.time.Duration/between
          @prevoius-debug-time
          (java.time.LocalTime/now)))]

    (reset! prevoius-debug-time (java.time.LocalTime/now))
    (println (format "  (%s) " duration))
    (apply print (.format ^java.text.SimpleDateFormat time-format 
                          (java.util.Date.)) " - " s)))


(defn- make-col-container [map-fn container-type res-dataype  container-size datas]
  (let [
        col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?) ; prevennts 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         )]
    (dt/concat-buffers res-dataype col-datas)))


(defn- make-metas-col-container [index-and-lable-lists col-size datatype container-type]
  (make-col-container
   (fn [index meta]
     (dt/const-reader meta index))
   (if (= :object datatype)
     :jvm-heap
     container-type)
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
   [(range (count (:index-list index-and-lable-lists)))
    (:index-list index-and-lable-lists)]))



(defn- make-term-pos-col-container [index-and-lable-lists col-size datatype container-type]
  (make-col-container
   range
   container-type
   datatype
   col-size
   [(:index-list index-and-lable-lists)]))




(defn ->tidy-text
  "Reads, parses and tokenizes a text file into a seq of tech.v3.dataset in the tidy-text format,
   so one word per row. 
   It does the parsing and conversion strictly line based, so it should work for large documents.

   Initial tests show that each byte of text size need one byte of heap.
   So a 8 GB text file can be sucessfully loaded when having at least 8 GB of heap for the JVM


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
             datatype-term-idx
             container-type]
      :or {skip-lines  0
           datatype-document :int32
           datatype-term-pos :int16
           datatype-metas    :int8
           datatype-term-idx :int32
           container-type    :jvm-heap
           max-lines Integer/MAX_VALUE}}]

  (let [_ (debug :parse)
        token->long (hf/mut-map [ ["" 0]])
        index-and-lable-lists
        (process-file reader
                      (partial process-line  token->long line-split-fn text-tokenizer-fn)
                      {:meta-list (dt/make-list datatype-metas)
                       :term-list (dt/make-list datatype-term-idx)
                       :index-list (dyn-int-list/dynamic-int-list)}
                      max-lines skip-lines)


        col-size (func/reduce-+ (:index-list index-and-lable-lists))
        _ (debug :count-index-aad-label-lists (count (:index-list index-and-lable-lists)))
        _ (debug :make-document-col-container)
        document-index (make-document-col-container index-and-lable-lists col-size datatype-document container-type)


        _ (debug :make-term-pos-col-container)
        term-pos (make-term-pos-col-container index-and-lable-lists col-size datatype-term-pos container-type)


        _ (debug :make-metas-col-container)
        metas (make-metas-col-container index-and-lable-lists col-size datatype-metas container-type)

        _ (debug :make-term-indx-container)

        term-idx (dt/make-container container-type datatype-term-idx (:term-list index-and-lable-lists))

        _ (debug :measure-term-index (mm/measure term-idx))
        _ (debug :measure-term-pos (mm/measure term-pos))
        _ (debug :measure-document-idx (mm/measure document-index))
        _ (debug :measure-metas (mm/measure metas))

        col-term-index (ds/new-column :term-idx term-idx  {} [])
        col-term-pos (ds/new-column :term-pos  term-pos {} [])
        col-document (ds/new-column :document document-index {} [])
        col-meta (ds/new-column :meta metas {} [])
        ds
        (ds/new-dataset
         [col-term-index col-term-pos col-document col-meta])]

    (debug :token->long-count (count token->long))
    (debug :measure-token->long (mm/measure token->long))


    (debug :measure-col-term-index (mm/measure col-term-index))
    (debug :measure-col-term-pos (mm/measure col-term-pos))
    (debug :measure-col-document-idx (mm/measure col-document))
    (debug :measure-col-metas (mm/measure col-meta))
    (debug :measure-ds (mm/measure ds))


    {:datasets [ds]
     :token->long token->long}))

(defn create-term->idf-map [tidy-text]
  (debug :create-term->idf-map)
  (let [N
        (->
         (reductions/aggregate
          {:count (reductions/count-distinct :document)}
          tidy-text)
         :count
         first)]


    (reductions/group-by-column-agg
     :term-idx
     {:idf (reductions/reducer :document
                               (fn [] (hf/long-array-list))
                               (fn [ acc ^long document]
                                 (when (zero? (rem document 10000))
                                   (println :reduce-idf document))
                                 (conj acc document)
                                 )
                               (fn [documents-1 documents-2]
                                 (hf/add-all! documents-1 documents-2))
                               (fn [documents] 
                                 (let [n-uniq-docs (count (hf-set/unique documents))]
                                   (float (Math/log10 (/ N n-uniq-docs))))))}
     tidy-text)))


(defn ->column [col-name data-type tfidf-data key ]

 (debug :->-col col-name)
  (let [data
        (->>
         (lznc/map key
                   (get tfidf-data :tfidf-cols))
         
         (dt/concat-buffers))

        meta-data {:datatype data-type
                   :name col-name}]
    (col-impl/construct-column [] data meta-data)))

(defn- >document-col [tfidf-data data-type]
 (debug :->document-col)
  (let [tfids-lengths
        (map #(-> % :tfidf count)
             (-> tfidf-data :tfidf-cols))
        data
        (->>
         (lznc/map
          (fn [doc-id len] (dt/const-reader doc-id len))
          (-> tfidf-data :document)
          tfids-lengths)
         (dt/concat-buffers))
        meta-data{:name :document :datatype data-type}]

    (col-impl/construct-column [] data meta-data)))


(defn- tf-idf-reducer [term-idx->idf-map container-type]
  (reductions/reducer
   [:document :term-idx]
   (fn [] {:term-counts  (Long2IntOpenHashMap.)
           :term-counter 0
           :document nil})
   (fn [acc ^long document ^long term-idx]
     (when (zero? (rem document 10000))
       (println :reduce-tfidf document))
  
     (.addTo ^Long2IntOpenHashMap  (:term-counts acc) term-idx 1)
     {:term-counts (:term-counts acc)
      :term-counter (inc (:term-counter acc))
      :document document})
   (fn [acc-1 acc-2]
     (throw (Exception. "merge should not get called")))
  
   (fn [{:keys [term-counts term-counter document]}]
     (when (zero? (rem  document 1000))
       (println :finalize-tfidf document))
     (let [term->tfidf-fn
           (fn [[term-index count]]
             (let [tf (float (/ count term-counter))]
               {term-index
                {:tf tf
                 :tfidf (* tf (get term-idx->idf-map term-index))}}))
  
           tf-idfs
           (apply hf/merge (lznc/map term->tfidf-fn term-counts))]
  
  
       {:term-idx (dt/make-container container-type :int32  (hf/keys tf-idfs))
        :term-count (dt/make-container container-type  :int32 (hf/vals term-counts))
        :tf (dt/make-container container-type :float32 (hf/mapv :tf (hf/vals tf-idfs)))
        :tfidf (dt/make-container container-type :float32 (hf/mapv :tfidf (hf/vals tf-idfs)))})))
  )


(defn ->tfidf [tidy-text &  {:keys [container-type] 
                             :or {container-type :jvm-heap}}]

  (println :container-type container-type)
  (let [idfs (create-term->idf-map tidy-text)

        _ (debug :term-idx->idf-map)
        term-idx->idf-map
        (Long2FloatLinkedOpenHashMap. (-> idfs :term-idx dt/->long-array)
                                      (-> idfs :idf dt/->float-array))


        _ (debug :tfidf-data)

        tfidf-data
        (reductions/group-by-column-agg
         :document

         {:tfidf-cols (tf-idf-reducer term-idx->idf-map container-type)}
         tidy-text)]

    (println :new-dataset)
    (ds/new-dataset
     [(>document-col tfidf-data container-type)
      (->column :tfidf container-type tfidf-data :tfidf)
      (->column :tf container-type tfidf-data :tf)
      (->column :term-idx container-type tfidf-data :term-idx)
      (->column :term-count container-type tfidf-data :term-count)])))



(def c
  (dt/make-container :native-heap :int16 10))

(def l
  (tech.v3.datatype.list/make-list c 0))


(get! l .capacity)

(def l (dt/make-list :int))
(.ptr l)
(map :name
     (:members
      (clojure.reflect/map->Field l)))

(class l)
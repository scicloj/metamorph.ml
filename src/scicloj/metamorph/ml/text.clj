(ns scicloj.metamorph.ml.text
  (:require
   [clj-memory-meter.core :as mm]
   [ham-fisted.api :as hf]
   [ham-fisted.lazy-noncaching :as lznc]
   [ham-fisted.set :as hf-set]
   [scicloj.metamorph.ml.tools :as tools]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.dynamic-int-list :as dyn-int-list]
   [tech.v3.dataset.impl.column :as col-impl]
   [tech.v3.dataset.reductions :as reductions]
   [tech.v3.dataset.string-table :as st]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.functional :as func])
  (:import
   [ham_fisted IMutList]
   [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2IntOpenHashMap LongOpenHashSet]
   [java.io BufferedReader]
   [java.util List Set]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)



(defn- make-col-container [map-fn  container-type res-dataype  datas]
  
  (let [col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?) ; prevennts 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         )
        col-size (or (func/reduce-+ (map count col-datas)) 0)
        container (dt/make-container container-type res-dataype col-size)]

    (when (not-empty col-datas)
      (dt/coalesce-blocks! container col-datas))
    container))


(defn- make-metas-col-container [index-and-lable-lists container-type datatype]
  (make-col-container
   (fn [index meta]
     (dt/const-reader meta index))
   (if (= :object datatype)
     :jvm-heap
     container-type)
   datatype
   [(:index-list index-and-lable-lists)
    (:meta-list index-and-lable-lists)]))


(defn- make-document-col-container [index-and-lable-lists container-type datatype]
  (make-col-container
   (fn [idx count]
     (dt/const-reader idx count))
   container-type
   datatype

   [(range (count (:index-list index-and-lable-lists)))
    (:index-list index-and-lable-lists)]))



(defn- make-term-pos-col-container [index-and-lable-lists container-type datatype]
  (make-col-container
   range
   container-type
   datatype

   [(:index-list index-and-lable-lists)]))



(defn process-line [token->long line-split-fn text-tokenizer-fn
                    acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        token-indices (map (partial tools/put-retrieve-token! token->long) tokens)
        index-count (count tokens)
        meta-list (:meta-list acc)
        index-list (:index-list acc)
        term-list (:term-list acc)]

    (.add ^List meta-list meta)
    (.add ^List index-list index-count)
    (.addAll ^List term-list token-indices)


    (when (zero? (rem (count index-list) 10000))
      (tools/debug :count (count index-list))))
    acc)


(defn- fill-string-table-from-line! [^IMutList string-table line-split-fn text-tokenizer-fn acc line]
  (let [[text _] (line-split-fn line)
        tokens (text-tokenizer-fn text)]
    (.addAllReducible string-table tokens)
    (when (zero? ^long (rem ^long acc 1000))
      (println
       acc " : "
       :num-tokens (dt/ecount string-table) " - "
       :num-unique-tokens (dt/ecount (st/int->string string-table))))
    (inc ^long acc)))



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


  (tools/process-file reader
                      line-seq
                (partial fill-string-table-from-line! 
                         term-index-string-table 
                         line-split-fn 
                         text-tokenizer-fn)
                0
                max-lines skip-lines))





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

  (let [_ (tools/debug :parse)
        token->long (hf/mut-map [["" 0]])
        acc
        (tools/process-file reader
                      line-seq
                      (partial process-line  token->long line-split-fn text-tokenizer-fn)
                      {:meta-list (dt/make-list datatype-metas)
                       :term-list (dt/make-list datatype-term-idx)
                       :index-list (dt/make-list datatype-document)}
                      max-lines skip-lines)


        term-pos-container (make-term-pos-col-container acc container-type datatype-term-pos)
        metas-container (make-metas-col-container acc container-type datatype-metas)
        document-container (make-document-col-container acc container-type datatype-document)
        term-index-container (dt/make-container container-type datatype-term-idx (:term-list acc))


        col-term-index (ds/new-column :term-idx term-index-container  {} [])
        col-term-pos (ds/new-column :term-pos  term-pos-container {} [])
        col-document (ds/new-column :document  document-container {} [])
        col-meta (ds/new-column :meta metas-container {} [])

        _ (tools/debug :measure-term-index (mm/measure col-term-index))
        _ (tools/debug :measure-term-pos (mm/measure col-term-pos))
        _ (tools/debug :measure-document-idx (mm/measure col-document))
        _ (tools/debug :measure-metas (mm/measure col-meta))

        ds
        (ds/new-dataset
         [col-term-index col-term-pos col-document col-meta])]

    (tools/debug :token->long-count (count token->long))
    (tools/debug :measure-token->long (mm/measure token->long))


    (tools/debug :measure-col-term-index (mm/measure col-term-index))
    (tools/debug :measure-col-term-pos (mm/measure col-term-pos))
    (tools/debug :measure-col-document-idx (mm/measure col-document))
    (tools/debug :measure-col-metas (mm/measure col-meta))
    (tools/debug :measure-ds (mm/measure ds))


    {:datasets [ds]
     :token->long token->long}))

(defn create-term->idf-map [tidy-text]
  (tools/debug :create-term->idf-map)
  (let [N
        (->
         (reductions/aggregate
          {:count (reductions/count-distinct :document)}
          tidy-text)
         :count
         first)]


    (tools/debug :N N)
    (reductions/group-by-column-agg
     :term-idx
     {:idf (reductions/reducer :document
                               (fn [] (LongOpenHashSet.))
                               (fn [ acc ^long document]
                                 (when (zero? (rem document 10000))
                                   (println :reduce-idf document))
                                 (.add ^LongOpenHashSet acc document)
                                 acc
                                 )
                               (fn [^LongOpenHashSet documents-1 ^LongOpenHashSet documents-2]
                                 (println :merge-idf)
                                 (.addAll documents-1 documents-2))
                               (fn [documents]
                                 (println :finalize-idf) 
                                 (let [n-uniq-docs ^int  (.size ^LongOpenHashSet documents)]
                                   (func/log10 ^float (/  ^float N ^int n-uniq-docs)))))}
     tidy-text)))



(defn ->column [col-name container-type data-type tfidf-data key]
  (tools/debug :->-col col-name)
  
  (let [cont-size
        (func/sum-fast
         (lznc/map #(dt/ecount (get % key))
                   (get tfidf-data :tfidf-cols)))

        container (dt/make-container container-type data-type cont-size)

        data (dt/coalesce-blocks!
              container
              (lznc/map key
                        (get tfidf-data :tfidf-cols)))
        meta-data {:datatype data-type
                   :name col-name}]
    (col-impl/construct-column [] data meta-data))

  )

(defn- >document-col [container-type data-type  tfidf-data]
 (tools/debug :->document-col)
  (let [tfidfs-lengths
        (map #(-> % :tfidf count)
             (-> tfidf-data :tfidf-cols))
        cont-size (func/sum-fast tfidfs-lengths)
        container (dt/make-container container-type data-type cont-size)
        data
        (->>
         (lznc/map
          (fn [doc-id len] (dt/const-reader doc-id len))
          (-> tfidf-data :document)
          tfidfs-lengths)
         (dt/coalesce-blocks! container))
        meta-data {:name :document
                   :datatype data-type}]

    (col-impl/construct-column [] data meta-data)))


(defn- tf-idf-reducer [term-idx->idf-map container-type]
  (reductions/reducer
   [:document :term-idx]
   (fn [] {:term-counts  (Long2IntOpenHashMap.)
           :term-counter 0
           :document nil})
   (fn [acc ^long document ^long term-idx]
     ;(tools/debug :reduce-tfidf document)
     (when (and (zero? (rem document 10000)) (zero? ^long (:term-counter acc ) ))
       (tools/debug :reduce-tfidf document ) )
  
     (.addTo ^Long2IntOpenHashMap  (:term-counts acc) term-idx 1)
     {:term-counts (:term-counts acc)
      :term-counter (inc ^long (:term-counter acc))
      :document document})
   (fn [acc-1 acc-2]
     (throw (Exception. "merge should not get called")))
  
   (fn [{:keys [term-counts ^long term-counter ^long document]}]

     ;(tools/debug :finalize-tfidf document)
     (when (zero? (rem  document 10000))
       (tools/debug :finalize-tfidf document))
     
     
     
     (let [term->tfidf-fn
           (fn [[^long term-index ^long count]]
             (let [tf ^float (/ count term-counter)]
               {term-index
                {:tf tf
                 :tfidf ^float (* ^float tf ^float (get term-idx->idf-map term-index))}}))
  
           tf-idfs
           (apply hf/merge (lznc/map term->tfidf-fn term-counts))]
  
  
       {:term-idx (dt/make-container container-type :int32  (hf/keys tf-idfs))
        :term-count (dt/make-container container-type  :int32 (hf/vals term-counts))
        :tf (dt/make-container container-type :float32 (hf/mapv :tf (hf/vals tf-idfs)))
        :tfidf (dt/make-container container-type :float32 (hf/mapv :tfidf (hf/vals tf-idfs)))})))
  )


(defn ->tfidf [tidy-text &  {:keys [container-type] 
                             :or {container-type :jvm-heap}}]

  (let [idfs (create-term->idf-map tidy-text)

        _ (tools/debug :term-idx->idf-map)
        term-idx->idf-map
        (Long2FloatLinkedOpenHashMap. (-> idfs :term-idx dt/->long-array)
                                      (-> idfs :idf dt/->float-array))


       _ (println :measure-term-idx->idf-map 
             (mm/measure term-idx->idf-map))

       _ (tools/debug :tfidf-data)
        tfidf-data
        (reductions/group-by-column-agg
         :document
         {:tfidf-cols (tf-idf-reducer term-idx->idf-map container-type)}
         tidy-text)]

    (println :new-dataset)
    (ds/new-dataset
     [(>document-col container-type :int32 tfidf-data)
      (->column :tfidf container-type :float32 tfidf-data :tfidf)
      (->column :tf container-type :float32 tfidf-data :tf)
      (->column :term-idx container-type :int32 tfidf-data :term-idx)
      (->column :term-count container-type :int16 tfidf-data :term-count)])))


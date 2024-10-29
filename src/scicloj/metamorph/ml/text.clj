(ns scicloj.metamorph.ml.text
  (:require
   [clj-memory-meter.core :as mm]
   [ham-fisted.api :as hf]
   [ham-fisted.lazy-noncaching :as lznc]
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
   [it.unimi.dsi.fastutil.objects Object2IntLinkedOpenHashMap Object2IntMaps Object2LongLinkedOpenHashMap]
   [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2IntOpenHashMap LongOpenHashSet]
   [java.util List]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)





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
  (st/make-string-table []))
  

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
                                   (tools/debug :reduce-idf document))
                                 (.add ^LongOpenHashSet acc document)
                                 acc)
                                 
                               (fn [^LongOpenHashSet documents-1 ^LongOpenHashSet documents-2]
                                 (tools/debug :merge-idf)
                                 (.addAll documents-1 documents-2))
                               (fn [documents]
                                 (tools/debug :finalize-idf) 
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
    (col-impl/construct-column [] data meta-data)))

  

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
     (when (and (zero? (rem document 10000)) (zero? ^long (:term-counter acc)))
       (tools/debug :reduce-tfidf document))
  
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
        :tfidf (dt/make-container container-type :float32 (hf/mapv :tfidf (hf/vals tf-idfs)))}))))
  


(defn ->tfidf [tidy-text &  {:keys [container-type] 
                             :or {container-type :jvm-heap}}]

  (let [idfs (create-term->idf-map tidy-text)

        _ (tools/debug :term-idx->idf-map)
        term-idx->idf-map
        (Long2FloatLinkedOpenHashMap. (-> idfs :term-idx dt/->long-array)
                                      (-> idfs :idf dt/->float-array))


        _ (tools/debug :measure-term-idx->idf-map 
              (mm/measure term-idx->idf-map))

        _ (tools/debug :tfidf-data)
        tfidf-data
        (reductions/group-by-column-agg
         :document
         {:tfidf-cols (tf-idf-reducer term-idx->idf-map container-type)}
         tidy-text)]

    (ds/new-dataset
     [(>document-col container-type :int32 tfidf-data)
      (->column :tfidf container-type :float32 tfidf-data :tfidf)
      (->column :tf container-type :float32 tfidf-data :tf)
      (->column :term-idx container-type :int32 tfidf-data :term-idx)
      (->column :term-count container-type :int16 tfidf-data :term-count)])))


(defn- make-col-container--concat-buffers [map-fn  container-type res-dataype  datas]
  (let [col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?))] ; prevents 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         

    (dt/concat-buffers res-dataype col-datas)))


(defn- make-col-container--coalesce-blocks! [map-fn  container-type res-dataype  datas]
  (let [col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?)) ; prevents 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         
        col-size (or (func/reduce-+ (map count col-datas)) 0)
        container (dt/make-container container-type res-dataype col-size)]

    (when (not-empty col-datas)
      (dt/coalesce-blocks! container col-datas))
    container))

(defn- make-col-container [map-fn combine-method container-type res-dataype  datas]
  (case combine-method
    :coalesce-blocks! (make-col-container--coalesce-blocks! map-fn container-type res-dataype datas)
    :concat-buffers (make-col-container--concat-buffers map-fn container-type res-dataype datas)))

(defn- make-meta-col-container [acc combine-method container-type datatype]
  (when (seq (:meta-list acc))
    (make-col-container
     (fn [index meta]
       (dt/const-reader meta index))
     combine-method
     (if (= :object datatype)
       :jvm-heap
       container-type)
     datatype
     [(:index-list acc)
      (:meta-list acc)])))

(defn- range-2 [ a b]
  (range a (+ ^int a ^int b)))


(defn- make-document-col-container [acc combine-method container-type datatype]
  (let [n-docs-parsed (:n-docs-parsed acc)]
    (make-col-container
     (fn [idx count]
       (dt/const-reader idx count))
     combine-method
     container-type
     datatype
     [(range-2 (- ^int n-docs-parsed ^int (count (:index-list acc)))
               (count (:index-list acc)))

      (:index-list acc)])))


(defn- make-term-pos-col-container [acc combine-method container-type datatype]
  (make-col-container
   range
   combine-method
   container-type
   datatype
   [(:index-list acc)]))


;; TODO : try using this
(defn- make-term-index-col-container [acc combine-method container-type datatype]
  (make-col-container
   (fn [x] [x])
   combine-method
   
   container-type
   datatype
   [(:term-list acc)]))


(defn- update-acc! [acc combine-method container-type datatype-term-pos datatype-meta datatype-document datatype-term-idx]

  (let [term-pos-container (make-term-pos-col-container acc combine-method container-type datatype-term-pos)
        metas-container (make-meta-col-container acc combine-method container-type datatype-meta)
        document-container (make-document-col-container acc combine-method container-type datatype-document)
        term-index-container (make-term-index-col-container acc combine-method container-type datatype-term-idx)
        ]
    (.add ^List (:term-pos-containers acc) term-pos-container)
    (when metas-container
      (.add ^List (:meta-containers acc) metas-container))
    (.add ^List (:document-containers acc) document-container)
    (.add ^List (:term-index-containers acc) term-index-container)))
  

(defn process-line [token-lookup-table line-split-fn text-tokenizer-fn
                    datatype-document
                    datatype-term-pos
                    datatype-meta
                    datatype-term-idx
                    container-type
                    compacting-document-intervall
                    combine-method
                    acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        token-indices (map (partial tools/put-retrieve-token! token-lookup-table) tokens)
        index-count (count tokens)
        meta-list (:meta-list acc)
        index-list (:index-list acc)
        term-list (:term-list acc)
        acc (update acc :n-docs-parsed inc)]


    (when meta
      (.add ^List meta-list meta))
    (.add ^List index-list index-count)
    (.addAll ^List term-list token-indices)


    (if (zero? (rem ^long (dt/ecount index-list) ^long compacting-document-intervall))
      (do
        (tools/debug :compact (* ^long compacting-document-intervall ^long (dt/ecount (:term-pos-containers acc))))
        (update-acc! acc combine-method container-type datatype-term-pos datatype-meta datatype-document datatype-term-idx)
        (assoc acc
               :meta-list (dt/make-list datatype-meta)
               :term-list (dt/make-list datatype-term-idx)
               :index-list (dt/make-list datatype-document)))
      acc)))


(defn make-column [name container-list combine-method container-type datatype]
    (let [data
        (case combine-method
          :concat-buffers (dt/concat-buffers datatype container-list)
          :coalesce-blocks! 
          (let [col-size  (func/reduce-+ (map count container-list))
                container (dt/make-container container-type datatype col-size)]
            
            (dt/coalesce-blocks! container container-list)) 
          )
        
        ]
    (col-impl/construct-column [] data  {:name name})))


(defn ->tidy-text
  "Reads, parses and tokenizes a text file or a TMD dataset 
   into a seq of tech.v3.dataset in the tidy-text format,
   so one word per row. 
   It does the parsing and conversion strictly line based, so it should work for large documents.

   Initial tests show that each byte of text size need 1.5 byte on average
   So a 8 GB text file can be sucessfully loaded when having at least 12 GB.

   `lines-source` Either a buffered reader or a TMD dadaset
   `line-seq-fn`  A function which return a lazy-list of lines , given the `lines-source`
   `line-split-fn` A fn which should seperate a single line of input in text and `other`
                   Supposed to return a seq of size 2, where the first is the 'text' of the line and `meta` can be 
                   anything non-nil (map, vector, scalar). It's value will be returned in column `meta` and is supposed 
                   to be further processed later. `meta` can be nil always,  so no column `meta` will be created 

   `text-tokenizer-fn` A function which will be called for any `text` as obtained by `line-split-fn`
                       It should split the text by word boundaries and return the obtained tokens as a seq of strings.
                       It can do any text normalisation desired.
   
   Optional `options` are: 
   `skip-lines`                      0           Lines to skip at beginning
   `max-lines`                       MAX_INT     max lines to return

   The following can be used to optimize the heap usage for larger texts.
   Can be tune depensing on how may documents, howmnay words per documen and how many 
   tokens overall are in te text corpus  
   
   `container-type`                 :jvm-heap          If the resulting table is created on heap (:jvm-heap ) of off heap (:native-heap)
                                                       :native-heap works (much) better on large texts
   `datatype-document`              :int16             Datatype of :document column (:int16 or :int32)
   `datatype-term-pos`              :int16             Datatype of :term-pos column (:int16 or :int32)
   `datatype-meta`                  :object            Datatype of :meta column (anything, need to match what `line-split-fn` returns as 'meta')
   `datatype-term-idx`              :int16             Datatype of :term-idx column (:int16 or :int32)
   `compacting-document-intervall`  10000              After how many lines the data is written into a contious block
   `combine-method`                 :coalesce-blocks!  Which metghod to use to combine blocks (:coalesce-blocks! or :concat-buffers)

   

   Function returns a map of :datasets and :token-lookup-table
   
   :datasets is a seq of TMD datasets each having 4 columns which represent
   the input text in the tidy-text format.

   :document    The 'document/line' a token is comming from
   :term-idx    The token/word (as int) , which is present as well in the token->int look up table returned
   :term-pos    The position of the token in the document
   :meta        The meta values if return by `line-split-fn`
                   
   Assuming that the `text-tokenizer-fn` does no text normalisation, the table is a exact representation of the input text-

                  
                   


   "
  [lines-source 
   line-seq-fn
   line-split-fn
   line-tokenizer-fn


   & {:keys [skip-lines max-lines
             container-type
             datatype-document
             datatype-term-pos
             datatype-meta
             datatype-term-idx
             
             compacting-document-intervall
             combine-method]
             
      :or {skip-lines  0
           max-lines Integer/MAX_VALUE
           container-type    :jvm-heap
           datatype-document :int16
           datatype-term-pos :int16
           datatype-meta    :object
           datatype-term-idx :int16 
           compacting-document-intervall 10000
           combine-method :coalesce-blocks!}}]

  (let [_ (tools/debug :parse)
        token-lookup-table (Object2IntLinkedOpenHashMap. 10000)
        _ (.put token-lookup-table "" 0)

        acc
        (tools/process-file
         lines-source
         line-seq-fn
         (partial process-line token-lookup-table line-split-fn line-tokenizer-fn
                  datatype-document
                  datatype-term-pos
                  datatype-meta
                  datatype-term-idx
                  container-type
                  compacting-document-intervall
                  combine-method)
         {:n-docs-parsed 0
          :meta-list (dt/make-list datatype-meta)
          :term-list (dt/make-list datatype-term-idx)
          :index-list (dt/make-list datatype-document)
          :term-pos-containers (hf/mut-list)
          :meta-containers (hf/mut-list)
          :document-containers (hf/mut-list)
          :term-index-containers (hf/mut-list)}
         max-lines skip-lines)


        _ (update-acc!  acc combine-method container-type datatype-term-pos datatype-meta datatype-document datatype-term-idx)

        acc (assoc acc
                   :meta-list (dt/make-list datatype-meta)
                   :term-list (dt/make-list datatype-term-idx)
                   :index-list (dt/make-list datatype-document))




        col-term-index (make-column :term-idx (:term-index-containers acc) combine-method container-type datatype-term-idx)
        col-term-pos (make-column :term-pos (:term-pos-containers acc) combine-method container-type datatype-term-pos)
        col-document (make-column :document (:document-containers acc) combine-method container-type datatype-document)
        
        col-meta (when (seq (:meta-containers acc))
                   (make-column :meta (:meta-containers acc) combine-method container-type datatype-meta))

        _ (tools/debug :measure-term-index (mm/measure col-term-index))
        _ (tools/debug :measure-term-pos (mm/measure col-term-pos))
        _ (tools/debug :measure-document-idx (mm/measure col-document))
        _ (tools/debug :measure-metas (mm/measure col-meta))

        ds
        (ds/new-dataset
         [col-term-index col-term-pos col-document])

        ds-withmetas
        (if col-meta
          (ds/add-column ds col-meta)
          ds)]

    (tools/debug :token-lookup-table (count token-lookup-table))
    (tools/debug :measure-token-lookup-table (mm/measure token-lookup-table))


    (tools/debug :measure-col-term-index (mm/measure col-term-index))
    (tools/debug :measure-col-term-pos (mm/measure col-term-pos))
    (tools/debug :measure-col-document-idx (mm/measure col-document))
    (tools/debug :measure-col-metas (mm/measure col-meta))
    (tools/debug :measure-ds (mm/measure ds-withmetas))


    {:datasets [ds-withmetas]
     :token-lookup-table  (Object2IntMaps/unmodifiable token-lookup-table)}))




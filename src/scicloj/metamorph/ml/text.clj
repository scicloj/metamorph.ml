(ns scicloj.metamorph.ml.text
  (:require
   [clj-memory-meter.core :as mm]
   [criterium.core :as crit]
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
   [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2IntOpenHashMap LongOpenHashSet]
   [it.unimi.dsi.fastutil.objects Object2IntMaps Object2IntOpenHashMap Object2LongOpenHashMap]
   [java.util List]
   [org.mapdb DBMaker]))

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


(defn create-token->idf-map [tidy-text]
  (tools/debug :create-token->idf-map)
  (let [N
        (->
         (reductions/aggregate
          {:count (reductions/count-distinct :document)}
          tidy-text)
         :count
         first)]


    (tools/debug :N N)
    (reductions/group-by-column-agg
     :token-idx
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


(defn- tf-idf-reducer [token-idx->idf-map container-type]
  (reductions/reducer
   [:document :token-idx]
   (fn [] {:token-counts  (Long2IntOpenHashMap.)
           :token-counter 0
           :document nil})
   (fn [acc ^long document ^long token-idx]
     ;(tools/debug :reduce-tfidf document)
     (when (and (zero? (rem document 10000)) (zero? ^long (:token-counter acc)))
       (tools/debug :reduce-tfidf document))
  
     (.addTo ^Long2IntOpenHashMap  (:token-counts acc) token-idx 1)
     {:token-counts (:token-counts acc)
      :token-counter (inc ^long (:token-counter acc))
      :document document})
   (fn [acc-1 acc-2]
     (throw (Exception. "merge should not get called")))
  
   (fn [{:keys [token-counts ^long token-counter ^long document]}]

     ;(tools/debug :finalize-tfidf document)
     (when (zero? (rem  document 10000))
       (tools/debug :finalize-tfidf document))
     
     
     
     (let [token->tfidf-fn
           (fn [[^long token-index ^long count]]
             (let [tf ^float (/ count token-counter)]
               {token-index
                {:tf tf
                 :tfidf ^float (* ^float tf ^float (get token-idx->idf-map token-index))}}))
  
           tf-idfs
           (apply hf/merge (lznc/map token->tfidf-fn token-counts))]
  
  
       {:token-idx (dt/make-container container-type :int32  (hf/keys tf-idfs))
        :token-count (dt/make-container container-type  :int32 (hf/vals token-counts))
        :tf (dt/make-container container-type :float32 (hf/mapv :tf (hf/vals tf-idfs)))
        :tfidf (dt/make-container container-type :float32 (hf/mapv :tfidf (hf/vals tf-idfs)))}))))
  


(defn ->tfidf [tidy-text &  {:keys [container-type] 
                             :or {container-type :jvm-heap}}]

  (let [idfs (create-token->idf-map tidy-text)

        _ (tools/debug :token-idx->idf-map)
        token-idx->idf-map
        (Long2FloatLinkedOpenHashMap. (-> idfs :token-idx dt/->long-array)
                                      (-> idfs :idf dt/->float-array))


        _ (tools/debug :measure-token-idx->idf-map 
              (mm/measure token-idx->idf-map))

        _ (tools/debug :tfidf-data)
        tfidf-data
        (reductions/group-by-column-agg
         :document
         {:tfidf-cols (tf-idf-reducer token-idx->idf-map container-type)}
         tidy-text)]

    (ds/new-dataset
     [(>document-col container-type :int32 tfidf-data)
      (->column :tfidf container-type :float32 tfidf-data :tfidf)
      (->column :tf container-type :float32 tfidf-data :tf)
      (->column :token-idx container-type :int32 tfidf-data :token-idx)
      (->column :token-count container-type :int16 tfidf-data :token-count)])))


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
     [(:token-counts-list acc)
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
     [(range-2 (- ^int n-docs-parsed ^int (count (:token-counts-list acc)))
               (count (:token-counts-list acc)))

      (:token-counts-list acc)])))


(defn- make-token-pos-col-container [acc combine-method container-type datatype]
  (make-col-container
   range
   combine-method
   container-type
   datatype
   [(:token-counts-list acc)]))


(defn- make-token-index-col-container-fast [acc combine-method container-type datatype]
  (dt/make-container container-type datatype (:token-indices-list acc)))


(defn- update-acc! [acc combine-method container-type datatype-token-pos datatype-meta datatype-document datatype-token-idx]
  (let [token-pos-container (make-token-pos-col-container acc combine-method container-type datatype-token-pos)
        metas-container (make-meta-col-container acc combine-method container-type datatype-meta)
        document-container (make-document-col-container acc combine-method container-type datatype-document)
        token-index-container (make-token-index-col-container-fast acc combine-method container-type datatype-token-idx)
        ]
    (.add ^List (:token-pos-containers acc) token-pos-container)
    (when metas-container
      (.add ^List (:meta-containers acc) metas-container))
    (.add ^List (:document-containers acc) document-container)
    (.add ^List (:token-index-containers acc) token-index-container)))
  

(defn process-line [token-lookup-table line-split-fn text-tokenizer-fn
                    datatype-document
                    datatype-token-pos
                    datatype-meta
                    datatype-token-idx
                    container-type
                    compacting-document-intervall
                    combine-method
                    acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        token-indices (map (partial tools/get-put-token token-lookup-table) tokens)
        token-count (count tokens)
        meta-list (:meta-list acc)
        token-counts-list (:token-counts-list acc)
        token-indices-list (:token-indices-list acc)
        acc (update acc :n-docs-parsed inc)]


    (when meta
      (.add ^List meta-list meta))
    (.add ^List token-counts-list token-count)
    (.addAll ^List token-indices-list token-indices)


    ;(println :n-docs-parsed (:n-docs-parsed acc))
    (if (zero? (rem ^long (:n-docs-parsed acc) ^long compacting-document-intervall))
      (do
        (tools/debug :compact (* ^long compacting-document-intervall ^long (dt/ecount (:token-pos-containers acc))))
        (update-acc! acc combine-method container-type datatype-token-pos datatype-meta datatype-document datatype-token-idx)
        (assoc acc
               :meta-list (dt/make-list datatype-meta)
               :token-indices-list (dt/make-list datatype-token-idx)
               :token-counts-list (dt/make-list datatype-document)))
      acc)))

(defn fill-lookup-from-line [token-lookup-table
                             line-split-fn text-tokenizer-fn
                             datatype-document
                             datatype-token-pos
                             datatype-meta
                             datatype-token-idx
                             container-type
                             compacting-document-intervall
                             combine-method
                             acc line]

  (let [[text _] (line-split-fn line)
        tokens (text-tokenizer-fn text)]
    (run! (partial tools/get-put-token token-lookup-table) tokens)

    (println :n-docs-parsed (:n-docs-parsed acc))
    (when (zero? (rem ^long (:n-docs-parsed acc) ^long compacting-document-intervall))
      (tools/debug :fill-look-up (:n-docs-parsed acc)))
    )
  
    (update acc :n-docs-parsed inc))


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
   `datatype-token-pos`              :int16             Datatype of :token-pos column (:int16 or :int32)
   `datatype-meta`                  :object            Datatype of :meta column (anything, need to match what `line-split-fn` returns as 'meta')
   `datatype-token-idx`              :int16             Datatype of :token-idx column (:int16 or :int32)
   `compacting-document-intervall`  10000              After how many lines the data is written into a contious block
   `combine-method`                 :coalesce-blocks!  Which metghod to use to combine blocks (:coalesce-blocks! or :concat-buffers)

   

   Function returns a map of :datasets and :token-lookup-table
   
   :datasets is a seq of TMD datasets each having 4 columns which represent
   the input text in the tidy-text format.

   :document    The 'document/line' a token is comming from
   :token-idx    The token/word (as int) , which is present as well in the token->int look up table returned
   :token-pos    The position of the token in the document
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
             datatype-token-pos
             datatype-meta
             datatype-token-idx
             compacting-document-intervall
             combine-method
             token->index-map 
             ]
             
      :or {skip-lines  0
           max-lines Integer/MAX_VALUE
           container-type    :jvm-heap
           datatype-document :int16
           datatype-token-pos :int16
           datatype-meta    :object
           datatype-token-idx :int16 
           compacting-document-intervall 10000
           combine-method :coalesce-blocks!
           token->index-map  (Object2IntOpenHashMap. 10000)
           }}]

  (let [_ (tools/debug :parse)
        

        

        _ (.put token->index-map "" 0)

        process-line-fn process-line
        ;fill-lookup-from-line

        acc
        (tools/process-file
         lines-source
         line-seq-fn
         (partial process-line-fn token->index-map line-split-fn line-tokenizer-fn
                  datatype-document
                  datatype-token-pos
                  datatype-meta
                  datatype-token-idx
                  container-type
                  compacting-document-intervall
                  combine-method)
         {:n-docs-parsed 0
          :meta-list (dt/make-list datatype-meta)
          :token-indices-list (dt/make-list datatype-token-idx)
          :token-counts-list (dt/make-list datatype-document)
          :token-pos-containers (hf/mut-list)
          :meta-containers (hf/mut-list)
          :document-containers (hf/mut-list)
          :token-index-containers (hf/mut-list)}
         max-lines skip-lines)


        _ (update-acc!  acc combine-method container-type
                        datatype-token-pos
                        datatype-meta
                        datatype-document
                        datatype-token-idx)

        acc (assoc acc
                   :meta-list (dt/make-list datatype-meta)
                   :token-indices-list (dt/make-list datatype-token-idx)
                   :token-counts-list (dt/make-list datatype-document))




        col-token-index (make-column :token-idx (:token-index-containers acc) combine-method container-type datatype-token-idx)
        col-token-pos (make-column :token-pos (:token-pos-containers acc) combine-method container-type datatype-token-pos)
        col-document (make-column :document (:document-containers acc) combine-method container-type datatype-document)

        col-meta (when (seq (:meta-containers acc))
                   (make-column :meta (:meta-containers acc) combine-method container-type datatype-meta))

        _ (tools/debug :measure-token-index (mm/measure col-token-index))
        _ (tools/debug :measure-token-pos (mm/measure col-token-pos))
        _ (tools/debug :measure-document-idx (mm/measure col-document))
        _ (tools/debug :measure-metas (mm/measure col-meta))

        ds
        (ds/new-dataset
         [col-token-index col-token-pos col-document])

        ds-withmetas
        (if col-meta
          (ds/add-column ds col-meta)
          ds)]

    (tools/debug :count--token->index-map (count token->index-map))
    (tools/debug :measure--token->index-map (mm/measure  token->index-map))


    (tools/debug :measure-col-token-index (mm/measure col-token-index))
    (tools/debug :measure-col-token-pos (mm/measure col-token-pos))
    (tools/debug :measure-col-document-idx (mm/measure col-document))
    (tools/debug :measure-col-metas (mm/measure col-meta))
    (tools/debug :measure-ds (mm/measure ds-withmetas))


    {:datasets [ds-withmetas]
     :token-lookup-table  token->index-map
     ;(Object2IntMaps/unmodifiable token-lookup-table)
     }))


(comment
  (require '[criterium.core :as crit])
  (import [org.mapdb DBMaker]
          [it.unimi.dsi.fastutil.objects Object2IntOpenHashMap
           Object2LongOpenHashMap])



  ;;  Execution time mean : 3.202750 ms
  (let [heap-db
        (.. DBMaker
            heapDB
            make)
        heap-db-map (.. heap-db (hashMap "map") createOrOpen)]
    (crit/quick-bench
     (run!
      #(.put heap-db-map (str "hello" %) 1)
      (range 10000))))

  ;;  Execution time mean :10.405044 ms 
  (let [memory-db
        (.. DBMaker
            memoryDB
            make)
        heap-db-map (.. memory-db (hashMap "map") createOrOpen)]
    (crit/quick-bench
     (do
       (run!
        #(.put heap-db-map (str "hello" %) 1)
        (range 10000))
       (run! 
        (fn [_] (.size heap-db-map))
        (range 10000)
        ))
     ))

  ;;Execution time mean : 722.928142 ms
  (let [heap-db
        (.. DBMaker
            tempFileDB
            make)
        heap-db-map (.. heap-db (hashMap "map") createOrOpen)]
    (crit/quick-bench
     (run!
      #(.put heap-db-map (str "hello" %) 1)
      (range 10000))))



  ;;Execution time mean : 1.449999 ms
  (let [o2l-map (Object2LongOpenHashMap.)]
    (crit/quick-bench
     (run!
      #(.put o2l-map (str "hello" %) 1)
      (range 10000)))
    (println
     (mm/measure o2l-map))
    )

  ;;Execution time mean : 1.221315 ms
  ;;; 667.2 KiB
  (let [o2i-map (Object2IntOpenHashMap.)]
    (crit/quick-bench
     (run!
      #(.put o2i-map (str "hello" %) 1)
      (range 10000)))
    (println
     (mm/measure o2i-map))
    )
  ;;Execution time mean : 1.901854 ms
  (let [hf-map (hf/mut-map)]
    (crit/quick-bench
     (run!
      #(.put hf-map (str "hello" %) 1)
      (range 10000)))
    (println
     (mm/measure hf-map)))
  

  (def heap-db-map
    (.. DBMaker
        heapDB
        make
        (hashMap "map") 
        counterEnable
        createOrOpen))
  (run!
   #(.put heap-db-map (str "hello" %) 1)
   (range 10000))

  ;;Execution time mean : 319.890385 ms
  (crit/quick-bench
   (run!
    (fn [_] (.size heap-db-map))
    (range 1000)))

  )

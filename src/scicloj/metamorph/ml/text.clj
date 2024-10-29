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
   [it.unimi.dsi.fastutil.longs Long2FloatLinkedOpenHashMap Long2IntOpenHashMap LongOpenHashSet]
   ))

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


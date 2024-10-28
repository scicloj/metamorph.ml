(ns scicloj.metamorph.ml.text2
  (:require
   [clj-memory-meter.core :as mm]
   [ham-fisted.api :as hf]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dt]
   [tech.v3.datatype.functional :as func]
   [scicloj.metamorph.ml.tools :as tools]
   
)
  (:import
   [java.util List]))


(defn- make-col-container--concat-buffers [map-fn  container-type res-dataype  datas]
  (let [col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?) ; prevents 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         )
        ]

    (dt/concat-buffers res-dataype col-datas)
    ))


(defn- make-col-container--coalesce-blocks! [map-fn  container-type res-dataype  datas]
  (let [col-datas
        (->>
         (apply dt/emap map-fn nil datas)
         (remove empty?) ; prevents 'buffer type class clojure.lang.PersistentList$EmptyList is not convertible to buffer'
         )
        col-size (or (func/reduce-+ (map count col-datas)) 0)
        container (dt/make-container container-type res-dataype col-size)
        ]

    (when (not-empty col-datas)
      (dt/coalesce-blocks! container col-datas)
      )
    container
    ))

(defn- make-col-container [map-fn combine-method container-type res-dataype  datas]
  (case combine-method
    :coalesce-blocks! (make-col-container--coalesce-blocks! map-fn container-type res-dataype datas)
    :concat-buffers (make-col-container--concat-buffers map-fn container-type res-dataype datas)
    )
  )

(defn- make-metas-col-container [index-and-lable-lists combine-method container-type datatype]
  (make-col-container
   (fn [index meta]
     (dt/const-reader meta index))
   combine-method
   (if (= :object datatype)
     :jvm-heap
     container-type)
   datatype
   [(:index-list index-and-lable-lists)
    (:meta-list index-and-lable-lists)]))

(defn range-2 [a b]
  (range a (+ a b)))


(defn- make-document-col-container [index-and-lable-lists combine-method container-type datatype]
  (let [n-docs-parsed (:n-docs-parsed index-and-lable-lists)]
    (make-col-container
     (fn [idx count]
       (dt/const-reader idx count))
     combine-method
     container-type
     datatype
     [(range-2 (- n-docs-parsed (count (:index-list index-and-lable-lists)))
               (count (:index-list index-and-lable-lists)))

      (:index-list index-and-lable-lists)])))


(defn- make-term-pos-col-container [index-and-lable-lists combine-method container-type datatype]
  (make-col-container
   range
   combine-method
   container-type
   datatype
   [(:index-list index-and-lable-lists)]))

(defn- make-term-index-col-container [index-and-lable-lists combine-method container-type datatype]
  (make-col-container
   combine-method
   identity
   container-type
   datatype
   [(:term-list index-and-lable-lists)]))


(defn- update-acc! [acc combine-method container-type datatype-term-pos datatype-metas datatype-document datatype-term-idx]
  ;(debug "before copy")
  #_{:clj-kondo/ignore [:unresolved-symbol]}
  (let [term-pos-container (make-term-pos-col-container acc combine-method container-type datatype-term-pos)
        metas-container (make-metas-col-container acc combine-method container-type datatype-metas)
        document-container (make-document-col-container acc combine-method container-type datatype-document)
        term-index-container (dt/make-container container-type datatype-term-idx (:term-list acc))]
        (.add ^List (:term-pos-containers acc) term-pos-container)
        (.add ^List (:metas-containers acc) metas-container)
        (.add ^List (:document-containers acc) document-container)
        (.add ^List (:term-index-containers acc) term-index-container))
  ;(debug "after copy")
  )


(defn process-line [token->long line-split-fn text-tokenizer-fn
                    datatype-document
                    datatype-term-pos
                    datatype-metas
                    datatype-term-idx
                    container-type
                    compacting-document-intervall
                    combine-method
                    acc line]
  (let [[text meta] (line-split-fn line)
        tokens (text-tokenizer-fn text)

        token-indices (map (partial tools/put-retrieve-token! token->long) tokens)
        index-count (count tokens)
        meta-list (:meta-list acc)
        index-list (:index-list acc)
        term-list (:term-list acc)
        acc (update acc :n-docs-parsed inc)]


    (.add ^List meta-list meta)
    (.add ^List index-list index-count)
    (.addAll ^List term-list token-indices)


    (if (zero? (rem (dt/ecount index-list) compacting-document-intervall))
      (do
        (tools/debug :compact (* compacting-document-intervall (dt/ecount (:term-pos-containers acc))))
        (update-acc! acc combine-method container-type datatype-term-pos datatype-metas datatype-document datatype-term-idx)
        (assoc acc
               :meta-list (dt/make-list datatype-metas)
               :term-list (dt/make-list datatype-term-idx)
               :index-list (dt/make-list datatype-document)))
      acc
      )))



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
             container-type
             compacting-document-intervall 
             combine-method]
      :or {skip-lines  0
           datatype-document :int32
           datatype-term-pos :int16
           datatype-metas    :int8
           datatype-term-idx :int32
           container-type    :jvm-heap
           max-lines Integer/MAX_VALUE
           compacting-document-intervall 10000
           combine-method :coalesce-blocks!}}]

  (let [_ (tools/debug :parse)
        token->long (hf/mut-map [["" 0]])
        acc
        (tools/process-file reader
                      (partial process-line  token->long line-split-fn text-tokenizer-fn
                               datatype-document
                               datatype-term-pos
                               datatype-metas
                               datatype-term-idx
                               container-type
                               compacting-document-intervall
                               combine-method
                               )
                      {:n-docs-parsed 0
                       :meta-list (dt/make-list datatype-metas)
                       :term-list (dt/make-list datatype-term-idx)
                       :index-list (dt/make-list datatype-document)
                       :term-pos-containers (hf/mut-list)
                       :metas-containers (hf/mut-list)
                       :document-containers (hf/mut-list)
                       :term-index-containers (hf/mut-list)}
                      max-lines skip-lines)


        _ (update-acc!  acc combine-method container-type datatype-term-pos datatype-metas datatype-document datatype-term-idx)

        acc (assoc acc
               :meta-list (dt/make-list datatype-metas)
               :term-list (dt/make-list datatype-term-idx)
               :index-list (dt/make-list datatype-document))


        
        col-term-index (ds/new-column :term-idx (dt/concat-buffers (:term-index-containers acc))  {} [])
        col-term-pos (ds/new-column :term-pos  (dt/concat-buffers (:term-pos-containers acc)) {} [])
        col-document (ds/new-column :document  (dt/concat-buffers (:document-containers acc)) {} [])
        col-meta (ds/new-column :meta (dt/concat-buffers (:metas-containers acc)) {} [])

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



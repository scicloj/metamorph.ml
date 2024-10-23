(ns scicloj.metamorph.text-perf
   (:require
    [clj-memory-meter.core :as mm]
    [clojure.data.csv :as csv]
    [clojure.java.io :as io]
    [clojure.string :as str]
    [ham-fisted.api :as hf]
    [scicloj.metamorph.ml.text :as text]
    [tablecloth.api :as tc]
    [tech.v3.dataset.reductions :as reductions]
    [tech.v3.dataset.string-table :as st]
    [tech.v3.datatype :as dt]
    [tech.v3.datatype.functional :as func]
    [tech.v3.datatype.mmap :as mmap] 
    ;[tech.v3.datatype.mmap-writer :as mmap-writer]
    [tech.v3.datatype.mmap-writer :as mmap-writer]))



 (defn load-reviews []
   (-> (text/->tidy-text
        (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
        (fn [line] [line
                    (rand-int 6)])
        #(str/split % #" ")
        (st/make-string-table)
        :max-lines 10000000
        :skip-lines 1
        :container-type :native-heap
        :datatype-document :int32
        :datatype-term-pos :int16
        :datatype-term-idx :int32
        :datatype-metas    :byte)))



 (def df
   (->
    (:dataset (load-reviews))
    (tc/drop-columns [:term-pos])))



; (mm/measure (-> df :document .data))
; (mm/measure (-> df :meta .data))
; (mm/measure (-> df :term-idx .data))

(println)
 (println :df-measures
          (mm/measure df))
 

 (println)
 (println :shape (tc/shape df))

 (println :col-classes
          (map
           
           #(hash-map
             :name %1
             :class (-> %2 .data class))
           (tc/column-names df)
           (tc/columns df))
          )
 
 (println :col-datatypes
          (map
           (fn [name col]
             [name (-> col meta :datatype)])
           (tc/column-names df)
           (tc/columns df)))

(System/exit 0)

(def tfidf (text/->tfidf df :container-type :jvm-heap))
(println)

(println :measure-tfidf-ds (mm/measure tfidf))
(println :measure-tfidf-tfidf (mm/measure (:tfidf tfidf)))
(println :measure-tfidf-termcount (mm/measure (:term-count tfidf)))
(println :measure-tfidf-document (mm/measure (:document tfidf)))
(println :measure-tfidf-term-idx (mm/measure (:term-idx tfidf)))

;; (println 
;;  (-> tfidf :tfidf .data) "\n"
;;  (-> tfidf :term-count .data) "\n"
;;  (-> tfidf :document .data) "\n"
;;  (-> tfidf :term-idx .data) "\n")
 
 
 
(println :col-datatypes-tfidf
         (map
          (fn [name col]
            [name (-> col meta :datatype)])
          (tc/column-names tfidf)
          (tc/columns tfidf)))

(println tfidf)


 (comment
   (def N
     (->
      (reductions/aggregate
       {:count (reductions/count-distinct :document)}
       df)
      :count
      first))
   

   df
   


   (def term-idf-map 
     (reductions/group-by-column-agg
      :term-idx 
      {:idf
       (reductions/reducer :document
                           (fn [] (hf/mut-set))
                           (fn [acc ^long document]
                             (.add acc document)
                             acc)
                           (fn [uniq-documents-1 uniq-documents-2]
                             (hf/add-all! uniq-documents-1 uniq-documents-2))
                           (fn [uniq-documents] (Math/log10 (/ N (count uniq-documents)))))}
      df))
   
   (reductions/group-by-column-agg
    :document
    {:tf
     (reductions/reducer :term-idx
                         (fn [] (hf/mut-list))
                         (fn [acc ^long term-idx]
                           (.add acc term-idx)
                           acc)
                         (fn [term-idx-1 term-idx-2]
                           (hf/add-all! term-idx-1 term-idx-2))
                         (fn [term-idx] 
                           (let [ freqs (hf/frequencies term-idx)
                                 n-term (count term-idx)
                                 term-counts (hf/vals freqs)]
                             {:term-idx term-idx
                              :term-count term-counts
                              :tf (seq (func// term-counts (float n-term)))}
                             )
                           ))}
    df)

   )



 
 
 







;; 14G of RAM needed
;; 23:42:33.0046  -  :parse10000
;; 20000
;; ...
;; ...
;; 3720000
;;   (311) 
;; 23:47:44.0875  -  :count-index-nad-labels 2  (0) 
;; 23:47:45.0178  -  :make-document-col-container  (8) 
;; 23:47:53.0534  -  :make-term-pos-col-container  (38) 
;; 23:48:32.0170  -  :make-metas-col-container  (10) 
;; 23:48:52.0096  -  :measure-term-index-st 4.5 GiB  (0) 
;; 23:48:52.0097  -  :measure-term-pos 2.2 GiB  (0) 
;; 23:48:52.0097  -  :measure-document-idx 4.5 GiB  (0) 
;; 23:48:52.0098  -  :measure-metas 1.1 GiB  (0) 
;; 23:48:52.0472  -  :string-table-count 1201891227  (0) 
;; 23:48:55.0307  -  :measure-term-index-string-table 4.8 GiB  (0) 
;; 23:48:52.0481  -  :measure-col-term-index 4.5 GiB  (0) 
;; 23:48:52.0482  -  :measure-col-term-pos 2.2 GiB  (0) 
;; 23:48:52.0483  -  :measure-col-document-idx 4.5 GiB  (0) 
;; 23:48:52.0484  -  :measure-col-metas 1.1 GiB  (2) 
;; 23:48:55.0310  -  :measure-ds 12.4 GiB



 ; --------------------------
 (comment

   (defn- parse-review-line [line]
     (let [splitted (first
                     (csv/read-csv line))]
       [(first splitted)
        (dec (Integer/parseInt (second splitted)))]))

   (def ds-and-st

     (text/->tidy-text
      (io/reader
       ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
       (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
       ;(io/reader "test/data/reviews.csv")
      parse-review-line
      #(str/split % #" ")
      :max-lines 5
      :skip-lines 0))



   (def text
     (-> (:dataset ds-and-st)
         (tc/rename-columns {:meta :label})))
   )
         





(require '[tech.v3.datatype.mmap.larray])
(mmap/set-mmap-impl! tech.v3.datatype.mmap.larray/mmap-file)


(var-get #'mmap/mmap-fn*)

;;dd if=/dev/zero of=zeros_10G.bin bs=100000 count=100000

;;dd if=/dev/zero of=zeros_3G.bin bs=100000 count=30000

(def mm
  (-> 
   (mmap/mmap-file 
                   "xxx"
                   {:mmap-mode :read-write})
   (tech.v3.datatype.native-buffer/set-native-datatype  
    :float32
    ;int8 -> quite some things fail now, alreday printing. others give wrong results, (take 10 x) gives empty list
    )
   ))


(dt/get-value mm 100)
(dt/set-value! mm 100 20)


(count mm)

(take 100 mm)




;;=> #array-buffer<float32>[1000]
;;   [30.00, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50, 20.50...]



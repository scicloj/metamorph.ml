(ns scicloj.metamorph.text-perf
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [scicloj.metamorph.ml.text :as text]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.datatype :as dt]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.functional :as fun]
            [criterium.core :as criterium]
            [criterium.core :as criterim]
            [tech.v3.datatype.functional :as func]
            [tech.v3.dataset.reductions :as reductions]
            [ham-fisted.api :as hf]))



(defn load-reviews []
  (-> (text/->tidy-text
       (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       :max-lines 1000
       :skip-lines 1
       :datatype-document :int32
       :datatype-term-pos :int32
       :datatype-metas    :int8)))



(def df
  (:dataset (load-reviews)))


(println)
(println :meta
         (-> df :meta .data class))
(println :shape (tc/shape df))
(println :col-datatypes
         (map
          (fn [name col]
            [name (-> col meta :datatype)])
          (tc/column-names df)
          (tc/columns df)))



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
         (tc/rename-columns {:meta :label}))))
         
         


(->tfidf text term->idf-map)


(comment
  ;; new idf






  (require '[tech.v3.datatype.argops :as argops]
           '[tech.v3.dataset.reductions :as reductions])


  (defn idf [text]
    (let [N
          (tc/row-count
           (tc/unique-by text :document))]
      (-> text

          (tc/unique-by [:term-idx :document])
          (tc/group-by  [:term-idx])
          (tc/aggregate #(Math/log10 (/ N (tc/row-count %))))
          (tc/rename-columns {"summary" :idf}))))


  (defn idf [text]
    (let [N
          (tc/row-count
           (tc/unique-by text :document))]
      (-> text

          (tc/unique-by [:term-idx :document])
          (reductions/group-by-column-agg :term-idx
                                          {:idf})
          (tc/group-by  [:term-idx])
          (tc/aggregate #(Math/log10 (/ N (tc/row-count %))))
          (tc/rename-columns {"summary" :idf}))))



  (defn idf-is [text terms-grouped-by-document]

    (let [N (dt/ecount terms-grouped-by-document)

          row-indexes-grouped-by-term-idx
          (ds/group-by-column->indexes text :term-idx)

          term-idx
          (dt/emap
           first
           :int32
           row-indexes-grouped-by-term-idx)

          idf
          (->>
           (dt/emap
            (fn [[term-idx  rows]]
      ;(println term-idx rows)
              (dt/ecount
               (distinct
                (ds/select-rows (:document text) rows))))
            :int32
            row-indexes-grouped-by-term-idx)
           (fun// N)
           (fun/log10))]
      (ds/new-dataset [(ds/new-column :term-idx term-idx {} [])
                       (ds/new-column :idf (seq idf) {} [])])))



  (defn tf [text terms-grouped-by-document]

    (let [n-terms-in-doc
          (dt/coalesce-blocks!
           (dt/make-container :int32 (ds/row-count text))
           (dt/emap
            (fn [[document terms]]
              (dt/const-reader (dt/ecount terms) (dt/ecount terms)))
            :int32
            terms-grouped-by-document))


          term-fq
          (dt/coalesce-blocks!
           (dt/make-container :int32 (ds/row-count text))
           (dt/emap
            (fn [[document terms]]
              (let [term-idxs (ds/select-rows (:term-idx text)
                                              terms)
                    f (frequencies term-idxs)]
                (dt/emap f :int32 term-idxs)))
            :int32
            terms-grouped-by-document))]

      (-> text
          (ds/add-column (ds/new-column :term-fq term-fq {} []))
          (ds/add-column (ds/new-column :n-terms-in-doc n-terms-in-doc {} []))
          (ds/select-columns [:document :term-idx :term-fq :n-terms-in-doc])
          (tc/unique-by [:document :term-idx])
          (tc/add-column :tf (fn [ds]
                               (fun// (:term-fq ds)
                                      (double-array (:n-terms-in-doc ds))))))))



  (time
   (let [terms-grouped-by-document
         (ds/group-by-column->indexes df :document)
         idf (idf-is df terms-grouped-by-document)
         tf (tf df terms-grouped-by-document)
         joined
         (tc/full-join idf tf :term-idx)
         tfidf (fun/* (:tf joined) (:idf joined))]
     (ds/add-column joined (ds/new-column :tfidf tfidf {} []))))


  (time
   (text/->term-frequency df)))



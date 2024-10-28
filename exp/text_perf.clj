(ns text-perf
  (:require
   [clj-memory-meter.core :as mm]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.metamorph.ml.text :as text]
   [scicloj.metamorph.ml.text2 :as text2]
   [tablecloth.api :as tc]
   [ham-fisted.set :as hf-set]))



(defn load-reviews [tidy-text-fn max-lines]
  (-> (tidy-text-fn
       (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       :max-lines max-lines
       :skip-lines 1
       :container-type :native-heap
       :datatype-document :int32
       :datatype-term-pos :int16
       :datatype-term-idx :int32
       :datatype-metas    :byte
       :compacting-document-intervall 10000)))


(defn tidy [& opts]
  (let [opts (first opts)
        tidy-text-fn
        (case (:tidy-algo opts)
          1 text/->tidy-text
          2 text2/->tidy-text)
        df
        (->
         (first (:datasets (load-reviews 
                            tidy-text-fn
                            (or (:max-lines opts) Integer/MAX_VALUE))))
         )]

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
              (tc/columns df)))

    (println :col-datatypes
             (map
              (fn [name col]
                [name (-> col meta :datatype)])
              (tc/column-names df)
              (tc/columns df)))
    (println df)))


(defn tfidf [& opts]

  (println :opts opts)
  (let [opts (first opts)
        tidy-text-fn
        (case (:tidy-algo opts)
          1 text/->tidy-text
          2 text2/->tidy-text)
        df
        (->
         (first (:datasets (load-reviews
                            tidy-text-fn
                            (or (:max-lines opts) Integer/MAX_VALUE))))
         (tc/drop-columns [:term-pos]))


        #_ (println :tidy-document-unique (-> df :document hf-set/unique count))
         
        _ (do
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
                      (tc/columns df)))

            (println :col-datatypes
                     (map
                      (fn [name col]
                        [name (-> col meta :datatype)])
                      (tc/column-names df)
                      (tc/columns df)))
            (println df))

        tfidf (text/->tfidf df :container-type :native-heap)

        _ (do
            (println)

            (println :measure-tfidf-ds (mm/measure tfidf))
            (println :measure-tfidf-tfidf (mm/measure (:tfidf tfidf)))
            (println :measure-tfidf-termcount (mm/measure (:term-count tfidf)))
            (println :measure-tfidf-document (mm/measure (:document tfidf)))
            (println :measure-tfidf-term-idx (mm/measure (:term-idx tfidf)))

            (println :col-datatypes-tfidf
                     (map
                      (fn [name col]
                        [name (-> col meta :datatype)])
                      (tc/column-names tfidf)
                      (tc/columns tfidf)))

            (println :tfidf-document-unique (-> tfidf :document hf-set/unique count))
            (println tfidf))]

  )
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
  



(ns text-perf
  (:require
   [clj-memory-meter.core :as mm]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [ham-fisted.api :as hf]
   [ham-fisted.set :as hf-set]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dt]))






(defn load-reviews [max-lines]
  (-> (text/->tidy-text
       (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
       line-seq
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       :max-lines max-lines
       :skip-lines 1

       :datatype-document :int32
       :datatype-token-pos :int16
       :datatype-token-idx :int32
       :datatype-meta    :byte
       ;:compacting-document-intervall 100000
       )))

(defn df->tidy [& opts]

  (let [opts (first opts)
        
        df
        (ds/->dataset "bigdata/repeatedAbstrcats_3.7m_.txt"
                      {:text-temp-dir "/tmp/xxx"
                       :file-type :tsv
                       :num-rows (or (:num-rows opts) Integer/MAX_VALUE)
                       :header-row? false})

        tidy-df
        (first (:datasets
                (-> (text/->tidy-text
                     df
                     (fn [df] (map str (-> df (get "column-0"))))
                     (fn [line] [line
                                 (rand-int 6)])
                     #(str/split % #" ")
                     :skip-lines 1
                     :datatype-document :int32
                     :datatype-token-pos :int16
                     :datatype-token-idx :int32
                     :datatype-meta    :byte))))]
    (println)
    (println :shape (tc/shape tidy-df))))


(defn tidy [& opts]
;  (def opts [{:max-lines 59999}])

  (let [opts (first opts)
        df
        (->
         (first (:datasets (load-reviews 
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
  ;(def opts [{:max-lines 10000}])
  (println :opts opts)
  (let [opts (first opts)
        df
        (->
         (first (:datasets (load-reviews
                            (or (:max-lines opts) Integer/MAX_VALUE))))
         (tc/drop-columns [:token-pos]))


        #_(println :tidy-document-unique (-> df :document hf-set/unique count))

        _ (do
            (println)
            (println :measure-col-token-index (mm/measure (:token-idx df)))
            (println :measure-col-token-pos (mm/measure (:token-pos df)))
            (println :measure-col-document-idx (mm/measure (:document df)))
            (println :measure-col-metas (mm/measure (:meta df)))


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

        tfidf (text/->tfidf df)

        _ (do
            (println)

            (println :measure-tfidf-ds (mm/measure tfidf))
            (println :measure-tfidf-tfidf (mm/measure (:tfidf tfidf)))
            (println :measure-tfidf-termcount (mm/measure (:token-count tfidf)))
            (println :measure-tfidf-document (mm/measure (:document tfidf)))
            (println :measure-tfidf-term-idx (mm/measure (:token-idx tfidf)))

            (println :col-datatypes-tfidf
                     (map
                      (fn [name col]
                        [name (-> col meta :datatype)])
                      (tc/column-names tfidf)
                      (tc/columns tfidf)))
            (println :col-classes-tfidf
                     (map

                      #(hash-map
                        :name %1
                        :class (-> %2 .data class))
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
  



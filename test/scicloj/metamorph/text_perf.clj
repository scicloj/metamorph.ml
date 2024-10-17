(ns scicloj.metamorph.text-perf
  (:require 
            [clojure.java.io :as io]
            [clojure.string :as str]
            
            [scicloj.metamorph.ml.text :as text]
            
            [tablecloth.api :as tc]
            
            
  
            ))

  



(defn load-reviews []
  (-> (text/->tidy-text
       (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       ;:max-lines 100
       :skip-lines 1
       :datatype-document :int32
       :datatype-term-pos :int32
       :datatype-metas    :int8)))



 (def df
      (:dataset (load-reviews)))

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

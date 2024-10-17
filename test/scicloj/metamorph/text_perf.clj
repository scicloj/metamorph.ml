(ns scicloj.metamorph.text-perf
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.text :as text]
            [tech.v3.dataset.string-table :as string-table]
            [tablecloth.api :as tc]
            [criterium.core :as criterim]
            [tech.v3.dataset.string-table :as st]
  
            [clj-memory-meter.core :as mm]))

  



(defn load-reviews []
  (-> (text/->tidy-text
       (io/reader "bigdata/repeatedAbstrcats_3.7m_.txt")
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       :skip-lines 1)))


(load-reviews)


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

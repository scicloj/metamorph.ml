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
       (io/reader "repeatedAbstrcats_3.7m_.txt")
       (fn [line] [line
                   (rand-int 6)])
       #(str/split % #" ")
       :max-lines 100000
       :skip-lines 1)))


(load-reviews)

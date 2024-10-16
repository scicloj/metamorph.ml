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

(defn- parse-review-line [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     (dec (Integer/parseInt (second splitted)))]))

(def r
  (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                        parse-review-line
                        #(str/split % #" ")
                        :skip-lines 1)))
  

(count (:string-table r))


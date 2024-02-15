(ns scicloj.metamorph.ml.toydata.ggplot
  (:require [tech.v3.dataset]
            [clojure.java.io :as io]))

;; datasets copied from the (MIT-licensed) R ggplot2 package version 3.4.1


(defn- ->resource-dataset [resource-path]
  (-> resource-path
      io/resource
      io/input-stream
      (tech.v3.dataset/->dataset
       {:file-type :csv
        :gzipped? true
        :key-fn keyword})))

(def diamonds
  (->resource-dataset "data/ggplot/diamonds.csv.gz"))

(def ecomonics
  (->resource-dataset "data/ggplot/ecomonics.csv.gz"))

(def ecomonics_long
  (->resource-dataset "data/ggplot/ecomonics_long.csv.gz"))

(def faithfuld
  (->resource-dataset "data/ggplot/faithfuld.csv.gz"))

(def luv_colours
  (->resource-dataset "data/ggplot/luv_colours.csv.gz"))

(def midwest
  (->resource-dataset "data/ggplot/midwest.csv.gz"))

(def mpg
  (->resource-dataset "data/ggplot/mpg.csv.gz"))

(def msleep
  (->resource-dataset "data/ggplot/msleep.csv.gz"))

(def presidential
  (->resource-dataset "data/ggplot/presidential.csv.gz"))

(def seals
  (->resource-dataset "data/ggplot/seals.csv.gz"))

(def txhousing
  (->resource-dataset "data/ggplot/txhousing.csv.gz"))

(ns scicloj.metamorph.linear-regression-test
  (:require  [clojure.test :as t]
             [scicloj.metamorph.ml :as ml]
             [scicloj.metamorph.ml.toydata :as data]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.column-filters :as cf]
             [tech.v3.dataset.tensor :as dtt]
             [tech.v3.datatype :as dt]
             [tech.v3.tensor :as tensor]
             [tech.v3.dataset.modelling :as ds-mod]
             [tablecloth.api :as tc]
             [tech.v3.datatype.functional :as func])
  (:import [org.apache.commons.math3.stat.regression OLSMultipleLinearRegression]))


(def ds
  (->
   (data/mtcars-ds)
   (ds/drop-columns [:model])
   (ds-mod/set-inference-target :mpg)))

(def model (ml/train ds {:model-type :metamorph.ml/ols}))

(ml/glance model)
;; => _unnamed [1 3]:
;;    |       :totss | :adj.r.squared |         :rss |
;;    |-------------:|---------------:|-------------:|
;;    | 1126.0471875 |     0.80664232 | 147.49443002 |
(ml/tidy model)
;; => _unnamed [11 3]:
;;    | :term |   :estimate |  :std.error |
;;    |-------|------------:|------------:|
;;    |  :mpg | 12.30337416 | 18.71788443 |
;;    |  :cyl | -0.11144048 |  1.04502336 |
;;    | :disp |  0.01333524 |  0.01785750 |
;;    |   :hp | -0.02148212 |  0.02176858 |
;;    | :drat |  0.78711097 |  1.63537307 |
;;    |   :wt | -3.71530393 |  1.89441430 |
;;    | :qsec |  0.82104075 |  0.73084480 |
;;    |   :vs |  0.31776281 |  2.10450861 |
;;    |   :am |  2.52022689 |  2.05665055 |
;;    | :gear |  0.65541302 |  1.49325996 |
;;    | :carb | -0.19941925 |  0.82875250 |

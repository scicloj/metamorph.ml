(ns scicloj.metamorph.ml.verify
  (:require
   [clojure.test :refer [is]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss] ;; [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype :as dtype]
   [tech.v3.datatype.functional :as dfn]))


(def target-colname "petal_width")


(def regression-iris* (delay
                        (-> (ds/->dataset "test/data/iris.csv")
                            (ds/remove-column "species")
                            (ds-mod/set-inference-target "petal_width"))))


(def classification-titanic* (delay
                               (-> (ds/->dataset "test/data/titanic.csv")
                                   (ds/remove-column "Name")
                                   ;;We have to have a lookup map for the column in order to
                                   ;;do classification on the column.
                                   (ds/update-column "Survived"
                                                     (fn [col]
                                                       (let [val-map {0 :drowned
                                                                      1 :survived}]
                                                         (dtype/emap val-map :keyword col))))
                                   (ds/categorical->number cf/categorical)
                                   (ds-mod/set-inference-target "Survived"))))


(defn basic-regression
  ([options-map max-avg-loss]

   (let [split (ds-mod/train-test-split @regression-iris* options-map)
         target-colname (first (ds/column-names (cf/target (:test-ds split))))
         train-fn (fn []
                    (let [
                          fitted-model (ml/train (:train-ds split) options-map)
                          predictions (ml/predict (:test-ds split) fitted-model)]
                      (loss/mae ((:test-ds split) target-colname) (predictions target-colname))))
         avg-mae
         (->>
          (repeatedly 5 train-fn)
          (dfn/mean))]

     (is (< avg-mae max-avg-loss))))
  ([options-map]
   (basic-regression options-map 0.5)))

(defn basic-classification
  ([options-map max-avg-loss]

   (let [split (ds-mod/train-test-split @classification-titanic* options-map)]
        target-colname (first (ds/column-names (cf/target (:test-ds split))))
         train-fn (fn []
                    (let [
                          fitted-model (ml/train (:train-ds split) options-map)
                          predictions (ml/predict (:test-ds split) fitted-model)]
                      (loss/mae ((:test-ds split) target-colname) (predictions target-colname))))
         avg-mae
         (->>
          (repeatedly 5 train-fn)
          (dfn/mean))

     (is (< avg-mae max-avg-loss))))
  ([options-map]
   (basic-classification options-map 0.5)))

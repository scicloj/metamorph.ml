(ns scicloj.metamorph.ml.verify
  {:no-doc true}
  (:require
   [clojure.test :refer [is]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss] ;; [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.datatype.functional :as dfn]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]))




(def regression-iris*
  "Lazy-loaded Iris dataset configured for regression testing.

  A delayed dataset with species removed and `:petal-width` set as the inference
  target. Used by `basic-regression` for model verification.

  Deref with `@regression-iris*` to load."
  (delay
    (->
     (rdatasets/datasets-iris)
     (ds/remove-column :species)
     (ds-mod/set-inference-target :petal-width))))


(def classification-titanic*
  "Lazy-loaded Titanic dataset configured for classification testing.

  A delayed dataset with missing values dropped, categorical columns converted
  to numeric, and `:survived` set as the inference target. Used by
  `basic-classification` for model verification.

  Deref with `@classification-titanic*` to load."
  (delay
    (->
     (rdatasets/carData-TitanicSurvival)

     (ds/remove-column :rownames)
        ;;We have to have a lookup map for the column in order to
        ;;do classification on the column.
     (ds/drop-missing)
     (ds/categorical->number cf/categorical)
     (ds-mod/set-inference-target :survived))))



(defn basic-regression
  "Verifies a regression model performs within acceptable error bounds.

  `options-map` - Model options map (must include `:model-type`)
  `max-avg-loss` - Maximum acceptable average MAE (default: 0.5)

  Trains the model 5 times on Iris data (predicting petal width), calculates
  average MAE on test set, and asserts it's below `max-avg-loss`.

  Returns a clojure.test assertion result.

  Used for testing model implementations and ensuring regression models work correctly."
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
  "Verifies a classification model performs within acceptable error bounds.

  `options-map` - Model options map (must include `:model-type`)
  `max-avg-loss` - Maximum acceptable average MAE (default: 0.5)

  Trains the model 5 times on Titanic survival data, calculates average MAE on
  test set, and asserts it's below `max-avg-loss`.

  Returns a clojure.test assertion result.

  Used for testing model implementations and ensuring classification models work correctly."
  ([options-map max-avg-loss]

   (let [split (ds-mod/train-test-split @classification-titanic* options-map)
         target-colname (first (ds/column-names (cf/target (:test-ds split))))
         train-fn (fn []
                    (let [fitted-model (ml/train (:train-ds split) options-map)
                          predictions (ml/predict (:test-ds split) fitted-model)]
                      (loss/mae ((:test-ds split) target-colname) (predictions target-colname))))
         avg-mae
         (->>
          (repeatedly 5 train-fn)
          (dfn/mean))
         ]
     (is (< avg-mae max-avg-loss))))
  ([options-map]
   (basic-classification options-map 0.5)))


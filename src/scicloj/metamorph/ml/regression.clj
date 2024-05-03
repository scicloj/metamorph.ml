(ns scicloj.metamorph.ml.regression
  (:require [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
            [tablecloth.api :as tc]
            [tech.v3.datatype :as dt]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.metamorph.ml.toydata :as data]
            [tech.v3.dataset.tensor :as dtt])
  (:import [org.apache.commons.math3.stat.regression OLSMultipleLinearRegression]))



(defn- tidy-ols [model]
   (ds/->dataset
    {:term
     (concat (:target-columns model)
             (:feature-columns model))

     :estimate
     (.estimateRegressionParameters (:model-data model))
     :std.error
     (.estimateRegressionParametersStandardErrors (:model-data model))}))


(defn- augment-fn [model data]
  (-> data
      (tc/add-column :.residuals (.estimateResiduals (:model-data model)))))


(defn- glance-ols [model]

  {:totss
     (.calculateTotalSumOfSquares (:model-data model))
     :adj.r.squared
     (.calculateAdjustedRSquared (:model-data model))
     :rss
     (.calculateResidualSumOfSquares (:model-data model))

     ;; (.estimateRegressandVariance (:model-data model)) ; TODO what this ?
     :sigma
     (.estimateErrorVariance (:model-data model))})

(defn- train-ols [feature-ds target-ds options]
  (let [
        values
        (->
         (tc/append target-ds feature-ds)

         (dtt/dataset->tensor)
         (dt/->double-array))
        ds
        (->
         (data/mtcars-ds)
         (ds/drop-columns [:model])
         (ds-mod/set-inference-target :mpg))
        shape
        (ds/shape ds)


        ols (OLSMultipleLinearRegression.)
        _
        (.newSampleData ols values
                        (second shape)
                        (dec (first shape)))]
    ols))

(defn- predict-ols [feature-ds thawed-model {:keys [options model-data target-categorical-maps] :as model}])


(ml/define-model! :metamorph.ml/ols
  train-ols
  predict-ols
  {
   :tidy-fn tidy-ols
   :glance-fn glance-ols
   :augment-fn augment-fn})

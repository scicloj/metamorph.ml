(ns scicloj.metamorph.ml.regression
  "Regression models for continuous target prediction.

   This namespace provides implementations of various regression algorithms with
   a consistent metamorph.ml training and prediction interface. Models support
   statistical output formats (tidy, glance, augment) for analysis and diagnostics.

   Available Models:

   **OLS (Ordinary Least Squares)**
   - `:metamorph.ml/ols`: Apache Commons Math implementation (Java-based)
   - `:fastmath/ols`: FastMath implementation (pure Clojure)
   Solves for regression coefficients β in: y = Xβ + ε
   Assumes linear relationships and homoscedastic errors.

   **GLM (Generalized Linear Model)**
   - `:fastmath/glm`: FastMath GLM implementation
   Extends linear regression to non-normal distributions and non-linear relationships
   via link functions and variance models.

   **Baseline Model**
   - `:metamorph.ml/dummy-regressor`: Predicts mean of training target
   Useful sanity check - models should outperform this baseline.

   Model Output Functions:

   - **:tidy-fn**: Extracts model coefficients with statistics
     Returns dataset with :term, :estimate, :std.error, :statistic, :p.value
   - **:glance-fn**: Extracts model-level diagnostics
     Returns dataset with :r.squared, :adj.r.squared, :rss, :aic, :bic, etc.
   - **:augment-fn**: Adds model predictions and residuals to data
     Returns augmented dataset with :.fitted and :.resid columns

   Example Usage (in metamorph pipeline):
   (ml/train
     data
     {:model-type :fastmath/ols
      :target-columns [:price]
      :feature-columns [:sqft :bedrooms]})

   Model Diagnostics:
   (glance model)      ; Overall model metrics
   (tidy model)        ; Coefficient table
   (augment model ds)  ; Predicted values and residuals

   See also: `scicloj.metamorph.ml.r-model-matrix` for formula-based feature engineering"
  (:require
   [fastmath.ml.regression :as fm-reg]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dt]
   [tech.v3.dataset.tensor :as dtt]
   [fastmath.core :as m]
   [fastmath.vector :as v]

   [tech.v3.dataset.column-filters :as cf]
   [tablecloth.column.api :as tcc])
  (:import [org.apache.commons.math3.stat.regression OLSMultipleLinearRegression]
           [fastmath.java Array]))


(defn- tidy-fm-ols [model]

  (let [coeff (:coefficients (:model-data model))]
    (ds/->dataset
     {
      :term
      (concat (:target-columns model)
              (:feature-columns model))

      :statistic (map :t-value coeff)
      :estimate (map :estimate coeff)
      :p.value (map :p-value coeff)
      :std.error (map :stderr coeff)})))




(defn- augment-fm-ols [model data]
  (let [residuals (-> model :model-data :residuals)]
    (-> data
        (tc/add-columns {:.resid (:raw residuals)
                         :.fitted (:fitted (:model-data model))}))))



(defn- glance-fm-ols [model]
  (let [{:as model-data :keys [ll]} (:model-data model)]
    (ds/->dataset
     {:mse (:msreg model-data)
      :log-lik (:log-likelihood ll)
      :aic (:aic ll)
      :bic (:bic ll)
      :totss (:tss model-data)
      :n (:observations model-data)
      :adj.r.squared (:adjusted-r-squared model-data)
      :r.squared (:r-squared model-data)
      :rss (:rss model-data)
      :statistic (:f-statistic model-data)
      :p.value (:p-value model-data)
      :df (-> model-data :df :model)
      :df.residual (-> model-data :df :residual)})))



(defn- train-fm [fn feature-ds target-ds options]
  (let [clean-options
        (-> options
            (dissoc :model-type)
            (assoc :names (vec (tc/column-names feature-ds))))
        xss
        (->
         feature-ds
         ds/rowvecs)

        ys
        (-> target-ds
            cf/target
            first
            second)
        model (fn ys xss clean-options)]

    (assoc model
           :analysis
           (-> model :analysis deref))))


(defn predict-fm [feature-ds thawed-model model]
  (let [prediction (map
                    (:model-data model)
                    (-> feature-ds ds/rowvecs))
        target-column-name (-> model :target-columns first)]

    (ds/new-dataset [target-column-name
                     (ds/new-column target-column-name
                                    prediction
                                    {:column-type :prediction})])))

  
(defn- train-fm-ols [feature-ds target-ds options]
  (train-fm fm-reg/lm feature-ds target-ds options))


(defn- train-fm-glm [feature-ds target-ds options]
  (train-fm fm-reg/glm feature-ds target-ds options))


(defn- tidy-ols [model]
  (let [ols (->  model :model-data :ols)]
    (ds/->dataset
     {:term
      (concat (:target-columns ols)
              (:feature-columns ols))

      :estimate
      (.estimateRegressionParameters ols)
      :std.error
      (.estimateRegressionParametersStandardErrors ols)})))


(defn- augment-fn [model
                   data]
  (let [ols (->  model :model-data :ols)]
    (-> data
        (tc/add-column :.resid (.estimateResiduals ols)))))


(defn- glance-ols [model]


  (let [ols (->  model :model-data :ols)]
    (ds/->dataset
     {
      :totss
      (.calculateTotalSumOfSquares ols)
      :adj.r.squared
      (.calculateAdjustedRSquared ols)
      :rss
      (.calculateResidualSumOfSquares ols)

      ;; (.estimateRegressandVariance (:model-data model)) ; TODO what this ?
      :sigma
      (.estimateErrorVariance ols)})))

(defn- train-ols[feature-ds target-ds options]
  (let [

        ds (tc/append target-ds feature-ds)
        values
        (->
         ds

         (dtt/dataset->tensor)
         (dt/->double-array))
        shape
        (ds/shape ds)


        ols (OLSMultipleLinearRegression.)

        _
        (.newSampleData ols values
                        (second shape)
                        (dec (first shape)))
        beta (.estimateRegressionParameters ols)]
    {:ols ols
     :beta beta}))


(defn- single-predict [model xs]
  (let [beta (-> model :model-data :beta)
        intercept (Array/aget beta 0)
        coefficients (vec (rest beta))]

    (m/+ (v/dot coefficients xs) intercept)))


(defn- predict-ols [feature-ds thawed-model model]



  (let [xs
        (->
         feature-ds
         ds/rowvecs)

        xs
        (into-array (map double-array xs))


        predicted-values
        (map
         #(single-predict model %)
         xs)
        target-column-name (-> model :target-columns first)]
    (ds/new-dataset [target-column-name
                     (ds/new-column target-column-name
                                    predicted-values
                                    {:column-type :prediction})])))



(ml/define-model! :metamorph.ml/ols
  train-ols
  predict-ols
  {
   :tidy-fn tidy-ols
   :glance-fn glance-ols
   :augment-fn augment-fn})

(ml/define-model! :fastmath/ols
  train-fm-ols
  predict-fm
  {
   :tidy-fn tidy-fm-ols
   :glance-fn glance-fm-ols
   :augment-fn augment-fm-ols})

(ml/define-model! :fastmath/glm
  train-fm-glm
  predict-fm
  {})


(ml/define-model! :metamorph.ml/dummy-regressor
  (fn [feature-ds target-ds options]
    (let [mean-y (tcc/mean
                  (-> target-ds
                      cf/target
                      tc/columns
                      first
                      ))]
      {:mean mean-y}
      )
    
    )
  (fn [feature-ds thawed-model model]
    (-> (tc/dataset {(first (:target-columns model))
                     (repeat (tc/row-count feature-ds) (-> model :model-data :mean))
                     })
        (ds/assoc-metadata ( :target-columns model) :column-type :prediction)
        
        )
    )
  {})




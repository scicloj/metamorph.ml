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
   ```
   (ml/train
     data
     {:model-type :fastmath/ols})
      
   ```

   Model Diagnostics:
   ```
   (ml/glance model)        ; Overall model metrics
   (ml/tidy model)          ; Coefficient table
   (ml/augment model data)  ; Predicted values and residuals
  ```   

   See also: [[scicloj.metamorph.ml.r-model-matrix]] for R-formula-based feature engineering"
  (:require
   [fastmath.core :as m]
   [fastmath.ml.regression :as fm-reg]
   [fastmath.random :as r]
   [fastmath.vector :as v]
   [scicloj.metamorph.ml :as ml]
   [scicloj.plotje.api :as pj]
   [scicloj.plotje.impl.render]
   [tablecloth.api :as tc]
   [tablecloth.column.api :as tcc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.tensor :as dtt]
   [tech.v3.datatype :as dt]
   [fastmath.stats :as stats])
  (:import [fastmath.java Array]
           [org.apache.commons.math3.stat.regression OLSMultipleLinearRegression]))


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
                         :.std.resid (-> model :model-data :analysis :residuals :standardized)
                         :.fitted (:fitted (:model-data model))
                         :.cooksd (-> model :model-data :analysis :influence :cooks-distance)
                         :.hat (-> model :model-data :analysis :laverage :hat)
                         }))))


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


(defn- predict-fm [feature-ds thawed-model model]
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
         ds/rowvecs)        ;; r-vs-f-dataset (-> dataset
                    ;;             (tc/add-column :fitted fitted)
                    ;;             (tc/add-column :residual residuals))


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

(defn ppoints
  "Mimics R's function `ppoints`.
   Used for making the  :residual-q-q plot
   "
  [n]
  (let [m-range n
        a (if (<= n 10) 3/8 1/2)
        denom (+ n 1 (- (* 2 a)))]
    (mapv
     (fn [m]
       (double (/ (- (inc m) a) denom)))
     (range m-range))))

;((y2-y1)/(x2-x1)) * (x - x1) + y1
(defn make-line-eq-fn [xs ys]
  (fn [x]
    (+
     (*
      (/ (- (second ys) (first ys))
         (- (second xs) (first xs)))
      (- x (first xs)))
     (first ys)))) 


(defn- residual-vs-fitted-pose [augmented-ds options]
  (->
   augmented-ds
  
   (pj/lay-point  :.fitted :.resid)
   (pj/lay-smooth (merge {:color "red"} options))
   (pj/options {:title "Fitted vs Residuals"
                :x-label "Fitted values"
                :y-label "Residuals"})
   (pj/lay-rule-h {:y-intercept 0 :color "grey" :alpha 0.2})
   (pj/lay-text :.fitted :.resid
                {:text :row-number
                 :color "red"                                 ;:nudge-x 1
                 :data (-> augmented-ds
                           (tc/add-column :.abs-resid #(tcc/abs (:.resid %)))
                           (tc/order-by :.abs-resid :desc)
                           (tc/head 3))}))
  )

(defn- residual-qq-pose [augmented-ds]
  (let [num-rows (tc/row-count augmented-ds)

        
        probs [0.25 0.75]
        normal-quantiles
        (map
         #(r/icdf r/default-normal %)
         (ppoints num-rows))

        xs
        (map
         #(r/icdf r/default-normal %)
         probs)

        ys
        (stats/quantiles
         (-> augmented-ds :.std.resid)
         probs)

        line-fn (make-line-eq-fn xs ys)

        start-x (tcc/reduce-min normal-quantiles)
        start-y (line-fn start-x)
        end-x (tcc/reduce-max normal-quantiles)
        end-y (line-fn end-x)
        
        qq-dataset 
        (-> (tc/dataset  {:x (sort normal-quantiles)
                          :y (sort (:.std.resid augmented-ds))})
            (tc/add-column :y-diff
                           (fn [ds]
                             (map
                              (fn [x y]
                                (abs (- y (line-fn x))))
                              (:x ds)
                              (:y ds)))))
        ]



    (->
     qq-dataset
     (pj/lay-point :x :y)
     (pj/lay-line {:data [{:x start-x
                           :y start-y}
                          {:x end-x
                           :y end-y}]})

     (pj/options {:title "Q-Q Residuals"
                  :x-label "Theoretical Quantiles"
                  :y-label "Standardised residuals"}))))

(defn- cooks-distance-pose [augmented-ds]
  (-> augmented-ds
      (pj/lay-lollipop :row-number :.cooksd)
      (pj/options {:title "Cooks distance"

                   :x-label "Obs. number"
                   :y-label "Cook's distance"})))


(defn- scale-location-pose [augmented-ds options]
  (->
   augmented-ds
   (tc/add-column :.sqrt-abs-resid (tcc/sqrt (tcc/abs (:.std.resid augmented-ds))))
   (pj/lay-point :.fitted :.sqrt-abs-resid)
   (pj/lay-smooth (merge {:color "red"} options))
   (pj/options {:title "Scale Location"
                :x-label "Fitted values"
                :y-label "(sqrt (abs standardised-residuals))"})))

(defn- diagnostic-plots-ols-fm [model dataset options]
  (let [augmented-ds (-> (ml/augment model dataset)
                         (tc/add-column :row-number (map str (range (tc/row-count dataset)))))]
    (def augmented-ds augmented-ds)
    {:residual-vs-fitted (residual-vs-fitted-pose augmented-ds options)
     :residual-q-q (residual-qq-pose augmented-ds)
     :scale-location (scale-location-pose augmented-ds options)
     :cooks-distance (cooks-distance-pose augmented-ds)

     :residual-vs-leverage
     (-> augmented-ds
         (pj/lay-point  :.hat  :.std.resid)


         (pj/options {:title "Residual vs Leverage"
                      :x-label "Leverage"
                      :y-label "Standardised residuals"}))

     :cooks-d-vs-leverage*
     (-> augmented-ds
         (tc/add-column :leverage* (fn [ds]
                                     (map
                                      (fn [hat]
                                        (/ hat (- 1 hat)))
                                      (:.hat ds))))

         (pj/lay-point  :leverage* :.cooksd)
         (pj/options {:title "Residual vs Leverage h_ii / (1 - h_ii)"
                      :x-label "Leverage h_ii / (1 - h_ii)"
                      :y-label "Cook's distance"}))}))
  

   
  
    
      

  





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
  {:tidy-fn tidy-fm-ols
   :glance-fn glance-fm-ols
   :augment-fn augment-fm-ols
   :plot-fn diagnostic-plots-ols-fm})

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


 
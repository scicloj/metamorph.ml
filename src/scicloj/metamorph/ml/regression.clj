(ns scicloj.metamorph.ml.regression
  (:require
   [fastmath.ml.regression :as fm-reg]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.datatype :as dt]
   [tech.v3.dataset.tensor :as dtt]
   [fastmath.core :as m]
   [fastmath.vector :as v]

   [tech.v3.dataset.column-filters :as cf])
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







(defn- augment-fm-fn [model data]
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



(defn- train-fm-ols [feature-ds target-ds options]
  (let [clean-options
        (dissoc options :model-type)
        xss
        (->
         feature-ds
         ds/rowvecs)

        ys
        (-> target-ds
            cf/target
            first
            second)]

    (fm-reg/lm ys xss clean-options)))

(defn- predict-fm-ols [feature-ds thawed-model model]
  (let [prediction (map
                    (:model-data model)
                    (-> feature-ds ds/rowvecs))
        target-column-name (-> model :target-columns first)]

    (ds/new-dataset [target-column-name
                     (ds/new-column target-column-name
                                    prediction
                                    {:column-type :prediction})])))


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
  predict-fm-ols
  {
   :tidy-fn tidy-fm-ols
   :glance-fn glance-fm-ols
   :augment-fn augment-fm-fn})



(comment
  (def xss [[ 0.00000000e+00,  0.00000000e+00,  2.50000000e+01],
            [ 4.08163265e-01,  3.96924149e-01,  2.10849646e+01],
            [ 8.16326531e-01,  7.28634783e-01,  1.75031237e+01],
            [ 1.22448980e+00,  9.40632785e-01,  1.42544773e+01],
            [ 1.63265306e+00,  9.98087482e-01,  1.13390254e+01],
            [ 2.04081633e+00,  8.91559230e-01,  8.75676801e+00],
            [ 2.44897959e+00,  6.38550320e-01,  6.50770512e+00],
            [ 2.85714286e+00,  2.80629400e-01,  4.59183673e+00],
            [ 3.26530612e+00, -1.23398137e-01,  3.00916285e+00],
            [ 3.67346939e+00, -5.07151709e-01,  1.75968347e+00],
            [ 4.08163265e+00, -8.07581691e-01,  8.43398584e-01],
            [ 4.48979592e+00, -9.75328286e-01,  2.60308205e-01],
            [ 4.89795918e+00, -9.82831204e-01,  1.04123282e-02],
            [ 5.30612245e+00, -8.28857736e-01,  9.37109538e-02],
            [ 5.71428571e+00, -5.38705288e-01,  5.10204082e-01],
            [ 6.12244898e+00, -1.60045086e-01,  1.25989171e+00],
            [ 6.53061224e+00,  2.44910071e-01,  2.34277384e+00],
            [ 6.93877551e+00,  6.09627196e-01,  3.75885048e+00],
            [ 7.34693878e+00,  8.74184299e-01,  5.50812162e+00],
            [ 7.75510204e+00,  9.95115395e-01,  7.59058726e+00],
            [ 8.16326531e+00,  9.52551848e-01,  1.00062474e+01],
            [ 8.57142857e+00,  7.53486727e-01,  1.27551020e+01],
            [ 8.97959184e+00,  4.30625870e-01,  1.58371512e+01],
            [ 9.38775510e+00,  3.70144015e-02,  1.92523948e+01],
            [ 9.79591837e+00, -3.62678429e-01,  2.30008330e+01],
            [ 1.02040816e+01, -7.02784220e-01,  2.70824656e+01],
            [ 1.06122449e+01, -9.27424552e-01,  3.14972928e+01],
            [ 1.10204082e+01, -9.99691655e-01,  3.62453145e+01],
            [ 1.14285714e+01, -9.07712248e-01,  4.13265306e+01],
            [ 1.18367347e+01, -6.66598288e-01,  4.67409413e+01],
            [ 1.22448980e+01, -3.15964115e-01,  5.24885464e+01],
            [ 1.26530612e+01,  8.65820672e-02,  5.85693461e+01],
            [ 1.30612245e+01,  4.74903061e-01,  6.49833403e+01],
            [ 1.34693878e+01,  7.85198826e-01,  7.17305289e+01],
            [ 1.38775510e+01,  9.66488646e-01,  7.88109121e+01],
            [ 1.42857143e+01,  9.88987117e-01,  8.62244898e+01],
            [ 1.46938776e+01,  8.48997803e-01,  9.39712620e+01],
            [ 1.51020408e+01,  5.69520553e-01,  1.02051229e+02],
            [ 1.55102041e+01,  1.96472687e-01,  1.10464390e+02],
            [ 1.59183673e+01, -2.08855085e-01,  1.19210746e+02],
            [ 1.63265306e+01, -5.79868557e-01,  1.28290296e+02],
            [ 1.67346939e+01, -8.55611267e-01,  1.37703040e+02],
            [ 1.71428571e+01, -9.90779466e-01,  1.47448980e+02],
            [ 1.75510204e+01, -9.63165404e-01,  1.57528113e+02],
            [ 1.79591837e+01, -7.77305991e-01,  1.67940441e+02],
            [ 1.83673469e+01, -4.63737404e-01,  1.78685964e+02],
            [ 1.87755102e+01, -7.39780734e-02,  1.89764681e+02],
            [ 1.91836735e+01,  3.27935645e-01,  2.01176593e+02],
            [ 1.95918367e+01,  6.75970465e-01,  2.12921699e+02],
            [ 2.00000000e+01,  9.12945251e-01,  2.25000000e+02]])

  (def ys [ 4.48933467,  5.34231784,  5.81681264,  6.69809894,  5.94032168,
           6.65120265,  6.59592036,  6.07958661,  6.36186999,  6.7775208 ,
           6.6836138 ,  6.56551043,  7.17869566,  7.25158876,  8.04903067,
           7.84930564,  8.39547421,  8.25715349,  9.42432217,  8.43407786,
           9.70661341,  8.17868934, 10.44401028,  8.72852257,  9.32853046,
           9.70458442,  8.58017153,  9.78906458, 10.20670793,  9.20752194,
           9.26055989, 10.66574522,  9.89388631, 11.87907875, 10.65688892,
           11.09346875, 10.77674283, 10.11844363, 10.75104318, 10.19883781,
           10.68264587, 10.73214468, 10.13324477,  9.6745733 , 10.05157595,
           10.74993283, 10.4821192 , 10.38354809, 11.52270194, 11.19546067])



  (def xss
    (-> ds
        cf/feature
        ds/rowvecs))

  (def ys
    (-> ds
        cf/target
        first
        second)))

(ns examples
  (:require [metadoc.examples :refer :all]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.column-metric :as col-metric]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.ml.smile.classification]
            [scicloj.metamorph.ml.gridsearch :as gs]
            )
  )

(add-examples col-metric/regression-metric
              (example
                  "do regression and calculate RMSE"
                  (require 'scicloj.metamorph.ml)
                  (let [split
                        (->
                         (rdatasets/datasets-iris)
                         (ds/remove-columns [:rownames :species])
                         (ds-mod/set-inference-target [:petal-width])
                         (ds-mod/train-test-split))
                        model (ml/train (:train-ds split) {:model-type :fastmath/ols})
                        prediction (ml/predict (:test-ds split) model)]
                    (col-metric/regression-metric
                     (cf/target (:test-ds split))
                     prediction
                     :rmse)))
              )

(add-examples
 ml/optimize-hyperparameter
 (example-session
  "Simple call to optimize-hyperparameter using single :holdout split and single pipeline. (so we try only one configuration) "
  (require 'scicloj.metamorph.ml)
  (def result
    (let [iris
          (->
           (rdatasets/datasets-iris)
           (ds/remove-columns [:rownames])
           (ds-mod/set-inference-target [:species])
           (ds/categorical->number cf/categorical))

          split
          (tc/split->seq iris :holdout {:ratio [0.1 0.9]})

          pipe
          (mm/pipeline
           {:metamorph/id :model}
           (ml/model {:model-type :metamorph.ml/random-forest}))

          result
          (ml/optimize-hyperparameter
           [pipe]
           split
           {:metric :accuracy
            :metric-type :classification
            :loss-or-accuracy :accuracy})]
      result))
  (def train-accuracy (-> result first first :train-transform :metric))
  train-accuracy
  (def test-accuracy (-> result first first :test-transform :metric))
  test-accuracy)
 (example-session
  "Grid-search over hyperparameter space of :random-forrest model"
  (def iris
    (-> (rdatasets/datasets-iris)
        (tc/drop-columns [:rownames])
        (ds/categorical->number [:species])
        (ds-mod/set-inference-target :species)))

  (def iris-split (tc/split->seq iris))

  (def hyperparms-space (ml/hyperparameters :smile.classification/random-forest))

  (def search-space (take 10 (gs/sobol-gridsearch hyperparms-space)))

  (def model-options
    (map
     (fn [m] (assoc m :model-type :smile.classification/random-forest))
     search-space))

  (def pipeline-fns
    (map
     (fn [opts] (mm/pipeline
                 {:metamorph/id :model}
                 (ml/model opts)))
     model-options))

  (def results
    (ml/optimize-hyperparameter
     pipeline-fns
     iris-split
     {:metric :accuracy
      :loss-or-accuracy :accuracy
      :metric-type :classification}
     {:return-best-pipeline-only false}))

  (->
   (map
    (fn [res]
      (hash-map
       :options (-> res first :fit-ctx :model :options)
       :test-accuracy (-> res first :test-transform :metric)))
    results)
   (tc/dataset)
   (tc/map-column->columns :options)
   str)))

(add-examples
 ml/train
 (example
  "do regression and calculate RMSE"
  (require 'scicloj.metamorph.ml)
  (let [split
        (->
         (rdatasets/datasets-iris)
         (ds/remove-columns [:rownames :species])
         (ds-mod/set-inference-target [:petal-width])
         (ds-mod/train-test-split))
        model (ml/train (:train-ds split) {:model-type :fastmath/ols})
        prediction (ml/predict (:test-ds split) model)]
    (col-metric/regression-metric
     (cf/target (:test-ds split))
     prediction
     :rmse))))



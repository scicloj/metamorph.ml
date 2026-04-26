[![Clojars Project](https://img.shields.io/clojars/v/org.scicloj/metamorph.ml.svg)](https://clojars.org/org.scicloj/metamorph.ml)
[![CI](https://github.com/scicloj/metamorph.ml/actions/workflows/main.yml/badge.svg)](https://github.com/scicloj/metamorph.ml/actions/workflows/main.yml)
[![cljdoc badge](https://cljdoc.org/badge/org.scicloj/metamorph.ml)](https://cljdoc.org/d/org.scicloj/metamorph.ml)

![ml logo](https://github.com/scicloj/graphic-design/blob/live/icons/scicloj.ml.svg)

# metamorph.ml

Machine learning functions based on  [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) and [metamorph](https://github.com/scicloj/metamorph).


## Main idea

This library is based on the idea, that in machine learning model evaluations,
we often do not want to tune only the model and its hyper-parameters,
but the whole data transformation pipeline.

It unifies the often separated concerns of tuning of data pre-processing and
tuning of model hyper-parameters.

In a lot of areas of machine learning, certain aspects of the data
pre-processing needed to be tuned (by trying), as no clear-cut decisions exists.

One example could be the number of dimensions in PCA or the vocabulary size in a NLP ML model.
But it can be as well a "boolean" alternative, such as if stemming should be used or not.

This library allows exactly this, namely hyper-tune an arbitrary complex data transformation pipeline.


## Quick start

If you just want to see code, here it is:

Some libraries are needed for a complete test case, 
see the deps.edn file in alias "test".


```clojure
(ns read 
  (:require
    [scicloj.metamorph.ml.rdatasets :as rdatasets]
    [tech.v3.dataset :as ds]
    [scicloj.metamorph.ml :as ml]))

(require
 '[tech.v3.dataset :as ds]
 '[tech.v3.dataset.metamorph :as ds-mm]
 '[scicloj.metamorph.core :as mm]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.column-filters :as cf]
 '[tablecloth.api.split :as split]
 '[scicloj.metamorph.ml :as ml]
 '[scicloj.metamorph.ml.rdatasets :as rdatasets]
 '[scicloj.metamorph.ml.column-metric :as col-metric]
 '[scicloj.ml.smile.classification]
 )

;;  the data

(def  ds (->
          (rdatasets/datasets-iris)
          (ds/drop-columns [:rownames])))

;;  the (single, fixed) pipe-fn
(def pipe-fn
  (mm/pipeline
   ;; set inference target column
   (ds-mm/set-inference-target :species)
   ;; convert all categorical variables to numbers
   (ds-mm/categorical->number cf/categorical)
   ;; train a random forrest model or use it for prediction , depending on :metamorph/mode
   {:metamorph/id :model}
   (ml/model {:model-type :smile.classification/random-forest})))

;;  the simplest split, produces a seq of length one, a single split into train/test
(def  train-split-seq (split/split->seq ds :holdout))

;; we have only one pipe-fn here
(def  pipe-fn-seq [pipe-fn])

(def  evaluations
  (ml/optimize-hyperparameter pipe-fn-seq train-split-seq
                              {:metric :accuracy
                               :loss-or-accuracy :accuracy
                               :averaging :macro}))

;; we have only one result
(def best-fitted-context (-> evaluations first first :fit-ctx))
(def best-pipe-fn (-> evaluations first first :pipe-fn))

;; get training accuracy
(-> evaluations first first :train-transform :metric)
;; => 0.97

;;  simulate new data
(def  new-ds (ds/sample ds 10 {:seed 1234}))

;;  make prediction on new data

(def  predictions
  (->
   (mm/transform-pipe new-ds best-pipe-fn best-fitted-context)
   :metamorph/data
   (ds-mod/column-values->categorical :species)
   seq))
predictions
;;["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]

```



This library contains the basis functions for machine learning, arround:

* Train a model
* Predict on a trained model
* Register a trained model
* Find best model via hyperparameter optimisation

## model plugins

`metamorph.ml` is a ML framework,  containing just a single model type of a linear regression.
It is meant to be used together with other libraries, which contribute models:

|library| url | descriptions
|----------------------------|-----------------------------------------------|------------- 
|org.scicloj/scicloj.ml.smile |  https://github.com/scicloj/scicloj.ml.smile | most models of Java Smile package
|org.scicloj/scicloj.ml.tribuo| https://github.com/scicloj/scicloj.ml.tribuo | all regression/classification models of Java Tribuo package
|org.scicloj/sklearn-clj      | https://github.com/scicloj/sklearn-clj       | most regression/classification models of python-sklearn
|org.scicloj/scicloj.ml.xgboost|https://github.com/scicloj/scicloj.ml.xgboost | xgboost4J models

## Train a model

For training a model, we have function `train` , `predict` and `evaluate`:

```clojure
(def  ds (->
          (rdatasets/datasets-iris)
          (ds/drop-columns [:rownames])))

(def preprocessed-ds
  (-> ds
      (ds-mod/set-inference-target :species)
      (ds/categorical->number cf/categorical)))

(def split (ds-mod/train-test-split preprocessed-ds))

(def model (ml/train (:train-ds split) {:model-type :smile.classification/random-forest}))

(def prediction (ml/predict (:test-ds split) model))

(col-metric/classification-metric (:test-ds split) prediction :accuracy :macro)
;;=> 1.0

```


## Evaluate pipelines
Instead of running  `train` and `predict` as separate steps, 
the library offers as well to combine this in one step, and to `evaluate` a model or a pipleine.
Additonaly it does operate on an abstrcation of the preprocessing + modeling steps, teh so caleed pipeline.

The function `ml/optimize-hyperparameter` which takes a sequence of metamorph compliant pipeline-fn (= each pipeline is a series of steps to transform the raw data and a model step), does this.

It executes each pipeline first in `mode` :fit and then in `mode` transform, as specified by [metamorph](https://github.com/scicloj/metamorph)
which a pipeline step containing a model then translates into a
train/predict pattern including evaluation of the result.

The last step of the pipeline function should be a "model", so something which
can be trained / predicted.

This library does not care, what exactly is the last step.
It calls the whole pipeline in mode :fit and mode :transform and the model step is
supposed to do the right thing.

Here is a well behaving model function:
https://github.com/scicloj/metamorph.ml/blob/973606776cfabbe5a666a6cc0bab5a1833f044c8/src/scicloj/metamorph/ml.clj#L662
which calls `train` and `predict` accordingly. 

So for each pipeline-fn in the sequence given to `optimize-hyperparameter` one model
will be trained and evaluated.

It does this for each pipeline-fn given and each pipeline-fn gets evaluated
using each of the given test/train splits.

This can be used to implement various cross-validation strategies, just as
holdout, k-fold and others.

Each pipeline is typically a variation of a certain standard pipeline,
and encapsulates therefore individual machine learning trials with the goal to
find the best performing model automatically.

This is often called hyper-parameter tuning. But here we can do it for all 
options of the pipeline, and not only for the hyper-parameters of the model itself.

It is of course possible to just have a single pipeline function in the sequence.
Then a single model will be trained.

The different pipeline-fns are completely independent from each other and can
contain the same model, different models, or anything the developer wants.

This library does not contain itself functions to create metamorph
pipelines including their variations.

This can be done in various ways, from hand coding each pipeline  or having 
a pipeline creation function  over to using grid search libraries:
https://github.com/scicloj/metamorph.ml/blob/973606776cfabbe5a666a6cc0bab5a1833f044c8/src/scicloj/metamorph/ml/gridsearch.clj#L109

A simple ML pipeline looks like this:

```clojure
 (mm/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (ml-mm/model {:model-type :smile.classification/random-forest}))
```
It does preprocessing of data and contains the `modelling` step as well.


Pipelines can be created as well declarative based on maps, see here: 
https://scicloj.github.io/tablecloth/index.html#Declarative
https://github.com/scicloj/tablecloth/blob/pipelines/src/tablecloth/pipeline.clj

The train/test split sequence can as well be generated in any way. It need to
be a sequence of maps contain a "tech.ml.dataset"
at key :train and an other dataset  at key :test. These will be used to
train / predict and evaluate one pipeline.


`optimize-hyperparameter` returns then a sequence  of model evaluations.
It returns #pipeline-fn x  #cross-validation-splits evaluation results. This is as well the total number of models trained in total.

Each evaluation result contains a map with these keys:

key | explanation
------ | --------
 :fit-ctx  | the fitted pipeline context (including the trained model and the dataset at end of pipeline) after the pipeline was run in mode :fit 
 :transform-ctx | the  pipeline context (including the prediction dataset) after pipeline was run in mode :transform
 :scicloj.metamorph.ml/target-ds  | A dataset containing the ground truth
 :pipe-fn | the pipeline-fn doing the full transformation including train/predict
 :metric | the score for this model evaluation
 :mean | average score of this pipe-fn over the score of all train/test splits
 :min | min score of this pipe-fn over all train/test splits)
 :max | max score of this pipe-fn over all train/test splits
 :timing | Execution time in ms of fit and transform

This returned information is as well self-contained, as the pipeline-fn should manipulated exclusively the dataset and the available pipeline context.
This means, the :pipe-fn functions can be re-executed simply on new data.

This is due to the metamorph approach, which keeps all input/output of the pipeline inside of the context / dataset.

## Metamorph

The pipeline functions passed into `evaluate-pipelines` need to be metamorph
compliant as explained here:
https://github.com/scicloj/metamorph

'compliant' means simply to adhere to interact with a context map with certain 
standard keys.

A pipeline-fn is a composition of metamorph compliant data transform functions.

The following projects contain them, for both

- dataset manipulations

and

- train/prediction of models
 
and custom ones can be created easily.

- https://github.com/techascent/tech.ml.dataset
- https://github.com/scicloj/tablecloth
- https://github.com/scicloj/sklearn-clj

## Further tutorials / example

- [Introducion to supervised ML with metamorph.ml](https://scicloj.github.io/metamorph.ml/supervised_ml_intro.html)
- [metamorph examples](https://github.com/scicloj/metamorph-examples)
- [Tutorial of data science topics](https://scicloj.github.io/clojure-data-tutorials/)
- [noj documenttaion, partialy using metamorph.ml](https://github.com/scicloj/noj) 


 



<!--  LocalWords:  metamorph
 -->

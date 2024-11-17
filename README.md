[![Clojars Project](https://img.shields.io/clojars/v/org.scicloj/metamorph.ml.svg)](https://clojars.org/org.scicloj/metamorph.ml)
[![CI](https://github.com/scicloj/metamorph.ml/actions/workflows/main.yml/badge.svg)](https://github.com/scicloj/metamorph.ml/actions/workflows/main.yml)
[![cljdoc badge](https://cljdoc.org/badge/scicloj/metamorph.ml)](https://cljdoc.org/d/scicloj/metamorph.ml)

![ml logo](https://github.com/scicloj/graphic-design/blob/live/icons/scicloj.ml.svg)

# metamorph.ml

Machine learning functions for [tech.ml.dataset](https://github.com/techascent/tech.ml.dataset) and [metamorph](https://github.com/scicloj/metamorph).


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
(require
 '[tech.v3.dataset :as ds]
 '[tech.v3.dataset.metamorph :as ds-mm]
 '[scicloj.metamorph.core :as morph]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.column-filters :as cf]
 '[tablecloth.api.split :as split]
 '[scicloj.metamorph.ml :as ml]
 '[scicloj.metamorph.ml.loss :as loss]
 '[scicloj.ml.smile.classification]

 )

;;  the data
(def  ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

;;  the (single, fixed) pipe-fn
(def pipe-fn
  (morph/pipeline
   ;; set inference traget column
   (ds-mm/set-inference-target :species)
   ;; convert all categorical variables to numbers
   (ds-mm/categorical->number cf/categorical)
   ;; train or predict , depending on :mode
   {:metamorph/id :model}         
   (ml/model {:model-type :smile.classification/random-forest})))

;;  the simplest split, produces a seq of one, a single split into train/test
(def  train-split-seq (split/split->seq ds :holdout))

;; we have only one pipe-fn here
(def  pipe-fn-seq [pipe-fn])

(def  evaluations
  (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss))

;; we have only one result
(def best-fitted-context (-> evaluations first first :fit-ctx))
(def best-pipe-fn (-> evaluations first first :pipe-fn))

;; get training loss
(-> evaluations first first :train-transform :metric)
;; => 0.06000000000000005

;;  simulate new data
(def  new-ds (ds/sample ds 10 {:seed 1234} ))

;;  do prediction on new data
(def  predictions
  (->
   (best-pipe-fn
    (merge best-fitted-context
           {:metamorph/data new-ds
            :metamorph/mode :transform}))
   (:metamorph/data)
   (ds-mod/column-values->categorical :species)
   seq))

;;["versicolor" "versicolor" "virginica" "versicolor" "virginica" "setosa" "virginica" "virginica" "versicolor" "versicolor" ]


```



This library contains the basis functions for machine learning, arround:

* Train a model
* Predict on a trained model
* Register a trained model

## model plugins

`metamorph.ml` is a framework, only containing a single model `linear regression`.
It is ment to be used together with other libraries, which contribute models:
|library| url | descriptions
|----------------------------|-----------------------------------------------|------------- 
|org.scicloj/scicloj.ml.smile |  https://github.com/scicloj/scicloj.ml.smile | most models of Java Smile package
|org.scicloj/scicloj.ml.tribuo| https://github.com/scicloj/scicloj.ml.tribuo | all models of Java Tribuo package
|org.scicloj/sklearn-clj      | https://github.com/scicloj/sklearn-clj       | most models of python-sklearn
|org.scicloj/scicloj.ml.xgboost|https://github.com/scicloj/scicloj.ml.xgboost | xgboost4J models


## Evaluate pipelines
Instead of running  `train` and `predict` as separate steps, 
the library offers as well to combine this in one step, and to `evaluate` a model or a pipleine.

The function `evaluate-pipelines` which takes a sequence of metamorph compliant pipeline-fn (= each pipeline is a series of steps to transform the raw data and a model step), does this.

It executes each pipeline first in `mode` :fit and then in `mode` transform, as specified by [metamorph](https://github.com/scicloj/metamorph)
which a pipeline step containing a model should then translates into a
train/predict pattern including evaluation of the result.

The last step of the pipeline function should be a "model", so something which
can be trained / predicted.

This library does not care, what exactly is the last step.
It calls the whole pipeline in mode :fit and mode :transform and the model step is
supposed to do the right thing.

Here is a well behaving model function:
https://github.com/scicloj/metamorph.ml/blob/973606776cfabbe5a666a6cc0bab5a1833f044c8/src/scicloj/metamorph/ml.clj#L662
which calls `train` and `predict` accordingly. 

So for each pipeline-fn in the sequence given to `evaluate-models` one model
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

A simple pipeline looks like this:

```clojure
 (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (ml-mm/model {:model-type :smile.classification/random-forest}))
```

Pipelines can be created as well declarative based on maps, see here: 
https://scicloj.github.io/tablecloth/index.html#Declarative
https://github.com/scicloj/tablecloth/blob/pipelines/src/tablecloth/pipeline.clj

The train/test split sequence can as well be generated in any way. It need to
be a sequence of maps contain a "tech.ml.dataset"
at key :train and an other dataset  at key :test. These will be used to
train / predict and evaluate one pipeline.


`evaluates-pipelines` returns then a sequence  of model evaluations.
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

Several code examples for metamorph are available in this repository: [metamorph-examples](https://github.com/scicloj/metamorph-examples)

Here we have some [tutorials](https://scicloj.github.io/clojure-data-tutorials/) of data science topics , some use metamorph.ml.

[noj](https://github.com/scicloj/noj) is as well using metamorph.ml and has a cookbook [here](https://scicloj.github.io/noj/)

We have as well a (very unpolished) collection of [notebooks](https://scicloj.github.io/metamorph.ml/)  showcasing
only metamorph.ml functionality.


<!--  LocalWords:  metamorph
 -->

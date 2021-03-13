[![Clojars Project](https://img.shields.io/clojars/v/scicloj/metamorph.ml.svg)](https://clojars.org/scicloj/metamorph.ml)

# metamorph.ml

Evaluation functions for [metamorph](https://github.com/scicloj/metamorph) based machine learning pipelines

## Main idea

This library is based on the idea, that in machine learning model evaluations,
we often do not want to tune only the model and its hyper-parameters,
but the whole data transformation pipeline.

It unifies the often seperated concerns of tunig of data preprocessing and
tuining of model hyper-parameters.

In a lot of areas of machine learning, certain aspects of the data
pre-processing needed to be tuned (by trying), as no clear-cut decisions exists.

One example could be the number of dimensions in PCA or the vocabulary size in a NLP ML model.
But it can be as well a "boolean" alternative, such as if stemming should be used or not.

This library allows exactly this, namely hyper-tune an arbitraty complex data transformation pipeline.

Several code examples for metamorph are available in this repo: [metamorph-examples](https://github.com/scicloj/metamorph-examples)

## Quick start

If you just want to see code, here it is:

Some libraries are needed for a complete test case, 
see the deps.edn file in alias "test".

```clojure
(require
 '[tech.v3.dataset :as ds]
 '[tech.v3.dataset.metamorph :as ds-mm]
 '[scicloj.metamorph.core :as morph]
 '[tech.v3.ml.metamorph :as ml-mm]
 '[tech.v3.dataset.modelling :as ds-mod]
 '[tech.v3.dataset.column-filters :as cf]
 '[tablecloth.api.split :as split]
 '[scicloj.metamorph.ml :as ml-eval]
 '[tech.v3.ml.loss :as loss]

 )

;;  the data
(def  ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

;;  the (single, fixed) pipe-fn
(def pipe-fn
  (morph/pipeline
   (ds-mm/set-inference-target :species)
   (ds-mm/categorical->number cf/categorical)
   ;; sets the ground truth for the prediction into the context at the required key
   (fn [ctx]
     (assoc ctx
            :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
   (ml-mm/model {:model-type :smile.classification/random-forest})))

;;  the simplest split, produces a seq of one, a single split into train/test
(def  train-split-seq (split/split ds :holdout))

;; we have only one pipe-fn here
(def  pipe-fn-seq [pipe-fn])

(def  evaluations
  (ml-eval/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss))

;; we have only one result
(def best-fitted-context (-> evaluations first :fitted-ctx))
(def best-pipe-fn (-> evaluations first :pipe-fn))


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


## Evaluate pipelines

This library contains very little code itself, just one function
`evaluate-pipelines` which takes a sequence of metamorph compliant pipeline-fn (= each pipeline is a series of steps to transform the raw data and a model step)

It executes each pipeline first in `mode` :fit and then in `mode` transform, as specified by [metamorph](https://github.com/scicloj/metamorph)
which a pipeline step containing a model should then translates into a train/predict pattern including evaluation of the result.

The last step of the pipelne function should be a "model", so something which can be trained / predicted.

This library does not care, what exactly is the last step.
It calls the whole pipeline in mode :fit and mode :transform and the model step is supposed to do the right thing.

Here is a well behaving model function from `tech.ml`: 
https://github.com/techascent/tech.ml/blob/38f523a7cea6465f639df6fc4eecd6b3f4de69d0/src/tech/v3/ml/metamorph.clj#L12
which calls `train` and `preditc` accordingly. 

So for each pipeline-fn in the sequence given to `evaluate-models` one model will be trained and evaluated.

It does this for each pipeline-fn given and each pipeline-fn gets evaluated using each of the given test/train splits.

This can be used to implement various cross-validation strategies, just as holdout, k-fold and others.

Each pipeline is typically a variation of a certain standard pipeline, 
and encapluslates therefore individual machine learning trials with the goal to find the best performing model automatically.

This is often called hyper-parameter tuning. But here we can do it for all options of the pipeline, and not only for the hyper-parameters of the model itself.

It is of course possible to just have a single pipeline function in the sequence. Then a single model will be trained.

The different pipeline-fns are completely indepedent from each other and can contain the same model, different models, or anything the developper wants.

This libray does not contain itself functions to create metamorph pipelines including their variations.

This can be done in various ways, from hand coding each pipeline  or having a pipeline creation function  over to using grid search libraries:
https://github.com/techascent/tech.ml/blob/38f523a7cea6465f639df6fc4eecd6b3f4de69d0/src/tech/v3/ml/gridsearch.clj#L111

A simple pipeline looks like thgis:

```clojure
 (morph/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number cf/categorical)
         (fn [ctx]
           (assoc ctx
                  :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
         (ml-mm/model {:model-type :smile.classification/random-forest}))
```

Pipelines can be created as well declarative based on maps, see here: 
https://scicloj.github.io/tablecloth/index.html#Declarative
https://github.com/scicloj/tablecloth/blob/pipelines/src/tablecloth/pipeline.clj

The train/test split sequence can as well be generated in any way. It need to be a sequecne of maps contain a "tech.ml.dataset" 
at key :train and an other dataset  at key :test. These will be used to train / predict and evaluate one pipeline.


`evaluates-pipelines` returns then a sequenxe  of model evaluations.
It returns #pipeline-fn x  #cross-validation-splits evaluation results. This is as well the total number of models trained in total.

Each evaluation result contains a map with these keys:

key | explanation
------ | --------
 :fitted-ctx  | the fitted pipeline context(including the trained model and the dataset at end of pipeline) after the pipleine was run in mode :fit 
 :prediction-context | the predicted pipline context (including the predition dataset) after pipeline was run in mode :transform
 :scicloj.metamorph.ml/target-ds  | A dataset containing the ground truth
 :pipe-fn | the pipeline-fn doing the full transformation including trin/predict
 :metric | the score for this model evaluation
 :avg | average score of this pipe-fn (averaged over the score of all train/test splits)
 :min | min score of this pipe-fn
 :max | max score of this pipe-fn

This returned information is as well self-contained, as the pipeline-fn should manipulated exclusively the dataset and the available pipeline context.
This means, the :pipe-fn functions can be re-executed simply on new data.

This is due to the metamorph approach, which keeps all input/output of the pipeline inside of the context / dataset.

## Metamorph

The pipeline functions passed into `evaluate-pipelines` need to be metamorph compliant as explained here: 
https://github.com/scicloj/metamorph

'compliant' means simply to adhere to interact with a context map with certain standard keys

A pipeline-fn is a composition of metamorph compliant data transform functions.

The following projects contain them, for both:
- dataset manipulations
- train/prediction of models
 
and custom ones can be created easely:

- https://github.com/techascent/tech.ml.dataset
- https://github.com/scicloj/tablecloth
- https://github.com/techascent/tech.ml
- https://github.com/scicloj/sklearn-clj



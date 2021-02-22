# metamorph.ml

Evaluation function of metamorph based ml pipelines

## Main idea

This library is based on the idea, that in machine learning model evaluations,
we often do not want to tune only the model and its hyperparameters,
but the whole data transformation pipeline.

It unifies the often seperated concerns of data preprocessing + 
hyper-parameter tuning.

In a lot of areas of machine learning, certain aspect of the 
pre-processing needed to be tuned (by trying), as no clear-cut decisions exists.

One example could be the number of dimensions in PCA or the vocabulary size in a NLP ML model.
But it can be as well a "boolean" alternative, such as if stemming should be used or not.

This library allows exactly this, namely hyper-tune an arbitraty complex data transformtion pipeline.



## Quick start

If you just want to see code, here it is:

Quite some libraies are needed for a complete test case, 
see the deps.edn file in alias "test"

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
   ;; sets the ground trueth for the prediction into the context at the required key
   (fn [ctx]
     (assoc ctx
            :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
   (ml-mm/model {:model-type :smile.classification/random-forest})))

;;  the simplest split, produces a seq of one, a single split into train/test
(def  train-split-seq (split/split ds :holdout))

;; we have only one pipe-fn 
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
`evaluate-pipelines` which takes a sequence of metamorph compliant pipeline-fn (= each function is a series of steps to transform the raw data and a model step)
It executes each pipeline in fit/transform, which translates into a train/predict pattern including evaluation of the result.

The last step of the pipelne function should be a "model", so something which can be trained / predicted.
This library does not care, what exactly is teh last step.
It calls the whole pipleine in mode :fit and mode :transform and teh model stpe, is supposed to do the right thing.
Here is such a well beahving model function from `tech.ml`: 
https://github.com/techascent/tech.ml/blob/38f523a7cea6465f639df6fc4eecd6b3f4de69d0/src/tech/v3/ml/metamorph.clj#L12

So for each pipeline-fn one model will be trained.

It does this for each pipeline-fn given and each pipeline gets evaluated using each of the given test/train splits.

This can be used to implement various cross-validation strategies, just as holdout, k-fold and others.

Each pipeline is typically a variation of a certian standard pipeline, 
and encapluslates therefore individual trials with the goal to find the best model automatically.

This is often called hyper-parameter tuning. But here we can do it for all options of the pipeline, and not only for the hyper-parameters of the model itself.

It is of course possible to just have a single pipline function in the sequnence. Then a single model will be trained.

The different pipeline-fn are completely indepedent from each other and can contain the same model, different models, or anything the developper codes.
Very often they are "variations" of each other, but this is not required.

This librray does not contain any code to create metamorph pipelines including their variations.

This can be done in various ways, from hand coding each pipelne  or having a pipelne cteation function  over using grid search libraries:
https://github.com/techascent/tech.ml/blob/38f523a7cea6465f639df6fc4eecd6b3f4de69d0/src/tech/v3/ml/gridsearch.clj#L111

Pipelines can be created as well declarative based on maps, see here: 
https://scicloj.github.io/tablecloth/index.html#Declarative
https://github.com/scicloj/tablecloth/blob/pipelines/src/tablecloth/pipeline.clj

The train/test split sequence can s well be genrated in any way. It need to be a sequecne of maps contain a "tech.ml.dataset" 
at key :train an dan other at :test. These will be used to train / predict and evaulate one pipeline.


`evaluates-pipelines` returns then a list of #pipeline-fn x  #cross-validation-splits evaluation result. This is as well the total number of models trained.

Each evaluation result contains a map with these keys:

- :fitted-ctx        - the fitted pipeline context(including the trained model and the dataset at end of pipeline)
- :predicted context - the predicted pipline context (including the predition dataset)
- :scicloj.metamorph.ml/target-ds   the ground trueth
- :pipe-fn - the pipeline-fn
- :loss the loss of this model evaluation
- :avg-loss average/min/max loss of this pipeline (over all train/test splits)
- :min-loss  
- :max-loss 

This returned information is self-contained, as the pipeline-fn should manipulated exclusively the dataset and the available pipeline context.
This means the :pipe-fn function can be re-executed simply on new data.

This is due to the metamorph approach, which keeps all input/output of the pipeline inside of the context / dataset.

## Metamorph

The pipeline functions passed into `evaluate-pipelines` need to be metamorph compliant as explained here: 
https://github.com/scicloj/metamorph

'compliant' means simply to adhere to interact with a context map with certain standard keys

A pipeline is a composition of metamorph compliant data transform functions.
The follwoing projects contain them, and custom ones can be created easely:

- https://github.com/techascent/tech.ml.dataset
- https://github.com/scicloj/tablecloth
- https://github.com/techascent/tech.ml
- https://github.com/scicloj/sklearn-clj

(at present the support for metamorph is in the baseline of the code, but not releases yet)

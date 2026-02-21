unreleased
- upgraded deps
- added missing Java classes from former tmd-smile support
- more clear error message for accuray of categorical maps
- made  internal functions private
   scicloj.metamorph.ml.rdatasets
     clean-R-relevant
     doc-url->md
     _fetch-dataset  

   scicloj.metamorph.ml.text
      make-mmap-container
      make-container
      ->column--concat-buffers
      ->column--coalesce-blocks

    scicloj.metamorph.ml.viz
      apply-xform-kvs

    scicloj.metamorph.ml
      score
          
         
    
          
        

1.2.1
- upgrade deps
- fixed @37


1.2
- removed most toy datasets and forward existing fns to rdatasets/*
    - attention: column nams of some 'toydatasets' have changed, now in kebab-keyword style
- additional validation of consitency of predict / cat maps 

1.1.1
* cleaned column names from "." in rdatasets
* better initial doc string for rdatasets functions
* new fn 'dataset-descriptions->doc-strings!' to attach dataset description as docsting to all rdadasets functions

1.1
* added access fn to all datasets from https://vincentarelbundock.github.io/Rdatasets/articles/data.html


1.0
* use malli to describe model options and validate them in ml/train
* renamed :other-metrices -> :other-metrics

0.12
* fixes #30 - dummy classifier does not predict by majority #30
* added dummy regression model
* improved design-matrix feature . Breaking !  columns need to be refered know by "precise name" (string, symbol, keyword)
* made model  :fastmath/ols frezzable by nippy
 

0.11.1
* allow parameters for :fastmath/ols (fixes #27)
* added optional caching for train / predict
* added new evaluation-handler: metrics-and-model-keep-fn
* added option :ppmap with :ppmap-grain-size 10 to ml/eval-pipelines
* added more evaluation-handler fns suitable for model-spec search
* added :probability-distributin to ml/eval-pipelines result
* breaking: move all eval handler to ns scicloj.metamorph.ml.evaluation-handler

0.10.4
* added :target-datatypes in train result and clarified expected 'shape' of prediction

0.10.3
 * re-added data.json dependency

0.10.2
 * re-added data.csv dependency

0.10.1
 * fixed cljdoc build issue

0.10.0

* exclude most hanami deps as we don't need them
* added support for tidy-text and 
* added read/write support for TMD<->libsvm files
 

0.9.0

- added linear models from fastmath 
- added tidy output validation
- added support for design-matrix

0.8.2
 -fixed metric bug
 
0.8.1
 -fixed bug in ml/tidy

0.8.0
- upgraded deps
- added suport for glance,augment,tidy

0.7.10
- added missing deps

0.7.9
- fixed and documented confusion-map->ds
- added confusin matrix plot
- added generic handling of loglikelyhood
- added mtcars data
- added aic and bic

0.7.8
- fixed default colors of error bands

0.7.7
- fixed test/train color assignments
- adde more docu

0.7.6
- fixed metric.clj filename for ClojureDoc generation

0.7.5
- removed WIP files

0.7.4
- fixed Clojars links and ClojureDoc
- refatored learnining curve

0.7.3

- added dummy classifier
- add checks for matching categorical maps between train and predict
- added titanic toy data
- added ggplot toydaya
- added visualisation for learning curve

0.7.2
 - fixed verify ns

0.7.1
 - fixed 'verify' ns
 
0.7
 - using tablecloth 7.0
 - added uid for grouping of splits
 - added learning curve
 - added AUC metric (thanks @Prometheus77)

0.6.4
- fix metric calculation in evaluate-pipelines 
    - use reverse mappings

0.6.3

- added  :as opts to methods signature

0.6.2
 - added ensembles

0.5.0
- unified dissoc-in options and handler-fn option



0.4.1
- adding support for unsupervised learning


0.4.0
- changed result of `evaluate-pipelines`` to be more consistent
- added Malli schema to several functios
- added support for experiment tracking on disk
- added experimenta / example code of using nippy to persist evaluation results

0.3.0-beta6
- fixed toy data

0.3.0-beta4
- added sonar data

0.3.0-beta3
- added std-scaler
- added min-max scaler
- added toy data


0.3.0-beta2

0.3.0-beta1 - 06.04.2021
- big change, tech.ml (core) was moved into here


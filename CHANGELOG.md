unreleased
* allow parameters for :fastmath/ols (fixes #27)

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


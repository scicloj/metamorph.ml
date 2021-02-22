# metamorph.ml

Evaluation function of metamorph based ml pipelines

# Quick start

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
   ;; sets the truth into the context at the required key
   (fn [ctx]
     (assoc ctx
            :scicloj.metamorph.ml/target-ds (cf/target (:metamorph/data ctx))))
   (ml-mm/model {:model-type :smile.classification/random-forest})))

;;  the simplest split, produce a seq of one split into train/test
(def  train-split-seq (split/split ds :holdout))

;; only one pipe-fn in the seq
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

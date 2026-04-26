(ns read)

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

;; -----------------------------




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

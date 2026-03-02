(ns classification-exp
  (:require
   [scicloj.ml.smile.classification]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.classification]
   [scicloj.metamorph.ml.rdatasets :as datasets]
   [scicloj.metamorph.ml.random-forest]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.column-filters :as cf]
   [tablecloth.api :as tc]
   [scicloj.sklearn-clj.ml]
   [libpython-clj2.python]
   ))

(apply tc/concat-copying (repeat 10 (datasets/openintro-birds)))

(def data
  (->
   (apply tc/concat-copying (repeat 10 (datasets/openintro-birds)))
   (tc/select-columns [:effect :speed :height :phase-of-flt :time-of-day])
   (tc/replace-missing [:effect] :value "None")
   (tc/drop-rows (fn [row]
                   (= "None" (:effect row))))
   (ds/categorical->number [:effect :phase-of-flt :time-of-day])
   (tc/drop-missing)
   (ds-mod/set-inference-target :effect)))

(-> data :effect frequencies)
(defn train [model-opts]
  (let [;; Create simple iris-like dataset
        
        ;; Mark species as categorical
        split (tc/split->seq data :holdout {:seed 1234})

        ;; Train model
        start (System/currentTimeMillis)
        model (ml/train  (:train (first split))
                         model-opts)

        _ (def model model)
        ;; Predict
        predictions (ml/predict (-> split first :test cf/feature) model)
        _ (def predictions predictions)
        end (System/currentTimeMillis)
        actual (vec (-> split first :test :effect))
        predicted (map int (vec (predictions :effect)))

        correct (count (filter true? (map = actual predicted)))
        accuracy (double (/ correct (count actual)))]

    {:model-opts model-opts
     :accuracy accuracy
     :time (/ (- end start) 1000.0)
     }))
(println 
 (tc/dataset
  [
   (train {:model-type :metamorph.ml/dummy-classifier})
   (train {:model-type :metamorph.ml/random-forest
           :n-trees 100
           :parallel true
           :max-depth 20
           })
   (train
    {:model-type :smile.classification/random-forest
     :trees 100})
   (train {:model-type :smile.classification/logistic-regression})
   (train {:model-type :smile.classification/ada-boost})
   (train {:model-type :sklearn.classification/decision-tree-classifier})
   (train {:model-type :sklearn.classification/random-forest-classifier :n-jobs -1})
   ]))


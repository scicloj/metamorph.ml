(ns scicloj.metamorph.learning-curve-test
  (:require  [clojure.test :as t]
             [tech.v3.dataset]
             [scicloj.metamorph.core :as mm]
             [tech.v3.dataset.metamorph :as mds]
             [tablecloth.api :as tc]
             [tablecloth.pipeline :as tc-mm]
             [scicloj.metamorph.ml]
             [scicloj.metamorph.ml.loss]
             [scicloj.metamorph.ml.learning-curve :as lc]
             [scicloj.ml.smile.classification]))


(require
 '[scicloj.ml.core :as ml]

 '[scicloj.ml.metamorph :as mm]
 '[scicloj.ml.dataset :as ds]
 '[scicloj.ml.smile.classification])

(def titanic-train
  (->
   (tech.v3.dataset/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                              {:key-fn keyword})
   (tc/shuffle {:seed 1234})))
                ;; :parser-fn :string




;; construct pipeline function including Logistic Regression model
(def pipe-fn
  (mm/pipeline
   (mds/select-columns [:Pclass :Survived :Embarked :Sex])
   (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                          1 "yes"
                                                          0 "no")
                                                      (:Survived ds))))
   (mm/def-ctx ss)
   (mds/categorical->number [:Survived :Sex :Embarked])
   (mds/set-inference-target :Survived)

   {:metamorph/id :model}
   (scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest})))


(def lc
  (lc/learning-curve titanic-train
                     pipe-fn
                     (range 0.1 1 0.1)
                     10))


(range 0.1 1 0.2)

(tc/row-count
 (tc/head titanic-train (* 20 0.8999999999999)))

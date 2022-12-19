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



(def titanic-train
  (->
   (tech.v3.dataset/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                              {:key-fn keyword})
   (tc/shuffle {:seed 1234})))

(def pipe-fn
  (mm/pipeline
   (mds/select-columns [:Pclass :Survived :Embarked :Sex])
   (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                          1 "yes"
                                                          0 "no")
                                                      (:Survived ds))))
   (mds/categorical->number [:Survived :Sex :Embarked])
   (mds/set-inference-target :Survived)

   {:metamorph/id :model}
   (scicloj.metamorph.ml/model {:model-type :smile.classification/random-forest})))

(t/deftest test-learnining-curve []
  (let [lc
        (lc/learning-curve titanic-train
                           pipe-fn
                           (range 0.3 1 0.3)
                           {:k 3})]


    (t/is (= [:metric-train :train-size-index :metric-test :test-ds-size :train-ds-size]
             (tc/column-names lc)))


    (t/is (= {"1" 3, "2" 3, "0" 3}
             (frequencies (:train-size-index lc))))))

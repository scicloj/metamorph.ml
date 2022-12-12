(ns nb
  (:require [nextjournal.clerk :as clerk]
            [tech.v3.dataset]
            [scicloj.metamorph.core :as mm]
            [tech.v3.dataset.metamorph :as mds]
            [tablecloth.api :as tc]
            [tablecloth.pipeline :as tc-mm]
            [scicloj.metamorph.ml]
            [scicloj.metamorph.ml.loss]
            [scicloj.metamorph.ml.learning-curve :as lc]
            [scicloj.metamorph.ml.viz :as ml-viz]
            [scicloj.ml.smile.classification]
            [aerial.hanami.common :as hc]))
            

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/serve! {:browse true}))

^{:nextjournal.clerk/viewer :table
  :nextjournal.clerk/opts {:page-size 5}}
(def titanic-train
  (->
   (tech.v3.dataset/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                              {:key-fn keyword})
   (tc/shuffle {:seed 1234})))
(defn make-pipe-fn [model-type]
  (mm/pipeline
   (mds/select-columns [:Pclass :Survived :Sex :Embarked :Parch])
   (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                          1 "yes"
                                                          0 "no")
                                                       (:Survived ds))))

   (mds/categorical->number [:Survived :Pclass :Sex :Embarked])
   (mds/drop-missing)
   (mds/set-inference-target :Survived)

   {:metamorph/id :model}
   (scicloj.metamorph.ml/model {:model-type model-type})))


^{:nextjournal.clerk/viewer :table}
(def lc-rf
  (lc/learning-curve titanic-train
                     (make-pipe-fn :smile.classification/random-forest)
                     (range 0.1 1 0.1)
                     10))



^{:nextjournal.clerk/viewer :vega-lite}
(hc/xform ml-viz/learning-curve-spec
          :VALDATA
          (-> lc-rf
              (tc/pivot->longer [:metric-test :metric-train]
                                {:value-column-name :metric
                                 :target-columns :train-test-metric})
              (tc/rows :as-maps)))



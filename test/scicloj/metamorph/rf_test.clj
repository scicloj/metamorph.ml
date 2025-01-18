(ns scicloj.metamorph.rf-test
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.modelling :as ds-mod]

   
   [scicloj.ml.smile.classification]
   [scicloj.ml.tribuo]
   [scicloj.metamorph.ml.classification]

   [scicloj.metamorph.core :as mm]
   [camel-snake-kebab.core :as csk]
   ))

(def titanic
  (tc/dataset "bigdata/huge_1M_titanic.csv" {:key-fn csk/->kebab-case-keyword}))

(tc/info titanic)


(def categorical-columns [:sex :pclass :survived])



(tc/column-names titanic)



(def titanic-final
  (-> titanic
      (tc/select-columns  [:sex :pclass :survived])
      (ds/categorical->number categorical-columns)
      (ds-mod/set-inference-target :survived)
      (tc/shuffle)
      (tc/head 10000)
      ))

(def titanic-split
  (tc/split->seq titanic-final :holdout))


(defn make-pipe [ model-spec]
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model
    model-spec))
  )

(def tribuo-rf-spec
  {:model-type :scicloj.ml.tribuo/classification
   :tribuo-components [{:name "rf" :type "org.tribuo.common.tree.RandomForestTrainer"
                        :properties {:innerTrainer "cart"
                                     :combiner "votecombiner"
                                     :numMembers "10"
                                     :seed "12345"}}

                       {:name "cart" :type "org.tribuo.classification.dtree.CARTClassificationTrainer"
                        :properties {:maxDepth "8"
                                     :fractionFeaturesInSplit "0.5"
                                     :seed "12345"
                                     :impurity "gini"}}
                       {:name "gini" :type "org.tribuo.classification.dtree.impurity.GiniIndex"}
                       {:name "entropy" :type "org.tribuo.classification.dtree.impurity.Entropy"}
                       {:name "votecombiner" :type "org.tribuo.classification.ensemble.VotingCombiner"}]
   :tribuo-trainer-name "rf"})


(def eval-result
  (ml/evaluate-pipelines 
   [(make-pipe {:model-type :metamorph.ml/dummy-classifier}) 
    (make-pipe {:model-type :metamorph.ml/rf-classifier  :n-trees 10}) 
    (make-pipe {:model-type :smile.classification/random-forest :trees 10}) 

    (make-pipe tribuo-rf-spec)
    ]
   titanic-split
   loss/classification-accuracy
   :accuracy
   {:return-best-pipeline-only false
    :evaluation-handler-fn identity}
   ))

(-> titanic-split first :train :survived frequencies)

(->>
 eval-result
 flatten

 (map (comp :survived :metamorph/data :ctx :test-transform))
 )


(println
 (tc/dataset
  (->>
   eval-result
   flatten
   
 ;(map (comp keys :test-transform))
   (map #(hash-map 
          :model-type (get-in % [:fit-ctx :model :options :model-type])
          :timing-fit (get-in % [:timing-fit])
          :test-transform--timing (get-in % [:test-transform :timing])
          :train-transform--timing (get-in % [:train-transform :timing])
          :test-metric (get-in % [:test-transform :metric])
          :prediction (frequencies (get-in %  [:test-transform :ctx :metamorph/data :survived]))
          ))
   )))


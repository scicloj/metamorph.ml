(ns sciloj.evaluation-test
  (:require  [clojure.test :refer [is deftest]]
             [scicloj.metamorph.core :as morph]
             [tech.v3.libs.smile.classification]
             [tech.v3.dataset.metamorph :as ds-mm]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.modelling :as ds-mod]
             [tech.v3.dataset.column-filters :as cf]
             [tech.v3.ml.metamorph :as ml-mm]
             [tech.v3.ml.gridsearch :as gs]
             [scicloj.metamorph.ml :as ml-eval]
             [tech.v3.ml.loss :as loss]

            [tech.v3.ml.gridsearch :as ml-gs]
            [tablecloth.api.split :as split]
            ))

(defn pipe-create-fn [options]
  (morph/pipeline
   (ds-mm/set-inference-target :species)
   (ds-mm/categorical->number cf/categorical)
   (fn [ctx]
     (assoc ctx
            :target-ds (cf/target (:metamorph/data ctx))
            )
     )
   (ml-mm/model (merge options
                       {:model-type :smile.classification/random-forest
                        }))))



(deftest evaluate-model-simplest

  (def ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

  (defn pipe-create-fn [options]
    (morph/pipeline
     (ds-mm/set-inference-target :species)
     (ds-mm/categorical->number cf/categorical)
     (fn [ctx]
       (assoc ctx
              :target-ds (cf/target (:metamorph/data ctx))))
     (ml-mm/model {:model-type :smile.classification/random-forest})))

  (def res
    (ml-eval/evaluate-model ds pipe-create-fn nil :holdout nil loss/classification-loss {}))


  (def best-fitted-context  (-> res first :fitted-ctx))

  (def best-pipe-fn (-> res first :pipe-fn))

  (def new-ds (ds/sample ds 10 {:seed 1234} ))

  (->
   (best-pipe-fn
    (merge best-fitted-context
           {:metamorph/data new-ds
            :metamorph/mode :transform}))
   (:metamorph/data)
   (ds-mod/column-values->categorical :species))

  )

(deftest grid-seach
  (let [
        ds
        (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword})


        grid-search-options
        {:trees (gs/categorical [10 50 100 500])
         :split-rule (gs/categorical [:gini :entropy])
         }


        res
        (ml-eval/train-auto-gridsearch
         ds pipe-create-fn
         grid-search-options
         {:loss-fn loss/classification-loss} )]

    (map
     #(-> % :avg-loss)
     res)
    ;; res
    )
  ;; (def fitted-ctx (pipe-fn {:metamorph/data (:train-ds split) :metamorph/mode :fit}))
  ;; (def prediction-ctx (pipe-fn (merge fitted-ctx {:metamorph/data (:test-ds split) :metamorph/mode :transform})))

  )
(comment

  (def  ds
    (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))

  (defn pipe-create-fn [options]
    (morph/pipeline
     (ds-mm/set-inference-target :species)
     (ds-mm/categorical->number cf/categorical)
     (fn [ctx]
       (assoc ctx
              :target-ds (cf/target (:metamorph/data ctx))))
     (ml-mm/model (merge options
                         {:model-type :smile.classification/random-forest}))))

  (def pipe-options
    {:trees (gs/categorical [10 50 100 500])
     :split-rule (gs/categorical [:gini :entropy])})

  (def split-type :kfold)
  (def split-options {:k 10})

  (def loss-fn loss/classification-loss)




  (def res
    (evaluate-model ds pipe-create-fn pipe-options split-type split-options loss-fn))

  (def best-fitted-context
    (-> res first first :fitted-ctx)) ;; ignoring "lowest" loss for th moment

  (def best-pipe-fn
    (-> res first first :pipe-fn))


  (def new-ds (ds/sample ds))

  (best-pipe-fn
   (merge best-fitted-context
          {:metamorph/data new-ds
           :metamorph/mode :transform}))




  (map
   (fn [{:keys [train test]}]
     [(ds/row-count train)
      (ds/row-count test)
      ]

     )
   ))

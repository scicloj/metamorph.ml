(ns scicloj.metamorph.ml.ensemble
  (:require [clojure.test :refer [deftest is] :as t]
            [scicloj.metamorph.core :as morph]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.gridsearch :as gs]
            [scicloj.metamorph.ml.loss :as loss]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tablecloth.api :as tc]
            [taoensso.nippy :as nippy]
            [scicloj.metamorph.ml.metrics]
            [tablecloth.pipeline :as tcp]
            [malli.core :as m]
            [malli.instrument :as mi]
            [malli.generator :as mg]
            [scicloj.metamorph.ml.evaluation-handler :as eval]
            [scicloj.metamorph.ml.evaluation-handler :refer [get-source-information qualify-pipelines qualify-keywords]]))

(defn majority [l]
  (->>
   (frequencies l)
   seq
   (sort-by second)
   reverse
   first
   first))


(defn  ensemble-pipe [pipe-1 pipe-2]
  (morph/pipeline
   (fn [{:metamorph/keys [id data mode] :as ctx}]

     (case mode
       :fit
       (assoc ctx
              id {
                  :pipe-1 (morph/fit-pipe (:metamorph/data ctx) pipe-1)
                  :pipe-2 (morph/fit-pipe (:metamorph/data ctx) pipe-2)})
       :transform
       (let [transformed-ctx-1 (morph/transform-pipe data pipe-1 (-> ctx (get id) :pipe-1))
             transformed-ctx-2 (morph/transform-pipe data pipe-1 (-> ctx (get id) :pipe-2))
             prediction-1 (cf/prediction (:metamorph/data transformed-ctx-1))
             prediction-2 (cf/prediction (:metamorph/data transformed-ctx-2))
             prediction-ds (-> (ds/new-dataset [
                                                (ds/new-column :model-1 (:species prediction-1))
                                                (ds/new-column :model-2 (:species prediction-2))])
                               (tc/add-column :prediction
                                              (fn [ds]
                                                (->> ds
                                                     tc/rows
                                                     (map majority)))))]
         ;; predictions (ds/->dataset)

         (def transformed-ctx-1 transformed-ctx-1)
         (def prediction-1 prediction-1)
         (:species prediction-1)
         (assoc ctx
                :metamorph/data prediction-ds
                id
                {:pipe-1 transformed-ctx-1
                 :pipe-2 transformed-ctx-2}))))))

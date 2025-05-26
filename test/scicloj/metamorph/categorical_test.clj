(ns scicloj.metamorph.categorical-test
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.categorical :as cat]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))




(deftest independent
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :independent))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z]})
          :metamorph/mode :fit})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx :yy :zz]})
                 :metamorph/mode :transform}))]
    (is (= #{:x :y :z} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:xx :yy :zz} (-> transformed-ctx :metamorph/data :a-xx meta :one-hot-map :one-hot-table keys set)))))


(deftest train-to-test
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :fit))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z :xx]})
          :metamorph/mode :fit})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx]})
                 :metamorph/mode :transform}))]

    (is (= #{:xx :x :y :z} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:xx :x :y :z} (-> transformed-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))))



(deftest global-ds
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :full {:result-datatype :int}))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z]})
          :metamorph/mode :fit
          :metamorph.ml/full-ds (tc/dataset {:a [:x :y :z :xx :yy :zz]})})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx :yy :zz]})
                 :metamorph/mode :transform}))]

    (is (= #{:x :y :z :xx :yy :zz} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:x :y :z :xx :yy :zz} (-> transformed-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))))



(deftest dummy-model-one-hot
  (let [simple-ready-for-train
        (->
         {:x-1 [0 1 0]
          :x-2 [1 0 1]
          :cat [:a :b :c]
          :y [:a :a :b]}

         (ds/->dataset)
         (ds/categorical->number [:y])
         (ds/categorical->one-hot [:cat])
         (ds-mod/set-inference-target [:y]))

        simple-split-for-train
        (first
         (tc/split->seq simple-ready-for-train :holdout {:seed 112723}))
        model
        (ml/train (ds-mod/set-inference-target
                   (:train simple-split-for-train) :y)
                  {:model-type :metamorph.ml/dummy-classifier})]
    (is (= [:model-data :options :train-input-hash :id :feature-columns :target-columns :target-datatypes :target-categorical-maps]
           (keys model)))))



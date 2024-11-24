(ns scicloj.metamorph.classification-test
  (:require [scicloj.metamorph.ml.classification :refer [confusion-map confusion-map->ds]]
            [clojure.test :refer :all]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.core :as mm]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.categorical :as ds-cat]
            [tablecloth.api :as tc]
            [scicloj.ml.smile.classification]
            [scicloj.metamorph.ml.toydata :as toydata]))


(deftest test-normalized
  (is (=
       (confusion-map [:a :b :c :a] [:a :c :c :a] :none)
       {:a {:a 2}
        :c {:b 1 :c 1}}))

  (is (=
       (confusion-map [:a :b :c :a] [:a :c :c :a])
       {:a {:a 1.0}
        :c {:b 0.5 :c 0.5}})))


(deftest test-confusion-map->ds
  (is (=
       [{:column-name :a, :a "2", :c "0"}
        {:column-name :c, :a "0", :c "1"}]
       (-> (confusion-map [:a :b :c :a] [:a :c :c :a] :none)
           (confusion-map->ds)
           (tc/rows :as-maps))))


  (is (=
       [{:column-name :a, :a "1.000", :c "0.000"}
        {:column-name :c, :a "0.000", :c "0.5000"}]
       (-> (confusion-map [:a :b :c :a] [:a :c :c :a] :all)
           (confusion-map->ds)
           (tc/rows :as-maps)))))



(deftest dummy-classification-fixed-label []
  (let [ds (toydata/iris-ds)
        model (ml/train ds  {:model-type :metamorph.ml/dummy-classifier
                             :dummy-strategy :fixed-class
                             :fixed-class 0})
        prediction (ml/predict ds model)]

    (is (= {:species :int16} (:target-datatypes model)))
    (is (= (:species prediction) (repeat 150 0)))))

(deftest dummy-classification-majority []
  (let [ds (toydata/breast-cancer-ds)
        model (ml/train ds {:model-type :metamorph.ml/dummy-classifier
                            :dummy-strategy :majority-class})


        prediction (ml/predict ds model)]

    (is (= (:class prediction) (repeat 569 0)))))



(deftest dummy-classification-random []
  (let [ds (toydata/breast-cancer-ds)
        model (ml/train ds {:model-type :metamorph.ml/dummy-classifier
                            :dummy-strategy :random-class})


        prediction (ml/predict ds model)]
   (is (= [0 1] (-> prediction :class distinct sort)))))

(deftest categorical-not-needed-for-ml

  (let [ds (->
            {:x [0] :y [:a]}
            (ds/->dataset)
            (ds-mod/set-inference-target :y))

        model (ml/train ds {:model-type :metamorph.ml/dummy-classifier
                            :dummy-strategy :random-class})]

    (is (= [:a ] (-> (ds/->dataset {:x [0]}) (ml/predict model) :y)))))


(deftest dummy-categorical

  (let [ds (->
            {:x [0] :y [:a]}
            (ds/->dataset)
            (ds/categorical->number [:y])
            (ds-mod/set-inference-target :y))

        model (ml/train ds {:model-type :metamorph.ml/dummy-classifier
                            :dummy-strategy :random-class})]


    (is (= [:a ] (-> (ds/->dataset {:x [0]}) (ml/predict model) (ds-cat/reverse-map-categorical-xforms) :y)))))

(deftest dummy-pipeline-eval
  (let [pipe-fn (mm/pipeline
                 {:metamorph/id :model} (ml/model {:model-type :metamorph.ml/dummy-classifier :dummy-strategy :majority-class}))
        data-split (update-keys
                    (ds-mod/train-test-split (toydata/breast-cancer-ds) {:seed 123})
                    {:train-ds :train
                     :test-ds :test})
        eval-results (ml/evaluate-pipelines
                      [pipe-fn]
                      [data-split]
                      loss/classification-accuracy
                      :accuracy)]
    (is (= 0.391812865497076
           (->
            eval-results
            flatten
            first
            :test-transform
            :metric)))))


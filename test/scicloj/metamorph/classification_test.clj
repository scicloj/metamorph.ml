(ns scicloj.metamorph.classification-test
  (:require [scicloj.metamorph.ml.classification :refer [confusion-map]]
            [clojure.test :refer :all]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as ds]
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


(deftest dummy-classification-fixed-label []
  (let [ds (toydata/iris-ds)
        model (ml/train ds  {:model-type :metamorph.ml/dummy-classifier
                             :dummy-strategy :fixed-class
                             :fixed-class 0})

        prediction (ml/predict ds model)]

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

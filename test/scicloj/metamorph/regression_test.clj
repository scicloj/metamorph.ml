(ns scicloj.metamorph.regression-test 
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset.modelling :as ds-mod]))


(deftest linear-regression-mtcars-fm-ols
  (let [ds (-> (tc/dataset {:x [1 2 3] :y [2 4 6]})
               (ds-mod/set-inference-target :y))




        model (ml/train ds {:model-type :metamorph.ml/dummy-regressor})


        prediction (:y (ml/predict ds model))]

    (is (= [4.0 4.0 4.0] prediction))
    (is (= :prediction (-> prediction meta :column-type)))))
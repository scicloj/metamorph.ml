(ns verify-test
  (:require [scicloj.metamorph.ml.verify :as verify]
            [scicloj.metamorph.ml.regression]
            [clojure.test :refer [deftest is]]
            )
  )

(deftest test-basic-regression
  (verify/basic-regression {:model-type :metamorph.ml/dummy-regressor} 0.8))

(deftest test-basic-classification
 (verify/basic-classification {:model-type :metamorph.ml/dummy-classifier} 0.3))
(ns scicloj.metamorph.loss-test
  (:require [scicloj.metamorph.ml.loss :refer :all]
            [clojure.test :refer :all]))


(deftest test-auc
  (is (=
       (auc [0.3 0.1 0.5 0.4 0.6 0.7 0.2] [0 0 0 0 1 1 1])
       0.75)))

(ns scicloj.metamorph.loss-test
  (:require [scicloj.metamorph.ml.loss :refer [auc classification-accuracy]]
            [clojure.test :refer [deftest is]]
            [tech.v3.dataset :as ds]))


(deftest test-auc
  (is (=
       (auc [0.3 0.1 0.5 0.4 0.6 0.7 0.2] [0 0 0 0 1 1 1])
       0.75)))

(deftest classification-accuracy-test

  (is (= 1.0
         (classification-accuracy [0 0] [0.0 0.0]))) ;; => 1.0

  (is (thrown? Exception
               (classification-accuracy [0 0] ["0" "0"])))

  (is (= 1.0
       (classification-accuracy (ds/new-column :x [0 0])
                                (ds/new-column :x  [0.0 0.0]))))
  (is (= 1.0
         (classification-accuracy (ds/new-column :x [0 0]) [0.0 0.0]))))

(ns scicloj.metamorph.metric-test
  (:require  [clojure.test :refer [is deftest]]
             [scicloj.metamorph.ml.metrics :as m]
             [tech.v3.datatype.functional :as fun]))


(def actual [1 1 1 1 1 1 1 1 0 0 0 0])
(def prediction [0 0 1 1 1 1 1 1 1 0 0 0])

(deftest test-cm-metrices

  (is (= 6.0 (fun/sum (m/true-positives actual prediction))))
  (is (= 2.0 (fun/sum (m/false-negatives actual prediction))))
  (is (= 1.0 (fun/sum (m/false-positives actual prediction))))
  (is (= 3.0 (fun/sum (m/true-negatives actual prediction)))))


(comment
  ;;  not sure what this does
  (m/roc-curve [1 1 1 1 1 1 1 1 0 0 0 0]
               [0.1 0.2 0.8 0.9 0.8 0.7 0.8 0.9 0.6 0.8 0.8 0.8]))

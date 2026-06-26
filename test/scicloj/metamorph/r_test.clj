(ns scicloj.metamorph.r-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.ml.r :as r]))


(deftest pretty 

  (is (= [1 1.5 2 2.5 3 3.5 4 4.5 5]
         (r/pretty [1 5] {:n 7} :ocpu)
         ))

  (is (= [1 1.5 2 2.5 3 3.5 4 4.5 5]
         (r/pretty [1 5] {:n 7} :renjin)))

  (is (= [1 1.5 2 2.5 3 3.5 4 4.5 5]
         (r/pretty [1 5] {:n 7} :clojisr)))


  (is (= [1 2 3 4 5 6]
         (r/pretty [1.1 5.2] {} :ocpu)))

  (is (= [1 2 3 4 5 6]
         (r/pretty [1.1 5.2] {} :renjin)))

  (is (= [1 2 3 4 5 6]
         (r/pretty [1.1 5.2] {} :clojisr))))



(ns scicloj.metamorph.csr-test
  (:require  [clojure.test :refer [deftest is]] 
             [scicloj.metamorph.ml.csr :as csr]
             [tech.v3.tensor :as t]))


(deftest ->csr
  (is (=
       {:values [5 8 3 6], :column-indices [0 1 2 1], :row-pointers [0 1 2 3 4]}
       (csr/->csr
        [[0 0 5]
         [1 1 8]
         [2 2 3]
         [4 1 6]]))))

(deftest ->dense
  (is (=
       [[10 20 0 0 0 0]
        [0 30 0 40 0 0]
        [0 0 50 60 70 0]
        [0 0 0 0 0 80]]

       (csr/->dense {:values [10 20 30 40 50 60 70 80]
                     :column-indices [0  1  1  3  2  3  4  5]
                     :row-pointers [0  2  4  7  8]}
                    4 6)))

  (is (=
       '((5 0 0 0) (0 8 0 0) (0 0 3 0) (0 6 0 0))
       (->
        [[0 0 5]
         [1 1 8]
         [2 2 3]
         [4 1 6]]
        (csr/->csr)
        (csr/->dense 4 4)))))

(comment
  (t/->tensor
   (csr/->dense
    (csr/->csr
     [[0 0 5]
      [1 1 8]
      [2 2 3]
      [4 1 6]])
    4 4)))
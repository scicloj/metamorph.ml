(ns scicloj.metamorph.r-model-matrix-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.common]
   [scicloj.metamorph.ml.r-model-matrix :as model-matrix]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]
   [tablecloth.column.api]))

(deftest model-matrix--ocpu
  (is (=
       [{"(Intercept)" 1, "as.factor(cyl)6" 1, "as.factor(cyl)8" 0, "disp" 160.0, "gear" 4}
        {"(Intercept)" 1, "as.factor(cyl)6" 1, "as.factor(cyl)8" 0, "disp" 160.0, "gear" 4}
        {"(Intercept)" 1, "as.factor(cyl)6" 0, "as.factor(cyl)8" 0, "disp" 108.0, "gear" 4}
        {"(Intercept)" 1, "as.factor(cyl)6" 1, "as.factor(cyl)8" 0, "disp" 258.0, "gear" 3}
        {"(Intercept)" 1, "as.factor(cyl)6" 0, "as.factor(cyl)8" 1, "disp" 360.0, "gear" 3}]


       (->
        (model-matrix/r-model-matrix (rdatasets/datasets-mtcars)
                               "~as.factor(cyl)+disp+gear"
                                     :ocpu)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))

(deftest model-matrix--renjine
  (is (=
       [{"X.Intercept." 1.0, "as.factor.cyl.6" 1.0, "as.factor.cyl.8" 0.0, "disp" 160.0, "gear" 4.0}
        {"X.Intercept." 1.0, "as.factor.cyl.6" 1.0, "as.factor.cyl.8" 0.0, "disp" 160.0, "gear" 4.0}
        {"X.Intercept." 1.0, "as.factor.cyl.6" 0.0, "as.factor.cyl.8" 0.0, "disp" 108.0, "gear" 4.0}
        {"X.Intercept." 1.0, "as.factor.cyl.6" 1.0, "as.factor.cyl.8" 0.0, "disp" 258.0, "gear" 3.0}
        {"X.Intercept." 1.0, "as.factor.cyl.6" 0.0, "as.factor.cyl.8" 1.0, "disp" 360.0, "gear" 3.0}]


       (->
        (model-matrix/r-model-matrix (rdatasets/datasets-mtcars)
                                  "~as.factor(cyl)+disp+gear"
                                     :renjine)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))


(deftest model-matrix--clojisr
  (is (=
       [{"(Intercept)" 1.0, :$row.names "1", "as.factor(cyl)6" 1.0, "as.factor(cyl)8" 0.0, "disp" 160.0, "gear" 4.0}
        {"(Intercept)" 1.0, :$row.names "2", "as.factor(cyl)6" 1.0, "as.factor(cyl)8" 0.0, "disp" 160.0, "gear" 4.0}
        {"(Intercept)" 1.0, :$row.names "3", "as.factor(cyl)6" 0.0, "as.factor(cyl)8" 0.0, "disp" 108.0, "gear" 4.0}
        {"(Intercept)" 1.0, :$row.names "4", "as.factor(cyl)6" 1.0, "as.factor(cyl)8" 0.0, "disp" 258.0, "gear" 3.0}
        {"(Intercept)" 1.0, :$row.names "5", "as.factor(cyl)6" 0.0, "as.factor(cyl)8" 1.0, "disp" 360.0, "gear" 3.0}]
       (->
        (model-matrix/r-model-matrix (rdatasets/datasets-mtcars)
                                     "~as.factor(cyl)+disp+gear"
                                     :clojisr)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))
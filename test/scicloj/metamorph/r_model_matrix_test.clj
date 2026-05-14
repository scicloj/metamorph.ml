(ns scicloj.metamorph.r-model-matrix-test
  (:require
   [clojure.test :as t :refer [deftest is]]
   [scicloj.metamorph.common]
   [scicloj.metamorph.ml.r-model-matrix :as model-matrix]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml.regression]
   [tablecloth.column.api]
   [tech.v3.dataset.modelling :as ds-mod]))

(def mpg-ds 
  (-> (rdatasets/datasets-mtcars)
      (ds-mod/set-inference-target :mpg)
      ))

(deftest model-matrix--ocpu
  (is (=
       [{"(Intercept)" 1 :mpg 21.0 "as.factor(cyl)6" 1 "as.factor(cyl)8" 0 "disp" 160.0 "gear" 4}
        {"(Intercept)" 1 :mpg 21.0 "as.factor(cyl)6" 1 "as.factor(cyl)8" 0 "disp" 160.0 "gear" 4}
        {"(Intercept)" 1 :mpg 22.8 "as.factor(cyl)6" 0 "as.factor(cyl)8" 0 "disp" 108.0 "gear" 4}
        {"(Intercept)" 1 :mpg 21.4 "as.factor(cyl)6" 1 "as.factor(cyl)8" 0 "disp" 258.0 "gear" 3}
        {"(Intercept)" 1 :mpg 18.7 "as.factor(cyl)6" 0 "as.factor(cyl)8" 1 "disp" 360.0 "gear" 3}]


       (->
        (model-matrix/r-model-matrix mpg-ds
                               "~as.factor(cyl)+disp+gear"
                                     :ocpu)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))

(deftest model-matrix--renjine
  (is (=
       [{:mpg 21.0 "X.Intercept." 1.0 "as.factor.cyl.6" 1.0 "as.factor.cyl.8" 0.0 "disp" 160.0 "gear" 4.0}
        {:mpg 21.0 "X.Intercept." 1.0 "as.factor.cyl.6" 1.0 "as.factor.cyl.8" 0.0 "disp" 160.0 "gear" 4.0}
        {:mpg 22.8 "X.Intercept." 1.0 "as.factor.cyl.6" 0.0 "as.factor.cyl.8" 0.0 "disp" 108.0 "gear" 4.0}
        {:mpg 21.4 "X.Intercept." 1.0 "as.factor.cyl.6" 1.0 "as.factor.cyl.8" 0.0 "disp" 258.0 "gear" 3.0}
        {:mpg 18.7 "X.Intercept." 1.0 "as.factor.cyl.6" 0.0 "as.factor.cyl.8" 1.0 "disp" 360.0 "gear" 3.0}]


       (->
        (model-matrix/r-model-matrix mpg-ds
                                  "~as.factor(cyl)+disp+gear"
                                     :renjin)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))


(deftest model-matrix--clojisr
  (is (=
       [{"(Intercept)" 1.0 :$row.names "1" :mpg 21.0 "as.factor(cyl)6" 1.0 "as.factor(cyl)8" 0.0 "disp" 160.0 "gear" 4.0}
        {"(Intercept)" 1.0 :$row.names "2" :mpg 21.0 "as.factor(cyl)6" 1.0 "as.factor(cyl)8" 0.0 "disp" 160.0 "gear" 4.0}
        {"(Intercept)" 1.0 :$row.names "3" :mpg 22.8 "as.factor(cyl)6" 0.0 "as.factor(cyl)8" 0.0 "disp" 108.0 "gear" 4.0}
        {"(Intercept)" 1.0 :$row.names "4" :mpg 21.4 "as.factor(cyl)6" 1.0 "as.factor(cyl)8" 0.0 "disp" 258.0 "gear" 3.0}
        {"(Intercept)" 1.0 :$row.names "5" :mpg 18.7 "as.factor(cyl)6" 0.0  "as.factor(cyl)8" 1.0 "disp" 360.0"gear" 3.0}]
       (->
        (model-matrix/r-model-matrix mpg-ds
                                     "~as.factor(cyl)+disp+gear"
                                     :clojisr)
        :model-matrix-dataset
        (tc/head 5)
        (tc/rows :as-maps)))))


(defn verify--lm [formula-impl]
  (=
   {"Intercept" 28.79765746432728
    "as.factor(cyl)6" -4.792830146567762
    "as.factor(cyl)8" -4.803708279733735
    "disp" -0.026725355438243094
    "gear" 0.16519203237173752}

   (-> (rdatasets/datasets-mtcars)
       (model-matrix/lm "~ as.factor(cyl)+disp+gear" :mpg formula-impl)
       :model-data

       ((fn [result]
          (zipmap
           (:names result)
           (map :estimate (:coefficients result))))))))

  (deftest lm
    (is (verify--lm :ocpu))
    (is (verify--lm :clojisr))

    (is (=
         {"Intercept" 28.79765746432728
          "as.factor.cyl.6" -4.792830146567762
          "as.factor.cyl.8" -4.803708279733735
          "disp" -0.026725355438243094
          "gear" 0.16519203237173752}

         (-> (rdatasets/datasets-mtcars)
             (model-matrix/lm "~ as.factor(cyl)+disp+gear" :mpg :renjin)
             :model-data

             ((fn [result]
                (zipmap
                 (:names result)
                 (map :estimate (:coefficients result)))))))))


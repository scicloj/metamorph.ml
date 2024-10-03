(ns scicloj.metamorph.ml.logistic-regression-test
  (:require 
   [clojure.test :refer [is deftest]]
   [scicloj.metamorph.ml.logistic-regression :as log-reg]
   [fastmath.vector :as v]
   [tech.v3.dataset.column-filters :as cf]
   [scicloj.metamorph.ml.toydata :as data]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [scicloj.metamorph.ml.loss :as loss]))


(def data [[4.0 1] [1.75 0] [4.25 1] [2.75 1] [5.0 1] [0.5 0] [1.0 0] [1.5 0]
           [5.5 1] [2.5 0] [2.0 0] [3.5 0] [1.75 1] [3.0 0] [4.75 1] [1.25 0]
           [4.5 1] [0.75 0] [3.25 1] [2.25 1]])

(deftest fit
  
  (let [alpha 0.1
        lambda 0
        iters 10000
        log-reg-fit
        (-> (log-reg/make-logistic-regression alpha lambda iters)
            (log-reg/regression-fit data))]

    (is (v/delta-eq 
         [ 0.8744474608195764 0.19083657134699333 0.9102776017566352]
         (log-reg/regression-predict log-reg-fit (take 3 (map butlast data)))))))

(def iris-split
  (-> (data/breast-cancer-ds)
      (ds-mod/train-test-split 
       )))


(def log-reg-fit
  (let [alpha 0.1
        lambda 0.1
        iters 200]
    
    (-> (log-reg/make-logistic-regression alpha lambda iters)
        (log-reg/regression-fit
         (ds/rowvecs (cf/feature (:train-ds iris-split)))
         (:class (cf/target (:train-ds iris-split)))))))

  
(def prediction
  (->>
   (log-reg/regression-predict
    log-reg-fit
    (ds/rowvecs (cf/feature (:test-ds iris-split))))

   (map
    #(if (> 0.5 %)
       0 1)
    )
   vec))

(def trueth
  (-> iris-split 
      :test-ds
      (ds/assoc-metadata [ :class] :categorical-map nil)
       :class
      vec
      ))

(loss/classification-accuracy prediction trueth)

(count trueth)
(count prediction)
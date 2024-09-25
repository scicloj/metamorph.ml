(ns roc-curve
  (:require
     [nextjournal.clerk :as clerk]
     [scicloj.metamorph.core :as mm]
     [scicloj.metamorph.ml :as ml]
     [scicloj.metamorph.ml.learning-curve :as lc]
     [scicloj.metamorph.ml.viz :as ml-viz]
     [scicloj.metamorph.ml.viz.learning-curve :as mlviz.lc]
   [scicloj.metamorph.ml.metrics :as m]
   [nextjournal.clerk.viewer]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml.loss]
   [scicloj.ml.smile.classification]
   [tablecloth.pipeline :as tc-mm]
   [tech.v3.dataset]
   [tech.v3.dataset.metamorph :as mds]))

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/serve! {:browse true})
  (nextjournal.clerk/show! "notebooks/roc_curve.clj"))

  

(def roc-curve-vals-1
  (m/roc-curve [1 1 1 1 1 1 1 1 0 0 0 0]
               [0.1 0.2 0.8 0.9 0.8 0.7 0.8 0.9 0.6 0.8 0.8 0.8]))
(def roc-curve-vals-2
  (m/roc-curve [1 1 1 1 1 1 1 1 0 0 0 0]
               [1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0]))

(def roc-curve
  (m/roc-curve
   [1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
    1, 0, 0]
   [ 0.64486643, -0.85075125,  0.48293928, -1.27476131, -0.55330807,
    0.12646559 ,  1.14070893 , -0.86382715 , -0.328133 ,  0.11486231 ,
    1.58035975 , -0.62832807 , -0.73370502 , -0.35207723 , -0.89652696 ,
    0.65648549 , -0.55407118 , -0.66486087 ,  0.62261375 ,  1.20936176 ,
    0.50380356 , -1.09923271 ,  0.24798007 , -0.6477504 , -0.44679984]))


(def roc-curve-vals
  (concat [{:tprs 0.0
            :fprs 0.0}]
            
          (map
           #(hash-map :fprs (first %)
                      :tprs (second %))
           roc-curve)
          [{:fprs 1.0
            :tprs 1.0}]))

(clerk/vl
 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :data {:values roc-curve-vals}
  :description "Stock prices of 5 Tech Companies over Time."
  :encoding { :color {:value "#ff0000"}
             :x {:field "fprs" :type "quantitative"}
             :y {:field "tprs" :type "quantitative"}}
  :mark :point})
  

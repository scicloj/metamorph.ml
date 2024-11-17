(ns confusionmatrix
  (:require
   [nextjournal.clerk :as clerk]
   [scicloj.metamorph.ml.viz :as viz]))

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/build! { :paths ["notebooks/confusionmatrix.clj"]})
  (nextjournal.clerk/show! "notebooks/confusionmatrix.clj")

  (nextjournal.clerk/serve! {:browse false
                             }))




(def labels           ["cat", "ant", "cat", "cat", "ant", "bird" "bird" "cat" "ant" "cat" "ant" "ant"])
(def predicted-labels ["ant", "ant", "cat", "cat", "ant", "cat"  "cat"  "ant" "ant" "ant" "cat" "bird"])




(clerk/vl
 (viz/confusion-matrix predicted-labels labels
          {:normalize :all}
          {:HEIGHT 200
           :WIDTH 200
           :CSCALE {:scheme "greys"}}))




(clerk/vl
 (viz/confusion-matrix predicted-labels labels))


(clerk/vl
 (viz/confusion-matrix (repeatedly 1000 #(rand-int 30))
                       (repeatedly 1000 #(rand-int 30))))

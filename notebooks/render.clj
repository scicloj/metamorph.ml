(ns render
  (:require
   [scicloj.clay.v2.api :as clay]))



(clay/make! {:format [:quarto :html]
             :base-source-path "notebooks/"
             :source-path ["austen.clj"
                           "confusionmatrix.clj"
                             ;"learning_curve.clj"
                           "roc_curve.clj"
                           "tidytext.clj"]
             :base-target-path "docs"
             :book {:title "metamorph.ml topics"}
               ;; Empty the target directory first:
             :clean-up-target-dir true
             :show false})
(System/exit 0)
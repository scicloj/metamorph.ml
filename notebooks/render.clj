(ns render
  (:require
   [nextjournal.clerk :as clerk]
   [scicloj.clay.v2.api :as clay]))


(clerk/build! {:paths ["notebooks/confusionmatrix.clj"
                       "notebooks/roc_curve.clj"
                       ]
               :package :single-file
               :out-path "docs/clerk"
               })

(clay/make! {:format [:quarto :html]
             :base-source-path "notebooks/"
             :source-path ["austen.clj"
                           ;"confusionmatrix.clj"
                           ;"roc_curve.clj"
                           "clerk_link.clj"
                           "tidytext.clj"]
             :base-target-path "docs"
             :book {:title "metamorph.ml topics"}
               ;; Empty the target directory first:
             :clean-up-target-dir false
             :show false})
(System/exit 0)
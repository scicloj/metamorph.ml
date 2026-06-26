(ns render
  (:require
   [nextjournal.clerk :as clerk]
   [scicloj.clay.v2.api :as clay]))


(clerk/build! {:paths ["notebooks/confusionmatrix.clj"
                       "notebooks/roc_curve.clj"]
               :package :single-file
               :out-path "docs/clerk"})

(clay/make! {:format [:quarto :html]
             :base-source-path "notebooks/"
             :source-path ["supervised-ml-intro.clj"
                           "unsupervised-ml-intro.clj"
                           "clerk_link.clj"]
             :base-target-path "docs"
             :book {:title "metamorph.ml topics"}
             :clean-up-target-dir false
             :show false})

(clay/make! {:format [:html]
             :base-source-path "notebooks/"
             :source-path ["plot_lm.clj"]
             :base-target-path "docs/plotlm"
             :clean-up-target-dir false
             :show false})

(System/exit 0)
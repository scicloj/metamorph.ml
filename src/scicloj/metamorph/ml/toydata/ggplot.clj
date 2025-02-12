(ns scicloj.metamorph.ml.toydata.ggplot
  "Deprecated ns. Use scicloj.metamorph.ml.rdatasets instead"
  {:deprecated "1.1"}
  (:require [tech.v3.dataset]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            ))

(def diamonds
  (rdatasets/ggplot2-diamonds))

(def ecomonics
  (rdatasets/ggplot2-economics))

(def ecomonics_long
  (rdatasets/ggplot2-economics_long))

(def faithfuld
  (rdatasets/ggplot2-faithfuld))

(def luv_colours
  (rdatasets/ggplot2-luv_colours))

(def midwest
  (rdatasets/ggplot2-midwest))

(def mpg
  (rdatasets/ggplot2-mpg))

(def msleep
  (rdatasets/ggplot2-msleep))

(def presidential
  (rdatasets/ggplot2-presidential))

(def seals
  (rdatasets/ggplot2-seals))

(def txhousing
  (rdatasets/ggplot2-txhousing))

^:kindly/hide-code
(ns plot-lm
  (:require [clojisr.v1.r :as r]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.r-model-matrix :as r-mm]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.metamorph.ml.regression]
            [scicloj.plotje.api :as pj]
            [scicloj.plotje.impl.scale]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset.modelling :as ds-mod]
            [wadogo.scale :as ws]
            [wadogo.scale :as s]
            [scicloj.metamorph.linear-regression-test :refer [diagnostic-plots]]
            [scicloj.metamorph.ml.r :as ml-r]
            ))




^:kindly/hide-code
(defmethod scicloj.plotje.impl.scale/make-scale :categorical [domain pixel-range scale-spec]
   (ws/scale :bands {:domain domain
                     :range pixel-range
                     :ticks (:n-ticks scale-spec)}))


^:kindly/hide-code
^:kindly/hide-code
(comment
  ;; reproduces plots in https://github.com/scicloj/plotje/issues/16
  ;; the plots have now identical xy points then R: plot(lm(mtcars))
  (def formula "mpg ~ .")
  (def r-dataset-name "mtcars")
  (diagnostic-plots
   (rdatasets/datasets-mtcars)
   formula))

^:kindly/hide-code
(defn plot-lm->pdf! [r-dataset-name formula id-n]
  (r/r "library('svglite')")
  (r/r (format "svglite('/tmp/plot_lm_%s_%%03d.svg',width = 7,height = 7)" r-dataset-name))
  (r/r (format "plot(lm(%s,%s),which=c(1,2,3,4,5,6),id.n=%s)" formula r-dataset-name id-n))
  (r/r "dev.off()"))

^:kindly/hide-code
(defn svg->hiccup [filename]
  (kind/html
   (->
    (slurp (format "/tmp/%s" filename)))))






^:kindly/hide-code
(defn compare-table [dataset-name formula]
  
  (let [opts   {:n-labeled-points 5
                :pretty-fn (fn [s]
                             (ml-r/pretty (seq s) {} :ocpu)
                             )}
        
        _ (plot-lm->pdf! dataset-name "" (:n-labeled-points opts))
        
        metamorph-plots
        (diagnostic-plots
         (eval (list (symbol (format "rdatasets/datasets-%s" dataset-name))))
         formula
         opts


         )]
    (kind/table {:column-names [(kind/code "R")
                                (kind/code "Clojure")]
                 :row-vectors
                 [[(svg->hiccup (format "plot_lm_%s_001.svg" dataset-name))
                   (metamorph-plots :residual-vs-fitted)]

                  [(svg->hiccup (format "plot_lm_%s_002.svg" dataset-name))
                   (metamorph-plots :residual-q-q)]

                  [(svg->hiccup (format "plot_lm_%s_003.svg" dataset-name))
                   (metamorph-plots :scale-location)]

                  [(svg->hiccup (format "plot_lm_%s_004.svg" dataset-name))
                   (metamorph-plots :cooks-distance)]

                  [(svg->hiccup (format "plot_lm_%s_005.svg" dataset-name))
                   (metamorph-plots :residual-vs-leverage)]

                  [(svg->hiccup (format "plot_lm_%s_006.svg" dataset-name))
                   (metamorph-plots :cooks-d-vs-leverage*)]]})))




(pj/set-config! 
 {:width 700
  :height 700})




; # mtcars
^:kindly/hide-code
(compare-table "mtcars" "mpg ~ ." )


; # iris
^:kindly/hide-code
(compare-table "iris" "`sepal-length` ~ ." )

; # rock
^:kindly/hide-code
(compare-table "rock" "area ~ ." )

; # ToothGrowth
^:kindly/hide-code
(compare-table "ToothGrowth" "len ~ ." )

; # airquality
^:kindly/hide-code
(compare-table "airquality" "ozone ~ ." )


 







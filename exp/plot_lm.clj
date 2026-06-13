(ns plot-lm
  (:require [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.r-model-matrix :as r-mm]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [tech.v3.dataset.modelling :as ds-mod]
            [clojisr.v1.r :as r]
            [scicloj.clay.v2.api :as clay]
            [scicloj.kindly.v4.kind :as kind]
            [scicloj.metamorph.ml.regression]
            [wadogo.scale :as ws]
            [hickory.core :as hick]
            [scicloj.plotje.impl.scale]
            ))

(defmethod scicloj.plotje.impl.scale/make-scale :categorical [domain pixel-range scale-spec]
   (ws/scale :bands {:domain domain
                     :range pixel-range
                     :ticks (:n-ticks scale-spec)}))


^:kindly/hide-code
(defn- diagnostic-plots [dataset formula & {:as opts}]
  (let [row-names (:rownames dataset)
        
        model-matrix
        (->
         dataset
         (tc/drop-columns [:rownames])
         (r-mm/r-model-matrix formula :ocpu)
         :model-matrix-dataset)

        inference-target (second (tc/column-names dataset))
        modelled-dataset
        (->
         model-matrix
         (tc/drop-columns ["(Intercept)"])
         (tc/add-column inference-target (get dataset inference-target))
         (ds-mod/set-inference-target inference-target))
        
        plot-opts (assoc opts :rownames row-names)


        model (ml/train modelled-dataset {:model-type :fastmath/ols})
        ;poses (ml/plot model dataset {:pretty-cooks-d-levels-plot-6 [0.0 0.5 1.0 1.5 2.0]})   ;if we want "the same" plot then R: plot(lm(mtcars))
        poses (ml/plot model modelled-dataset plot-opts)    ;; plot 6 produces different cook's d lines, as "pretty" function is not available for clojure
        ]
    poses

    ;; (pj/arrange (map val poses)
    ;;             {:cols 1
    ;;              :height (* 400 (count poses))}
    ;;             )
    ))
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
(defn plot-lm->pdf [r-dataset-name formula]
  (r/r "library('svglite')")
  (r/r (format "svglite('data/plot_lm_%s_%%03d.svg',width = 7,height = 7)" r-dataset-name))
  (r/r (format "plot(lm(%s,%s),which=c(1,2,3,4,5,6))" formula r-dataset-name))
  (r/r "dev.off()"))

^:kindly/hide-code
(defn svg->hiccup [filename]
  (kind/html
   (->
    (slurp (format "data/%s" filename)))))




^:kindly/hide-code
(defn compare-table [dataset-name formula]
  (plot-lm->pdf dataset-name "")
  (let [metamorph-plots
        (diagnostic-plots
         (eval (list (symbol (format "rdatasets/datasets-%s" dataset-name))))
         formula)]
    (kind/table {:column-names [(kind/code ":x")
                                (kind/code ":y")]
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
 {:width 600
  :height 600})

; # mtcars
^:kindly/hide-code
(compare-table "mtcars" "mpg ~ .")

; # iris
^:kindly/hide-code
(compare-table "iris" "`sepal-length` ~ .")

; # rock
^:kindly/hide-code
(compare-table "rock" "area ~ .")

; # ToothGrowth
^:kindly/hide-code
(compare-table "ToothGrowth" "len ~ .")

 
;; ; # marketing
;; (diagnostic-plots
;;  (tc/dataset "https://raw.githubusercontent.com/prasertcbs/basic-dataset/refs/heads/master/marketing.csv")
;;  "sales ~ youtube",
;;  {:pretty-cooks-d-levels-plot-6 [0 1 2 3]})







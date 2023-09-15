(ns scicloj.metamorph.ml.viz
  (:require
   [aerial.hanami.common :as hc]
   [aerial.hanami.templates :as ht]
   [clojure.math :as math]
   [scicloj.metamorph.ml.learning-curve]
   [scicloj.metamorph.ml.loss :as loss]
   [tablecloth.api :as tc]
   [tech.v3.datatype.functional :as fun]))



(def errorband-encoding-train
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-train-min"
    :X "train-ds-size"
    :XTYPE "quantitative"}
   :color {:value :TEST-COLOR}
   :y2 {:field "metric-train-max"
        :legend nil}))

(def errorband-encoding-test
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-test-min"
    :X "train-ds-size"
    :XTYPE "quantitative"}
   :color {:value :TRAIN-COLOR}
   :y2 {:field "metric-test-max"
        :legend nil}))

(def metric-encoding
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:X "train-ds-size"
    :XTYPE "quantitative"
    :Y "metric"
    :YTYPE "quantitative"}
   :color {:field "train-test-metric"
           :type "nominal"
           :scale {:range [:TRAIN-COLOR :TEST-COLOR]}
           :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' :
                                 datum.label == 'metric-train' ? 'Training score' : ''  "}}))


(def _learning-curve-spec
  (assoc ht/layer-chart
         :aerial.hanami.templates/defaults
         {:TITLE "Learning Curve"
          :XTITLE "Training size"
          :YTITLE "metric"}
          
         :layer [{:mark :errorband
                  :encoding errorband-encoding-train}
                 {:mark :errorband
                  :encoding errorband-encoding-test}
                 (assoc ht/line-layer
                        :aerial.hanami.templates/defaults
                        {:POINT true
                         :ENCODING metric-encoding})]))

(defn- rounded-mean [coll]
  (math/round (fun/mean coll)))


(defn- mean+std [col]
  (+
   (fun/mean col)
   (fun/standard-deviation col)))

(defn- mean-std [col]
  (-
   (fun/mean col)
   (fun/standard-deviation col)))

(defn apply-xform-kvs [spec kvs]
  (apply hc/xform spec (into [] cat kvs)))


(defn learning-curve-vl-data [lc-rf]
  (-> lc-rf
            (tc/group-by :train-size-index)

            (tc/aggregate {:metric-test      #(fun/mean (:metric-test %))
                           :metric-test-min  #(mean-std (:metric-test %))
                           :metric-test-max  #(mean+std (:metric-test %))
                           :metric-train     #(fun/mean (:metric-train %))
                           :metric-train-min #(mean-std (:metric-train %))
                           :metric-train-max #(mean+std (:metric-train %))
                           :train-ds-size    #(rounded-mean (:train-ds-size %))
                           :test-ds-size     #(rounded-mean (:test-ds-size %))})))

(defn learning-curve-spec [lc-vl-data]
  (assoc _learning-curve-spec
         :aerial.hanami.templates/defaults {
                                            :TRAIN-COLOR "blue"
                                            :TEST-COLOR "orange"
                                            :VALDATA
                                            (-> lc-vl-data
                                                (tc/pivot->longer [:metric-test :metric-train]
                                                                  {:value-column-name :metric
                                                                   :target-columns :train-test-metric})
                                                (tc/rows :as-maps))}))


(defn learning-curve
  "Generates a learnining curve plot.

  The functions splits  the dataset  in a fixed size test set
  and increasingly larger  training sets. A model is trained at each
  step and evaluated.

  Returns a vega lite spec of the learninig curve plot.

  `dataset` The TMD dataset to use
  `pipe-fn` The metamorph pipleine to use for learning
  `train-sizes` vector of double from 0 to 1, controlling the sizes of the training data.
  `lc-opts` Options to create the learnining curve data
     `k` At each step a k cross-validation is done
     `metric-fn` the metric to use for evaluation the model
     `loss-or-accuracy`   If the metric-fn calculates :loss or :accuracy
     Defaults are k=3 and `:accuracy`
  `hanami-kvs` Hanami substitution keys passed to hanami to control the learninig curve plot.
     Standard keys are allowed and additionaly the plot specific keys:
     `:TRAIN-COLOR`  Color of the train curve
     `:TEST-COLOR`   Color of the test curve
  "
  ([dataset pipe-fn
    lc-opts hanami-kvs]
   (->
    (scicloj.metamorph.ml.learning-curve/learning-curve
     dataset
     pipe-fn
     (:train-sizes lc-opts) lc-opts)
    (learning-curve-vl-data)
    (learning-curve-spec)
    (apply-xform-kvs hanami-kvs)))
  ([dataset pipe-fn]
   (learning-curve dataset pipe-fn
                     {:train-sizes [0.1 0.325 0.55 0.775 1]
                      :k 3
                      :metric-fn loss/classification-accuracy
                      :loss-or-accuracy :accuracy}
                     {})))



(def residual-plot-chart
  (assoc ht/layer-chart
         :aerial.hanami.templates/defaults
         {:XSCALE {:zero false} :YSCALE {:zero false}
          :RESIDUAL hc/RMV
          :LINE hc/RMV
          :POINTS ht/point-layer}
         ;; :STANDARD (hc/xform ht/line-chart :DATA :REG-DATA :MCOLOR "blue")

         :encoding ht/xy-encoding
         :layer [:RESIDUAL :POINTS :LINE])) ;; :STANDARD



(def residual-rule
  {:mark (merge ht/mark-base {:type "rule"
                              :strokeWidth :STROKEWIDTH})
   :encoding (assoc ht/xy-encoding

                    :y2 {:field :Y2 :type :Y2TYPE :bin :Y2BIN
                         :axis :Y2AXIS :scale :Y2SCALE :sort :Y2SORT
                         :aggregate :Y2AGG})
   :aerial.hanami.templates/defaults
   {:MCOLOR :RESIDUAL-COLOR
    :XSCALE {:zero false} :YSCALE {:zero false} :Y2SCALE {:zero false}
    :Y2 :prediction :Y2TYPE :quantitative
    :Y2BIN hc/RMV :Y2AXIS hc/RMV :Y2SORT hc/RMV :Y2AGG hc/RMV}})

;; (defn residual-plot-spec [data]
;;   (assoc residual-plot-chart
;;          :aerial.hanami.templates/defaults))
         

(def my-line-chart
  (assoc ht/view-base
         :mark (merge  ht/mark-base {:type "line"
                                     :strokeWidth :STROKEWIDTH})))

(defn residual-plot-spec []
  (assoc-in residual-plot-chart
            [:aerial.hanami.templates/defaults]
            (merge (residual-plot-chart :aerial.hanami.templates/defaults)
                   {:LINE-COLOR "blue"
                    :RESIDUAL-COLOR "red"
                    :STROKEWIDTH ht/RMV
                    :RESIDUAL residual-rule
                    :LINE (hc/xform  my-line-chart :MCOLOR :LINE-COLOR)}))) ;;



(defn residual-plot [dataset hanami-opts]
  (->
   (residual-plot-spec)
   (apply-xform-kvs (assoc
                     hanami-opts
                     :DATA (tc/rows dataset :as-maps)))))

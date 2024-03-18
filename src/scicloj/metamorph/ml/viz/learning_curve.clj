(ns scicloj.metamorph.ml.viz.learning-curve
  (:require
     [tablecloth.api :as tc]
     [clojure.math :as math]
     [tech.v3.datatype.functional :as fun]
     [scicloj.metamorph.ml.learning-curve]
     [aerial.hanami.templates :as ht]
     [aerial.hanami.common :as hc]))

(def errorband-encoding-train
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-train-min"
    :X "train-ds-size"
    :XTYPE "quantitative"
    :TEST-COLOR "orange"}
   :color {:value :TEST-COLOR}
   :y2 {:field "metric-train-max"
        :legend nil}))

(def errorband-encoding-test
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-test-min"
    :X "train-ds-size"
    :XTYPE "quantitative"
    :TRAIN-COLOR "blue"}

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
    :YTYPE "quantitative"
    :TRAIN-COLOR "blue"
        :TEST-COLOR "orange"}
   :color {:field "train-test-metric"
           :type "nominal"
           :scale {:range [:TRAIN-COLOR :TEST-COLOR]}
           :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' : datum.label == 'metric-train' ? 'Training score' : ''  "}}))

(def layer
  [{:mark :errorband
            :encoding errorband-encoding-train}
   {:mark :errorband
    :encoding errorband-encoding-test}
   (assoc ht/line-layer
          :aerial.hanami.templates/defaults
          {:POINT true
           :ENCODING metric-encoding})])

(defn spec [lc-vl-data]
  (assoc ht/layer-chart
         :aerial.hanami.templates/defaults
         {:TITLE "Learning Curve"
          :XTITLE "Training size"
          :YTITLE "metric"
          :LAYER layer
          :VALDATA
          (-> lc-vl-data
              (tc/pivot->longer [:metric-test :metric-train]
                                {:value-column-name :metric
                                 :target-columns :train-test-metric})
              (tc/rows :as-maps))}))


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


(defn vl-data [lc-rf]
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

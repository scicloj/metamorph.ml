(ns scicloj.metamorph.ml.viz
  (:require
            [aerial.hanami.templates :as ht]))



(def errorband-encoding-train
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-train-min"
    :X "train-ds-size"
    :XTYPE "nominal"}
   :color {:value "orange"}
   :y2 {:field "metric-train-max"
        :legend nil}))

(def errorband-encoding-test
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-test-min"
    :X "train-ds-size"
    :XTYPE "nominal"}
   :color {:value "blue"}
   :y2 {:field "metric-test-max"
        :legend nil}))

(def metric-encoding
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:X "train-ds-size"
    :XTYPE "nominal"
    :Y "metric"
    :YTYPE "quantitative"}
   :color {:field "train-test-metric"
           :type "nominal"
           :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' : datum.label == 'metric-train' ? 'Training score' : ''  "}}))


(def learning-curve-spec
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
                        {:ENCODING metric-encoding})]))




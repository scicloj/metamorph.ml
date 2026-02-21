(ns scicloj.metamorph.ml.viz.learning-curve
  (:require
     [tablecloth.api :as tc]
     [clojure.math :as math]
     [tech.v3.datatype.functional :as fun]
     [aerial.hanami.templates :as ht]))
     

(def errorband-encoding-train
  "Hanami encoding specification for training metric error bands.

  Displays the range (mean ± stddev) of training metrics as a blue error band.
  Y-axis spans from `metric-train-min` to `metric-train-max`, X-axis is training
  dataset size."
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-train-min"
    :X "train-ds-size"
    :XTYPE "quantitative"
    :TRAIN-COLOR "blue"}
   :color {:value :TRAIN-COLOR}
   :y2 {:field "metric-train-max"
        :legend nil}))

(def errorband-encoding-test
  "Hanami encoding specification for test metric error bands.

  Displays the range (mean ± stddev) of test/validation metrics as an orange
  error band. Y-axis spans from `metric-test-min` to `metric-test-max`, X-axis
  is training dataset size."
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-test-min"
    :X "train-ds-size"
    :XTYPE "quantitative"
    :TEST-COLOR "orange"}

   :color {:value :TEST-COLOR}
   :y2 {:field "metric-test-max"
        :legend nil}))

(def metric-encoding
  "Hanami encoding specification for metric line plots.

  Displays training and test metrics as separate colored lines (blue for training,
  orange for test). X-axis is training dataset size, Y-axis is the metric value.
  Legend distinguishes between training score and cross-validation metric."
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
           :scale {:range [:TEST-COLOR :TRAIN-COLOR]}
           :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' : datum.label == 'metric-train' ? 'Training score' : ''  "}}))

(def layer
  "Hanami layer specification for learning curve visualization.

  Combines three layers:

  1. Training error band (blue, showing mean ± stddev)
  2. Test error band (orange, showing mean ± stddev)
  3. Line plot with points (showing mean metrics)

  Creates a comprehensive learning curve showing both central tendency and variance."
  [{:mark :errorband
            :encoding errorband-encoding-train}
   {:mark :errorband
    :encoding errorband-encoding-test}
   (assoc ht/line-layer
          :aerial.hanami.templates/defaults
          {:POINT true
           :ENCODING metric-encoding})])

(defn spec
  "Creates a Hanami/Vega-Lite specification for a learning curve chart.

  `lc-vl-data` - Dataset with columns: `:train-ds-size`, `:metric-test`,
  `:metric-train`, `:metric-test-min`, `:metric-test-max`, `:metric-train-min`,
  `:metric-train-max`

  Returns a Hanami layer chart showing:

  * Training and test metrics over varying training set sizes
  * Error bands indicating variance (mean ± standard deviation)
  * Line plots with points for mean metric values

  Use with `vl-data` to prepare data from learning curve results.

  See also: `scicloj.metamorph.ml/learning-curve`"
  [lc-vl-data]
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


(defn vl-data
  "Prepares learning curve data for Vega-Lite visualization.

  `lc-rf` - Raw learning curve results dataset from `scicloj.metamorph.ml/learning-curve`

  Returns an aggregated dataset grouped by `:train-size-index` with columns:

  * `:metric-test`, `:metric-train` - Mean metric values
  * `:metric-test-min`, `:metric-train-min` - Mean minus standard deviation
  * `:metric-test-max`, `:metric-train-max` - Mean plus standard deviation
  * `:train-ds-size`, `:test-ds-size` - Rounded mean dataset sizes

  Use with `spec` to create a learning curve visualization."
  [lc-rf]
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

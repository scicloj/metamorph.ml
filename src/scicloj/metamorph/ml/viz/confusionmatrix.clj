(ns scicloj.metamorph.ml.viz.confusionmatrix
  (:require
   [aerial.hanami.templates :as ht]
   [scicloj.metamorph.ml.classification :as cl]
   [scicloj.metamorph.ml.learning-curve]
   [scicloj.metamorph.ml.viz.learning-curve]))



(def layer
  "Hanami layer specification for confusion matrix visualization.

  Combines a colored rectangle layer (for the heatmap) with a text layer (for
  count values). Uses XY encoding with count values displayed as text overlay."
  [(assoc ht/rect-layer :encoding ht/xy-encoding)
   (assoc ht/text-layer
          :encoding
          (assoc ht/xy-encoding
                 :color nil
                 :text {:field "count" :type "quantitative"}))])


(defn confusion-matrix-chart
  "Creates a Hanami/Vega-Lite specification for a confusion matrix heatmap.

  `values` - Sequence of maps with keys: `:actual`, `:predicted`, `:count`

  Returns a Hanami layer chart with:

  * X-axis: predicted labels (nominal)
  * Y-axis: actual labels (nominal)
  * Color: count values (yellow-orange-red scale)
  * Text overlay: count values

  Use with `cm-values` to generate input data from predictions and labels."
  [values]
  (assoc
   ht/layer-chart

   :aerial.hanami.templates/defaults
   {
    :VALDATA values
    :LAYER layer
    :X "predicted"
    :XTYPE "nominal"
    :Y "actual"
    :YTYPE "nominal"
    :COLOR ht/default-color
    :CFIELD "count"
    :CTYPE "quantitative"
    :CSCALE {:scheme "yelloworangered"}
    :TXT "count"}))
           

(defn cm-values
  "Generates confusion matrix data for visualization.

  `predicted-labels` - Sequence of predicted class labels
  `labels` - Sequence of actual class labels
  `opts` - Options map with optional `:normalize` key (`:none`, `:true`, `:pred`, `:all`)

  Returns a sequence of maps, each containing:

  * `:actual` - Actual class label
  * `:predicted` - Predicted class label
  * `:count` - Count or normalized value from confusion matrix

  Use with `confusion-matrix-chart` to create a visualization.

  See also: `scicloj.metamorph.ml.classification/confusion-map`"
  [predicted-labels labels opts]
  (let [cm (cl/confusion-map
             predicted-labels
             labels
             (get opts :normalize :none))


         distinct-labels
         (distinct
          (concat predicted-labels labels))]


    (for [actual distinct-labels
          prediction distinct-labels]
     (hash-map :actual actual
               :predicted prediction
               :count (-> cm (get actual 0) (get prediction 0))))))

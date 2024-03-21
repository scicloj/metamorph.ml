(ns scicloj.metamorph.ml.viz.confusionmatrix
  (:require
   [aerial.hanami.common :as hc]
   [aerial.hanami.templates :as ht]
   [scicloj.metamorph.ml.classification :as cl]
   [scicloj.metamorph.ml.learning-curve]
   [scicloj.metamorph.ml.viz.learning-curve]))




(def layer
  [(assoc ht/rect-layer :encoding ht/xy-encoding)
   (assoc ht/text-layer
          :encoding
          (assoc ht/xy-encoding
                 :color nil
                 :text {:field "count" :type "quantitative"}))])


(defn confusion-matrix-chart [values]
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
           

(defn cm-values [predicted-labels labels opts]
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

(ns scicloj.metamorph.ml.viz
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
    :XTYPE "quantitative"}
   :color {:value "orange"}
   :y2 {:field "metric-train-max"
        :legend nil}))

(def errorband-encoding-test
  (assoc
   ht/xy-encoding
   :aerial.hanami.templates/defaults
   {:Y "metric-test-min"
    :X "train-ds-size"
    :XTYPE "quantitative"}
   :color {:value "blue"}
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
           :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' : datum.label == 'metric-train' ? 'Training score' : ''  "}}))


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

(defn rounded-mean [coll]
  (math/round (fun/mean coll)))


(defn mean+std [col]
  (+
   (fun/mean col)
   (fun/standard-deviation col)))

(defn mean-std [col]
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
                                            :VALDATA
                                            (-> lc-vl-data
                                                (tc/pivot->longer [:metric-test :metric-train]
                                                                  {:value-column-name :metric
                                                                   :target-columns :train-test-metric})
                                                (tc/rows :as-maps))}))


(defn learning-curve-vl [lc-vl-spec hanami-opts]
  (apply-xform-kvs lc-vl-spec hanami-opts))


(defn learnining-curve
  "Generates a learnining curve.

  The functions splits  the dataset  in a fixed size test set
  and increasingly larger  training sets. A model is trained at each
  step and evaluated.

  `dataset` the TMD dataset to use
  `train-sizes` vector of double from 0 to 1, controlling the sizes of the training data.
  `lc-opts`
     `k` At each step a k cross-validation is done
     `metric-fn` the metric to use for evaluation the model
     `loss-or-accuracy`   If the metric-fn calculates :loss or :accuracy
  `hanami-opts` Options passed to hanami to control the plot
  "
  ([dataset pipe-fn train-sizes
    lc-opts hanami-opts]
   (->
    (scicloj.metamorph.ml.learning-curve/learning-curve
     dataset
     pipe-fn
     train-sizes lc-opts)
    (learning-curve-vl-data)
    (learning-curve-spec)
    (learning-curve-vl hanami-opts)))
  ([dataset pipe-fn]
   (learnining-curve dataset pipe-fn
                     [0.1 0.325 0.55 0.775 1]
                     {} {})))

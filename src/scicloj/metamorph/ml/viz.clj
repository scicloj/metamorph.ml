(ns scicloj.metamorph.ml.viz
  (:require
   [aerial.hanami.common :as hc]
   [scicloj.metamorph.ml.learning-curve]
   [scicloj.metamorph.ml.viz.learning-curve]
   [scicloj.metamorph.ml.viz.confusionmatrix]))

(defn- apply-xform-kvs [spec kvs]
  (apply hc/xform spec (into [] cat kvs)))


(defn learning-curve
  "Generates a learning  curve.

  The functions splits the dataset in a fixed size test set
  and increasingly larger training sets. A model is trained at each
  step and evaluated.

  Returns a vega lite spec of the learninig curve plot.

  `dataset` the TMD dataset to use
  `train-sizes` vector of double from 0 to 1, controlling the sizes of the training data.
  `lc-opts`
     `k` At each step a k cross-validation is done
     `metric-fn` the metric to use for evaluation the model
     `loss-or-accuracy`   If the metric-fn calculates :loss or :accuracy
  `hanami-opts` Options passed to hanami to control the plot. Can be the default hanami
   substituions keys or:
       `TRAIN-COLOR:`   Color used for the train curve (default: blue)
       `TEST-COLOR:`    Color used for the test curve (default: orange)

  "
  ([dataset pipe-fn train-sizes
    lc-opts hanami-opts]
   (->
    (scicloj.metamorph.ml.learning-curve/learning-curve
     dataset
     pipe-fn
     train-sizes lc-opts)
    (scicloj.metamorph.ml.viz.learning-curve/vl-data)
    (scicloj.metamorph.ml.viz.learning-curve/spec)

    (apply-xform-kvs hanami-opts)))
  ([dataset pipe-fn lc-opts]
   (learning-curve dataset pipe-fn
                   [0.1 0.325 0.55 0.775 1]
                   lc-opts
                   {}))
  ([dataset pipe-fn lc-opts hanami-opts]
   (learning-curve dataset pipe-fn
                   [0.1 0.325 0.55 0.775 1]
                   lc-opts
                   hanami-opts)))


(defn confusion-matrix
  "Generates a confusin matrix plot out of `predicted-labels` and `labels`

   `opts`
     `normalize` : Can be :none (default) or :all and decides if the values in the matrix are counts or percentages.

   `hanami-opts` Options passed to hanami to control the plot. Can be any of the default hanami
   substituions keys.

  "
  ([predicted-labels labels
    opts
    hanami-opts]
   (->
    (scicloj.metamorph.ml.viz.confusionmatrix/cm-values predicted-labels labels opts)
    (scicloj.metamorph.ml.viz.confusionmatrix/confusion-matrix-chart)
    (apply-xform-kvs hanami-opts)))
   
  ([predicted-labels labels] (confusion-matrix predicted-labels labels {} {})))

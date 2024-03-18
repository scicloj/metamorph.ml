(ns learning_curve
  (:require
   [nextjournal.clerk :as clerk]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.learning-curve :as lc]
   [scicloj.metamorph.ml.viz :as ml-viz]
   [scicloj.metamorph.ml.viz.learning-curve :as ml-viz-lc]

   [nextjournal.clerk.viewer]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml.loss]
   [scicloj.ml.smile.classification]
   [tablecloth.pipeline :as tc-mm]
   [tech.v3.dataset]
   [tech.v3.dataset.metamorph :as mds]))

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/show! "notebooks/learning_curve.clj")

  (nextjournal.clerk/serve! {:browse true}))

^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/table-viewer
  :nextjournal.clerk/opts {:page-size 5}}
(def titanic-train
  (->
   (tech.v3.dataset/->dataset "https://github.com/scicloj/metamorph-examples/raw/main/data/titanic/train.csv"
                              {:key-fn keyword})
   (tc/shuffle {:seed 1234})))

(defn make-pipe-fn [model-type]
  (mm/pipeline
   (mds/select-columns [:Pclass :Survived :Sex :Embarked :Parch])
   (tc-mm/add-or-replace-column :Survived (fn [ds] (map #(case %
                                                          1 "yes"
                                                          0 "no")
                                                       (:Survived ds))))

   (mds/categorical->number [:Survived :Pclass :Sex :Embarked])
   (mds/drop-missing)
   (mds/set-inference-target :Survived)

   {:metamorph/id :model}
   (scicloj.metamorph.ml/model {:model-type model-type})))


(def train-sizes
  [ 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])


^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/table-viewer};;
(def lc-df
  (lc/learning-curve titanic-train
                     (make-pipe-fn :smile.classification/logistic-regression)
                     train-sizes
                     {:k 3
                      :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
                      :loss-or-accuracy :accuracy}))


(defn render-with-svg [vl-spec]
  (assoc vl-spec
          :usermeta {:embedOptions {:renderer "svg"}}))

^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/table-viewer}
(def lc-vl-data
  (ml-viz-lc/vl-data lc-df))

(def lc-vl-spec
  (ml-viz-lc/spec lc-vl-data))




^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/vega-lite-viewer}
(render-with-svg
 (ml-viz/apply-xform-kvs lc-vl-spec {:TITLE "Learning Curve"
                                     :YTITLE "Accuracy"}))


^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/vega-lite-viewer}
(render-with-svg
 (ml-viz/learning-curve
  titanic-train
  (make-pipe-fn :smile.classification/logistic-regression)
  {:k 3
   :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
   :loss-or-accuracy :accuracy}))





^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/vega-lite-viewer}
(render-with-svg
 (ml-viz/learning-curve
  titanic-train
  (make-pipe-fn :smile.classification/logistic-regression)
  train-sizes
  {:k 3
   :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
   :loss-or-accuracy :accuracy}
  {:YSCALE {:zero false}
   :TITLE "Learning Curve"
   :YTITLE "Accuracy"}))
 


^{:nextjournal.clerk/viewer nextjournal.clerk.viewer/vega-lite-viewer}
(render-with-svg
 (ml-viz/learning-curve
  titanic-train
  (make-pipe-fn :smile.classification/logistic-regression)
  train-sizes
  {:k 3
   :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
   :loss-or-accuracy :accuracy}
  {:YSCALE {:zero false}
   :TRAIN-COLOR "green"
   :TEST-COLOR "red"
   :TITLE "Learning Curve"
   :YTITLE "Accuracy"}))
   

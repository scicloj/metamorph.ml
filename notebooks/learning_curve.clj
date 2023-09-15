(ns learning_curve
  (:require
   [nextjournal.clerk :as clerk]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.learning-curve :as lc]
   [scicloj.metamorph.ml.viz :as ml-viz]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml.loss]
   [scicloj.ml.smile.classification]
   [scicloj.ml.smile.regression]
   [tablecloth.pipeline :as tc-mm]
   [tech.v3.dataset]
   [scicloj.metamorph.ml.toydata :as toydata]
   [tech.v3.dataset.metamorph :as mds]
   [aerial.hanami.common :as hc]
   [aerial.hanami.templates :as ht]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.modelling :as ds-mod]))

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/show! "notebooks/learning_curve.clj")
  (nextjournal.clerk/serve! {:browse true}))

^{:nextjournal.clerk/viewer :table
  :nextjournal.clerk/opts {:page-size 5}}

(def titanic-data
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


(def logistic-regression-pipe-fn (make-pipe-fn :smile.classification/logistic-regression))

^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/learnining-curve
 titanic-data
 logistic-regression-pipe-fn)



(def train-sizes
  [ 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1])



^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/learning-curve
 titanic-data
 (make-pipe-fn :smile.classification/logistic-regression)
 {:train-sizes train-sizes
  :k 3
  :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
  :loss-or-accuracy :accuracy}
 {})


^{:nextjournal.clerk/viewer :table}
(def lc-df
  (lc/learning-curve titanic-data
                     (make-pipe-fn :smile.classification/logistic-regression)
                     train-sizes
                     {:k 3
                      :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
                      :loss-or-accuracy :accuracy}))






^{:nextjournal.clerk/viewer :table}
(def lc-vl-data
  (ml-viz/learning-curve-vl-data lc-df))

(def lc-vl-spec
   (ml-viz/learning-curve-spec lc-vl-data))



^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/apply-xform-kvs lc-vl-spec {:TITLE "Learning Curve"
                                      :YTITLE "Accuracy"})




^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/learning-curve
 titanic-data
 logistic-regression-pipe-fn

 {:train-sizes [ 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
  :k 3
  :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
  :loss-or-accuracy :accuracy}

 {:YSCALE {:zero false}
  :TITLE "Learning Curve"
  :YTITLE "Accuracy"})


^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/learning-curve
 titanic-data
 (make-pipe-fn :smile.classification/logistic-regression)

 {:train-sizes [ 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
  :k 3
  :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
  :loss-or-accuracy :accuracy}

 {:YSCALE {:zero true}
  :TRAIN-COLOR "green"
  :TEST-COLOR "red"
  :XTITLE "Learning Curve"
  :YTITLE "Accuracy"})

(def spec
  (ml-viz/learning-curve

   titanic-data

   logistic-regression-pipe-fn


   {:train-sizes [ 0.04 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1]
    :k 3
    :metric-fn scicloj.metamorph.ml.loss/classification-accuracy
    :loss-or-accuracy :accuracy}

   {:YSCALE {:zero true}
    :TRAIN-COLOR "green"
    :TEST-COLOR "red"
    :TITLE "Learning Curve"
    :XTITLE "Training size"
    :YTITLE "Accuracy"}))


(def text-spec
  {:mark {:type "text"
          :fontSize 170
          :angle 29
          :color "grey"}
   :data {:values
          [{"a" 300, "b" 0.72, "label" "Draft"}]}
   :encoding {:text {:field "label"}
                     
              :x {:field "a"
                  :axis true
                  :type "quantitative"
                  :scale {:domain [0 600]}}

              :y {:field "b"
                  :type "quantitative"
                  :scale {:domain [0.6 0.9]}}}})


^{:nextjournal.clerk/viewer :vega-lite}
text-spec

^{:nextjournal.clerk/viewer :vega-lite}
(-> spec
    ( update-in [:layer] concat [text-spec]))


(ml-viz/residual-plot-spec)




^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/residual-plot
 (tc/dataset
  [
   {:x 1 :y 1   :prediction 2}
   {:x 2 :y 1.5 :prediction 3}
   {:x 3 :y -1  :prediction 1}])
 {:RESIDUE-COLOR "orange"
  :LINE-COLOR "black"
  :STROKEWIDTH 5})


(def diabetes
  (->
   (scicloj.metamorph.ml.toydata/diabetes-ds)))
   ;; (tc/select-columns [:age :disease-progression])



(def pipe-fn
  (mm/pipeline
   {:metamorph/id :model}
   (ml/model {:model-type  :smile.regression/ordinary-least-square})))

(def fitted-context
  (mm/fit-pipe
   diabetes
   pipe-fn))


(def model
  (ml/thaw-model (-> fitted-context :model)))



^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/residual-plot
 (tc/dataset
         {:x (:age diabetes)
          :prediction (:disease-progression diabetes)
          :y (.fittedValues model)})

 {:LINE ht/RMV
  :XTITLE "age"
  :YTITLE "diabetes progression"
  :RESIDUAL-COLOR "orange"
  :LINE-COLOR "black"
  :STROKEWIDTH 1})


^{:nextjournal.clerk/viewer :vega-lite}
(ml-viz/residual-plot
 (tc/dataset
  {:x (.fittedValues model)

   :y (:disease-progression diabetes)})

 {:XTITLE "Predicted value for disease progression"
  :YTITLE "Actual value for disease progression"
  :LINE (hc/xform  ht/line-chart :X :x :Y :x :MCOLOR "red")
  :RESIDUAL ht/RMV
  :STROKEWIDTH 1})


(defn residuals-plot [pipe-fn dataset]
  (let [inference-colum (first (ds-mod/inference-target-column-names dataset))

        fitted-context
        (mm/fit-pipe
         dataset
         pipe-fn)

        transformed-ctx
        (mm/transform-pipe dataset pipe-fn fitted-context)

        prediction (-> transformed-ctx :metamorph/data (get  inference-colum))
        observed (get dataset inference-colum)]

    (ml-viz/residual-plot
     (tc/dataset
      {:x prediction
       :y (map - observed prediction)
       :zero (repeat (tc/row-count dataset) 0)})
     {:XTITLE "Predicted value"
      :YTITLE "Actual value"
      :LINE (hc/xform  ht/line-chart :X :x :Y :zero :MCOLOR "red")
      :RESIDUAL ht/RMV})))
     



(defn residuals-plot-2 [pipe-fn fitted-ctx dataset]
  (let [inference-colum (first (ds-mod/inference-target-column-names dataset))


        transformed-ctx
        (mm/transform-pipe dataset pipe-fn fitted-ctx)

        prediction (-> transformed-ctx :metamorph/data (get  inference-colum))
        observed (get dataset inference-colum)]

    (ml-viz/residual-plot
     (tc/dataset
      {:x prediction
       :y (map - observed prediction)
       :zero (repeat (tc/row-count dataset) 0)})
     {:XTITLE "Predicted value"
      :YTITLE "Actual value"
      :LINE (hc/xform  ht/line-chart :X :x :Y :zero :MCOLOR "red")
      :RESIDUAL ht/RMV})))




^{:nextjournal.clerk/viewer :vega-lite}
(residuals-plot
 (mm/pipeline (ml/model {:model-type  :smile.regression/ordinary-least-square}))
 diabetes)



^{:nextjournal.clerk/viewer :vega-lite}
(let [pipe-fn (mm/pipeline (ml/model {:model-type  :smile.regression/ordinary-least-square}))
      fitted-context (mm/fit-pipe diabetes pipe-fn)]
  (residuals-plot-2 pipe-fn fitted-context diabetes))


(defn residuals-plot-transformar [ctx]
  ctx)

(->
 (mm/fit-pipe
  diabetes
  (mm/pipeline
   (ml/model {:model-type  :smile.regression/ordinary-least-square})
   residuals-plot-transformar))
 :metamorph/plot-data)

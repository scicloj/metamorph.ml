(ns scicloj.metamorph.ml.viz
  (:require [tablecloth.api :as tc]))

(defn learning-curve-viz [learning-curve-data]

  (let [plot-data
        (tc/pivot->longer learning-curve-data [:metric-test :metric-train]
                        {:value-column-name :metric
                         :target-columns :train-test-metric})]
        

    {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
     :width 500
     :height 500
     :config {
              "mark" {"tooltip" nil}}
     :data {:values (tc/rows plot-data :as-maps)}
     :Title "Learning Curve"
     :layer [{
              :mark :errorband
              :encoding {
                         :color {:value "orange"}
                         :y {
                             :field "metric-train-min"
                             :type "quantitative"
                             :title "METRIC"
                             :scale {"zero" true}}


                         :y2 {:field "metric-train-max"
                              :legend nil}
                         :x {:field "train-ds-size"
                             :title "Traininig instances"}}}
             {
              :mark :errorband
              :encoding {
                         :color {:value "blue"}
                         :y {
                             :field "metric-test-min"
                             :type "quantitative"
                             :title "METRIC"
                             :legend nil
                             :scale {:zero true}}


                         :y2 {:field "metric-test-max"
                              :legend nil},
                         :x {:field "train-ds-size"}}},
             {
              :encoding
              {:color {:field "train-test-metric" :type "nominal"

                       :legend {"labelExpr" "datum.label == 'metric-test' ? 'Cross validation metric' : datum.label == 'metric-train' ? 'Training score' : ''  "}}

               :x {:field "train-ds-size"}
               :y {:field "metric"
                   :type "quantitative"}}

              :mark {:point true :type "line"}}]}))

(ns heatmaps
  (:require [scicloj.metamorph.ml.classification :as cl]
            [tech.v3.dataset :as ds]
            [scicloj.metamorph.ml.viz :as viz]
            [aerial.hanami.templates :as ht]
            [nextjournal.clerk :as clerk]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.learning-curve :as lc]
            [scicloj.metamorph.ml.viz :as ml-viz]
            [scicloj.metamorph.ml.viz.learning-curve :as mlviz.lc]

            [nextjournal.clerk.viewer]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml.loss]
            [scicloj.ml.smile.classification]
            [tablecloth.pipeline :as tc-mm]
            [tech.v3.dataset]
            [tech.v3.dataset.metamorph :as mds]))

(comment
  (clerk/clear-cache!)
  (nextjournal.clerk/show! "notebooks/heatmaps.clj")

  (nextjournal.clerk/serve! {:browse true}))


;; (def predicted-labels [:a :b :c :a :b :c :a :a :a])
;; (def labels [:b :b :a :a :c :c :c :c :b])


(def labels           ["cat", "ant", "cat", "cat", "ant", "bird" "bird" "cat" "ant" "cat" "ant" "ant"])
(def predicted-labels ["ant", "ant", "cat", "cat", "ant", "cat"  "cat"  "ant" "ant" "ant" "cat" "bird"])


(def cm (cl/confusion-map
         predicted-labels
         labels
         :none))

(def distinct-labels
  (distinct
   (concat predicted-labels labels)))



;; (def values-old
;;   (->>
;;    (map
;;     (fn [[label prediction-counts]]
;;       (def prediction-counts prediction-counts)
;;       (map (fn [prediction-count]
;;              (def prediction-count prediction-count)
;;              (hash-map :actual label
;;                        :predicted (first prediction-count)
;;                        :count (second prediction-count)))

;;            (seq prediction-counts)))
;;     (seq))

;;    flatten))

(def values
  (for [actual distinct-labels
        prediction distinct-labels]
    (hash-map :actual actual
              :predicted prediction
              :count (-> cm (get actual 0) (get prediction 0)))))

(clerk/vl

 {:$schema "https://vega.github.io/schema/vega-lite/v5.json"
  :config {:axis {:domain false}
           ;; :range {:ramp {:scheme "yellowgreenblue"}}
           ;; :scale {:bandPaddingInner 0 :bandPaddingOuter 0}
           :view {:step 40}}
  :encoding {
             :x {:field "predicted" :type "nominal"}
             :y {:field "actual" :type "nominal"}}
  :data {:values values}
  :layer [{:encoding {:color {:field "count"
                              :type "quantitative"
                              :scale {:scheme "yelloworangered"}}}
           :mark {:strokeWidth 2 :type "rect"}}
          {:encoding {:text {:field "count" :type "quantitative"}}
           :mark "text"}]})


(def layer
  [(assoc ht/rect-layer :encoding ht/xy-encoding)
   (assoc ht/text-layer
          :encoding
          (assoc ht/xy-encoding
                 :color nil
                 :text {:field "count" :type "quantitative"}))])



     

(def chart
  (assoc
   ht/layer-chart
   :aerial.hanami.templates/defaults
   {:LAYER layer
    :VALDATA values
    ;; :CFGAXIS {:grid true}
    :X "predicted"
    :XTYPE "nominal"
    :Y "actual"
    :YTYPE "nominal"
    :COLOR ht/default-color
    :CFIELD "count"
    :CTYPE "quantitative"
    :CSCALE {:scheme "yelloworangered"}
    :TXT "count"}))

(clerk/vl
 (viz/apply-xform-kvs chart {}))

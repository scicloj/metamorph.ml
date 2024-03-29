(ns scicloj.metamorph.ml.malli
  (:require
   [malli.core :as m]
   [malli.dev.pretty :as pretty]
   [malli.instrument :as mi]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]))

(defn instrument-mm [fn]
  (m/-instrument
     {:report (pretty/thrower) :scope #{:input}
      :schema [:=> [:cat [:map
                          [:metamorph/id any?]
                          [:metamorph/data [:fn dataset?]]
                          [:metamorph/mode [:enum :fit :transform]]]]

               map?]}
     fn))

(defn instrument-ns [ns]
  (mi/collect! {:ns ns})
  (mi/instrument! {:report (pretty/thrower) :scope #{:input}}))

(ns scicloj.metamorph.ml.malli
  (:require
   [malli.util :as mu]
   [malli.core :as m]
   [malli.instrument :as mi]
   [malli.dev.pretty :as pretty]
   [malli.error :as me]
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

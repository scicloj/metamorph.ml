(ns scicloj.metamorph.ml.malli
  (:require
   [malli.core :as m]
   [malli.dev.pretty :as pretty]
   [malli.instrument :as mi]
   [malli.util :as mu]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]))

(defn instrument-mm
  "Instruments a metamorph function with input validation via Malli schema.

  `fn` - Function to instrument

  Returns an instrumented version of `fn` that validates its input is a metamorph
  context map containing:

  * `:metamorph/id` - Any value (typically a keyword identifier)
  * `:metamorph/data` - A tech.ml.dataset
  * `:metamorph/mode` - Either `:fit` or `:transform`

  Throws detailed validation errors on schema violations. Use this to enforce
  metamorph contract compliance at runtime.

  See also: `scicloj.metamorph.ml.malli/instrument-ns`"
  [fn]
  (m/-instrument
     {:report (pretty/thrower) :scope #{:input}
      :schema [:=> [:cat [:map
                          [:metamorph/id any?]
                          [:metamorph/data [:fn dataset?]]
                          [:metamorph/mode [:enum :fit :transform]]]]

               map?]}
     fn))

(defn instrument-ns
  "Instruments all Malli-schema functions in a namespace for runtime validation.

  `ns` - Namespace symbol (e.g., `'my.namespace`)

  Collects all functions with `:malli/schema` metadata in the namespace and
  instruments them to validate inputs at runtime. Throws detailed errors on
  schema violations.

  Use this to enable validation for an entire namespace during development or
  testing.

  See also: `scicloj.metamorph.ml.malli/instrument-mm`"
  [ns]
  (mi/collect! {:ns ns})
  (mi/instrument! {:report (pretty/thrower) :scope #{:input} :filters [(mi/-filter-ns ns)]}))

(defn model-options->full-schema
  "Converts model options map to a full Malli schema with `:model-type` field.

  `model-options` - Map containing `:options` key with a Malli schema

  Returns a Malli schema that includes the `:options` schema plus a required
  `:model-type` keyword field. Defaults to empty `:map` schema if no `:options`
  key is present.

  Used internally to validate model option maps during model registration."
  [model-options]
  (->
   (get model-options :options [:map ])
   m/schema
   (mu/assoc :model-type keyword?)))
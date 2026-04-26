(ns scicloj.metamorph.ml.malli
  {:no-doc true}
  (:require
   [malli.core :as m]
   [malli.dev.pretty :as pretty]
   [malli.instrument :as mi]
   [malli.util :as mu]
   [tech.v3.dataset.impl.dataset :refer [dataset?]]
   [malli.registry :as mr]))

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


(def custom-schemas
   
   
    
    {
     :scicloj.metamorph.ml/optimize-hyperparams--metric-fn
     fn?

     :scicloj.metamorph.ml/optimize-hyperparams--loss-or-accuracy
     [:enum :accuracy :loss]

     :scicloj.metamorph.ml/optimize-hyperparams--pipeline-fn-or-decl-seq
     [:sequential [:or vector? fn?]]
     
     :scicloj.metamorph.ml/optimize-hyperparams--train-test-split-seq
     [:sequential [:map {:closed true}
                   [:split-uid {:optional true} string?]
                   [:train [:fn dataset?]]
                   [:test {:optional true} [:fn dataset?]]]]
     
     :scicloj.metamorph.ml/optimize-hyperparams--options
     [:or empty? [:map
                  [:return-best-pipeline-only {:optional true} boolean?]
                  [:return-best-crossvalidation-only {:optional true} boolean?]
                  [:map-fn {:optional true} [:enum :map :pmap :mapv :ppmap]]
                  [:ppmap-grain-size {:optional true} int?]
                  [:evaluation-handler-fn {:optional true} fn?]
                  [:other-metrics {:optional true} [:sequential [:map
                                                                 [:name keyword?]
                                                                 [:metric-fn fn?]]]]
                  [:attach-fn-sources {:optional true} [:map [:ns any?]
                                                        [:pipe-fns-clj-file string?]]]]]
     :scicloj.metamorph.ml/optimize-hyperparams--evaluation-result
     [:sequential
      [:sequential
       [:map {:closed true}
        [:split-uid [:maybe string?]]
        [:fit-ctx [:map [:metamorph/mode [:enum :fit :transform]]]]
        [:timing-fit int?]

        [:train-transform [:map {:closed true}
                           [:other-metrics [:sequential [:map {:closed true}
                                                         [:name keyword?]
                                                         [:metric-fn fn?]
                                                         [:metric float?]]]]
                           [:timing int?]
                           [:metric float?]
                           [:probability-distribution  [:maybe [:fn dataset?]]]
                           [:min float?]
                           [:mean float?]
                           [:max float?]
                           [:ctx map?]]]
        [:test-transform [:map {:closed true}
                          [:other-metrics [:sequential [:map {:closed true}
                                                        [:name keyword?]
                                                        [:metric-fn fn?]
                                                        [:metric float?]]]]
                          [:timing int?]
                          [:metric float?]
                          [:probability-distribution  [:maybe [:fn dataset?]]]
                          [:min float?]
                          [:mean float?]
                          [:max float?]
                          [:ctx map?]]]
        [:loss-or-accuracy [:enum :accuracy :loss]]
        [:metric-fn fn?]
        [:pipe-decl [:maybe sequential?]]
        [:pipe-fn fn?]
        [:source-information [:maybe [:map [:classpath [:sequential string?]]
                                      [:fn-sources [:map-of :qualified-symbol [:map [:source-form any?]
                                                                               [:source-str string?]]]]]]]]]]})
   
(mr/set-default-registry!
 (-> 
  (merge 
   (m/default-schemas)
   custom-schemas
   )
  (mr/simple-registry)
  ))

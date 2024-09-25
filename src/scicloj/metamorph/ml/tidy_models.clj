(ns scicloj.metamorph.ml.tidy-models
  (:require
   [clojure.edn :as edn]
   [clojure.set :as set]
   [tech.v3.dataset :as ds]))

(def ^:dynamic
 ^{:doc "Controls if the result columns of the tidy fns of a model
(glance-fn, tidy-fn, augment-fn is validated against these base
https://github.com/scicloj/metamorph.ml/tree/main/resources/*.edn
 and if on violation they fail."
   :added "1.0"}
 *validate-tidy-fns* true)

(defn allowed-glance-columns []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-glance.edn"))))

(defn allowed-tidy-columns []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-tidy.edn"))))

(defn allowed-augment-columns []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-augment.edn"))))

(defn _get-allowed-keys []
  {:glance (allowed-glance-columns)
   :tidy (allowed-tidy-columns)
   :augment (allowed-augment-columns)})

(def get-allowed-keys (memoize _get-allowed-keys))



(defn- validate-ds [ds allowed-columns fn-name]
  (if (true? *validate-tidy-fns*)
    (let [
          invalid-keys
          (set/difference
           (into #{} (ds/column-names ds))
           (into #{} allowed-columns))]
      (if (empty? invalid-keys)
        ds
        (throw (Exception. (format "invalid keys from %s: %s" fn-name invalid-keys)))))
    ds))

(defn validate-tidy-ds [ds]
  (validate-ds ds (:tidy (get-allowed-keys))  "tidy-fn"))

(defn validate-glance-ds [ds]
  (validate-ds ds (:glance (get-allowed-keys)) "glance-fn"))

(defn validate-augment-ds [ds data]
  (validate-ds
   ds
   (concat (:augment (get-allowed-keys)) (ds/column-names data))
   "augment-fn"))

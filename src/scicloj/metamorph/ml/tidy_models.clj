(ns scicloj.metamorph.ml.tidy-models
  (:require
   [clojure.edn :as edn]
   [clojure.set :as set]
   [tech.v3.dataset :as ds]))


(def allowed-glance-columns
  (keys
   (edn/read-string (slurp "resources/columms-glance.edn"))))

(def allowed-tidy-columns
  (keys
   (edn/read-string (slurp "resources/columms-tidy.edn"))))

(def allowed-augment-columns
  (keys
   (edn/read-string (slurp "resources/columms-augment.edn"))))



(defn- validate-ds [ds allowed-columns fn-name]
  (let [
        invalid-keys
        (set/difference
         (into #{} (ds/column-names ds))
         (into #{} allowed-columns))]
    (if (empty? invalid-keys)
      ds
      (throw (Exception. (format "invalid keys from %s: %s" fn-name invalid-keys))))))

(defn validate-tidy-ds [ds]
  (validate-ds ds allowed-tidy-columns "tidy-fn"))

(defn validate-glance-ds [ds]
  (validate-ds ds allowed-glance-columns "glance-fn"))



(defn validate-augment-ds [ds data]
  (validate-ds
   ds
   (concat allowed-augment-columns (ds/column-names data))
   "augment-fn"))

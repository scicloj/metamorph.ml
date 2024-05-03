(ns scicloj.metamorph.ml.tidy-models
  (:require
   [clojure.edn :as edn]))


(def allowed-glance-columns
  (keys
   (edn/read-string (slurp "resources/columms-glance.edn"))))

(def allowed-tidy-columns
  (keys
   (edn/read-string (slurp "resources/columms-tidy.edn"))))

(def allowed-augment-columns
  (keys
   (edn/read-string (slurp "resources/columms-augment.edn"))))




(defn validate-tidy-ds [ds]

  (let [

        invalid-keys
        (clojure.set/difference
         (into #{} (keys ds))
         (into #{} allowed-tidy-columns))]

    (if (empty? invalid-keys)
      ds
      (throw (Exception. "" (format "invalid keys from tidy-fn: %s" invalid-keys))))))

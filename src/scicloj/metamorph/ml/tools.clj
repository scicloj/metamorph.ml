(ns scicloj.metamorph.ml.tools
  (:require
   [clojure.pprint :as pprint]))

(defn dissoc-in
  "Dissociate a value in a nested assocative structure, identified by a sequence
  of keys. Any collections left empty by the operation will be dissociated from
  their containing structures."
  [m ks]
  (if-let [[k & ks] (seq ks)]
    (if (seq ks)
      (let [v (dissoc-in (get m k) ks)]
        (if (empty? v)
          (dissoc m k)
          (assoc m k v)))
      (dissoc m k))
    m))


(defn multi-dissoc-in [m kss]
  (reduce (fn [x y]
            (dissoc-in x y))
          m
          kss))


(defn pp-str [x]
  (with-out-str (pprint/pprint x)))

(ns scicloj.metamorph.persistence-tools
  (:require
   [clojure.java.classpath]
   [clojure.repl]))
   

(defn keys-in
    "Returns a sequence of all key paths in a given map using DFS walk."
    [m]
    (letfn [(children [node]
              (let [v (get-in m node)]
                (if (map? v)
                  (map (fn [x] (conj node x)) (keys v))
                  [])))
            (branch? [node] (-> (children node) seq boolean))]
      (->> (keys m)
           (map vector)
           (mapcat #(tree-seq branch? children %)))))


(defn find-model-data [m]
  (->>
   (keys-in m)
   (filter #(= :model-data (last %)))))

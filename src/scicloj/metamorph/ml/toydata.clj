(ns scicloj.metamorph.ml.toydata
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            )

  )

(defn sonar-ds []
  (->  (ds/->dataset
        (clojure.java.io/input-stream (clojure.java.io/resource "data/sonar.csv"))
        ;; "data/sonar.csv"
        {:header-row? false :file-type :csv})
       (tc/rename-columns
        (zipmap
         (map #(str "column-" %) (range 61))
         (concat
          (map #(keyword (str "x" %)) (range 60))
          [:material]
          )

         ))))

;; (sonar-ds)

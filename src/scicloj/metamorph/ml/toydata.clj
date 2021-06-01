(ns scicloj.metamorph.ml.toydata
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.functional :as dtfn]
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
;;

(defn diabetes-ds []
  (let [data
        (-> (clojure.java.io/resource "data/diabetes_data.csv")
            (clojure.java.io/input-stream )
            (ds/->dataset
             {:file-type :csv :gzipped? true :separator " " :header-row? false} )
            ;; (ds/column-names)
            (ds/rename-columns
             (zipmap
              ( map #(str "column-" %) (range 10))
              [:age :sex :bmi :bp
               :s1 :s2 :s3 :s4 :s5 :s6]

              )
             )
            )



        targets
        (-> (clojure.java.io/resource "data/diabetes_target.csv")
            (clojure.java.io/input-stream )
            (ds/->dataset
             {:file-type :csv :gzipped? true :separator " " :header-row? false} )
            (ds/rename-columns {"column-0" :disease-progression})
            (ds/update-column :disease-progression (fn [col] (map #(Integer/valueOf (Math/round %)) col)))
            )]


    (->
     (ds/concat data targets)
     (ds-mod/set-inference-target :disease-progression))
    ))

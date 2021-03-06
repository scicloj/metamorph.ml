(ns scicloj.metamorph.ml.toydata
  (:require [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.functional :as dtfn]
            [camel-snake-kebab.core :as csk]
            ))

(defn sonar-ds []
  (->  (ds/->dataset
        (clojure.java.io/input-stream (clojure.java.io/resource "data/sonar.csv"))
        {:header-row? false :file-type :csv})
       (tc/rename-columns
        (zipmap
         (map #(str "column-" %) (range 61))
         (concat
          (map #(keyword (str "x" %)) (range 60))
          [:material]
          )))))


(defn diabetes-ds []
  (let [data
        (-> (clojure.java.io/resource "data/diabetes_data.csv")
            (clojure.java.io/input-stream )
            (ds/->dataset
             {:file-type :csv :gzipped? true :separator " " :header-row? false} )
            (ds/rename-columns
             (zipmap
              ( map #(str "column-" %) (range 10))
              [:age :sex :bmi :bp
               :s1 :s2 :s3 :s4 :s5 :s6])))

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
     (ds-mod/set-inference-target :disease-progression))))

(defn iris-ds []
  (-> (clojure.java.io/resource "data/iris.csv")
      (clojure.java.io/input-stream )
      (ds/->dataset
       {:file-type :csv :gzipped? true :header-row? false  :n-initial-skip-rows 1} )
      (ds/rename-columns
       (zipmap
        ( map #(str "column-" %) (range 5))
        [:sepal_length :sepal_width
         :petal_length :petal_width
         :species
         ]))
      (ds-mod/set-inference-target :species)
      (ds/categorical->number [:species] {} :int16)))


(defn breast-cancer-ds []
  (-> (clojure.java.io/resource "data/breast_cancer.csv")
      (clojure.java.io/input-stream )
      (ds/->dataset
       {:file-type :csv :gzipped? true :header-row? false  :n-initial-skip-rows 1} )
      (ds/rename-columns
       (zipmap
        ( map #(str "column-" %) (range 31))
        (conj
         (mapv csk/->kebab-case-keyword
               ["mean radius" "mean texture"
                "mean perimeter" "mean area"
                "mean smoothness" "mean compactness"
                "mean concavity" "mean concave points"
                "mean symmetry" "mean fractal dimension"
                "radius error" "texture error"
                "perimeter error" "area error"
                "smoothness error" "compactness error"
                "concavity error" "concave points error"
                "symmetry error" "fractal dimension error"
                "worst radius" "worst texture"
                "worst perimeter" "worst area"
                "worst smoothness" "worst compactness"
                "worst concavity" "worst concave points"
                "worst symmetry" "worst fractal dimension"])
         :class)))

      (ds/update-column :class (fn [col] (map
                                         #(case % 0 :malignant 1 :benign )
                                         col
                                         ))
                        )
      (ds/categorical->number [:class] {} :int16)
      (ds-mod/set-inference-target :class)))

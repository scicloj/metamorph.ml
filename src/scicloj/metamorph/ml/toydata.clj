(ns scicloj.metamorph.ml.toydata
  "Deprecated ns. Use scicloj.metamorph.ml.rdatasets instead"
  {:deprecated "1.1"}
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.java.io :as io]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))


(defn sonar-ds []
  (->  (ds/->dataset
        (io/input-stream (io/resource "data/sonar.csv"))
        {:header-row? false :file-type :csv})
       (tc/rename-columns
        (zipmap
         (map #(str "column-" %) (range 61))
         (concat
          (map #(keyword (str "x" %)) (range 60))
          [:material])))
       (ds-mod/set-inference-target :material)))
          


(defn diabetes-ds []
  (let [data
        (-> (io/resource "data/diabetes_data.csv")
            (io/input-stream)
            (ds/->dataset
             {:file-type :csv :separator " " :header-row? false})
            (ds/rename-columns
             (zipmap
              ( map #(str "column-" %) (range 10))
              [:age :sex :bmi :bp
               :s1 :s2 :s3 :s4 :s5 :s6])))

        targets
        (-> (io/resource "data/diabetes_target.csv")
            (io/input-stream)
            (ds/->dataset
             {:file-type :csv :separator " " :header-row? false})
            (ds/rename-columns {"column-0" :disease-progression})
            (ds/update-column :disease-progression (fn [col] (map #(Integer/valueOf (Math/round %)) col))))]
    (->
     (ds/add-column data (:disease-progression targets))
     (ds-mod/set-inference-target :disease-progression))))

(defn iris-ds []
  (-> 
   (rdatasets/datasets-iris)
   (tc/drop-columns :rownames)
   (ds-mod/set-inference-target :species)
   (ds/categorical->number [:species] {} :int16)))


(defn breast-cancer-ds []
  (let [dslabs-brca
        (->
         (rdatasets/dslabs-brca)
         (tc/drop-columns :rownames))
        col-names (mapv csk/->kebab-case-keyword
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
        old-col-names
        (tc/column-names dslabs-brca)]
       (-> 
        dslabs-brca
        (ds/rename-columns
         (zipmap
          old-col-names
          (conj col-names :class)))

        (ds/update-column :class (fn [col] (map #(case %
                                                   "M"  :malignant
                                                   "B"  :benign)
                                                col)))
        
        
        (ds/categorical->number [:class] {} :int16)
        (ds-mod/set-inference-target :class))))


(defn titanic-ds-split []
  {:train
   (->
    (io/resource "data/titanic-train.nippy")
    (io/input-stream)
    (ds/->dataset {:file-type :nippy}))
   :test
   (->
    (io/resource "data/titanic-test.nippy")
    (io/input-stream)
    (ds/->dataset {:file-type :nippy}))})

(defn mtcars-ds []
  (->
   (rdatasets/datasets-mtcars)
   (tc/drop-columns [:rownames])))



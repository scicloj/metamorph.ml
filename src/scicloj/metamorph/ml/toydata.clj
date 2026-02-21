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


(defn sonar-ds
  "Loads the Sonar dataset with 60 features for binary classification.

  Returns a dataset for detecting material type (metal vs. rock) from sonar
  signals. Contains 60 numeric features (:x0 through :x59) representing sonar
  frequency returns, with `:material` as the target variable.

  Inference target is set to `:material`."
  []
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
          


(defn diabetes-ds
  "Loads the Diabetes dataset with 10 features for regression.

  Returns a dataset for predicting disease progression from baseline measurements.
  Features include `:age`, `:sex`, `:bmi` (body mass index), `:bp` (blood pressure),
  and `:s1` through `:s6` (six blood serum measurements).

  Target variable is `:disease-progression` (integer). Inference target is set
  to `:disease-progression`."
  []
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

(defn iris-ds
  "Loads the classic Iris dataset with 4 features for multi-class classification.

  Returns the famous Fisher's Iris dataset containing measurements of 150 iris
  flowers from three species. Features are sepal/petal dimensions, target is
  `:species` (setosa, versicolor, or virginica).

  Species values are converted to numeric codes. Inference target is set to `:species`."
  []
  (->
   (rdatasets/datasets-iris)
   (tc/drop-columns :rownames)
   (ds-mod/set-inference-target :species)
   (ds/categorical->number [:species] {} :int16)))


(defn breast-cancer-ds
  "Loads the Breast Cancer Wisconsin (Diagnostic) dataset with 30 features for binary classification.

  Returns a dataset for diagnosing breast cancer from digitized images of cell
  nuclei. Contains 30 features describing mean, standard error, and worst values
  of cell characteristics (radius, texture, perimeter, area, smoothness,
  compactness, concavity, concave points, symmetry, fractal dimension).

  Target variable is `:class` (:malignant or :benign, converted to numeric).
  Inference target is set to `:class`."
  []
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


(defn titanic-ds-split
  "Loads the Titanic dataset pre-split into training and test sets.

  Returns a map with `:train` and `:test` keys, each containing a dataset for
  predicting passenger survival on the Titanic. Datasets are loaded from Nippy
  format (fast binary serialization).

  Use this for evaluating models with a pre-defined train/test split."
  []
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

(defn mtcars-ds
  "Loads the Motor Trend Car Road Tests dataset with 11 features.

  Returns the classic mtcars dataset from the 1974 Motor Trend magazine,
  containing specifications and performance metrics for 32 automobiles.
  Features include mpg, cylinders, displacement, horsepower, weight, etc.

  Commonly used for regression and clustering examples."
  []
  (->
   (rdatasets/datasets-mtcars)
   (tc/drop-columns [:rownames])))



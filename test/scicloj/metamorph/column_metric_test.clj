(ns scicloj.metamorph.column-metric-test
  (:require
   [clojure.test :refer [deftest is]]
   [fastmath.core :as m]
   [fastmath.vector :as v]
   [libpython-clj2.python :as py]
   [scicloj.metamorph.ml.column-metric :as col-metric]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column :as col]
   [fastmath.stats :as stats]))

(py/initialize!)

(deftest accuracy-score-valid
  (is (= 0.3333333333333333
         (col-metric/classification-metric-fastmath
          (ds/new-dataset [(col/new-column :my-target [1 3 2] {:inference-target? true})])
          (ds/new-dataset [(col/new-column :pred [1 2 3] {:column-type :prediction})])
          :classification
          :micro))))



(deftest f1-invalid
  (is (thrown? AssertionError
               (col-metric/classification-metric--tribuo
                (ds/new-dataset [(col/new-column :my-target-1 [:a :c :b] {:inference-target? true})
                                 (col/new-column :my-target-2 [:a :c :b] {:inference-target? true})])
                (ds/new-dataset [(col/new-column :pred [:a :b :c] {:column-type :prediction})])
                :f1
                :macro)))


  (is (thrown? AssertionError
               (col-metric/classification-metric--tribuo
                (ds/new-dataset [(col/new-column :my-target [0, 1, 2, 0, 1, 2] {:inference-target? true})])
                (ds/new-dataset [(col/new-column :pred [0.0, 2.0, 1.0, 0.0, 0.0, 1.0] {:column-type :prediction})])
                :f1 :macro)))
  (is (thrown? AssertionError
               (col-metric/classification-metric--tribuo
                (ds/new-dataset [(col/new-column :my-target [0, 1, 2, 0, 1, 2] {})])
                (ds/new-dataset [(col/new-column :pred [0, 2, 1, 0, 0, 1] {:column-type :prediction})])
                :f1 :macro)))
  (is (thrown? AssertionError
               (col-metric/classification-metric--tribuo [0, 1, 2, 0, 1, 2] [0, 2, 1, 0, 0, 1]
                                                 :f1 :macro))))


(deftest classification-metrix-fastmath
  (let [y-true (ds/new-dataset [(col/new-column :my-target [0, 1, 2, 0, 1, 2] {:inference-target? true})])
        y-pred (ds/new-dataset [(col/new-column :pred [0, 2, 1, 0, 0, 1] {:column-type :prediction})])]

    (is (= 0.2666666666666667 (col-metric/classification-metric-fastmath y-true y-pred :f1-score :macro)))
    (is (= 0.3333333333333333 (col-metric/classification-metric-fastmath y-true y-pred :f1-score :micro)))


    (is (= 0.3333333333333333 (col-metric/classification-metric-fastmath y-true y-pred :accuracy :micro)))
    (is (= 0.5555555555555556 (col-metric/classification-metric-fastmath y-true y-pred :accuracy :macro)))


    (is (=
         [1.3333333333333333
          1.3333333333333333
          0.6666666666666667
          2.6666666666666665
          0.2666666666666667
          0.2222222222222222
          0.33333333333333337]

         [(col-metric/classification-metric-fastmath y-true y-pred :fn :macro)
          (col-metric/classification-metric-fastmath y-true y-pred :fp :macro)
          (col-metric/classification-metric-fastmath y-true y-pred :tp :macro)
          (col-metric/classification-metric-fastmath y-true y-pred :tn :macro)
          (col-metric/classification-metric-fastmath y-true y-pred :f-beta :macro {:beta 1.0})
          (col-metric/classification-metric-fastmath y-true y-pred :precision :macro)
          (col-metric/classification-metric-fastmath y-true y-pred :recall :macro)]))))




(deftest classification-metrix-tribuo-valid
  (let [y-true (ds/new-dataset [(col/new-column :my-target [0, 1, 2, 0, 1, 2] {:inference-target? true})])
        y-pred (ds/new-dataset [(col/new-column :pred [0, 2, 1, 0, 0, 1] {:column-type :prediction})])]
    (is (= 0.26666666666666666 (col-metric/classification-metric--tribuo y-true y-pred :f1 :macro)))
    (is (= 0.3333333333333333 (col-metric/classification-metric--tribuo y-true y-pred :f1 :micro)))


    (is (= 0.3333333333333333 (col-metric/classification-metric--tribuo y-true y-pred :accuracy :micro)))
    (is (= 0.3333333333333333 (col-metric/classification-metric--tribuo y-true y-pred :accuracy :macro)))

    (is (=
         [1.3333333333333333
          1.3333333333333333
          0.6666666666666666
          2.6666666666666665
          0.26666666666666666
          0.2222222222222222
          0.3333333333333333]

         [(col-metric/classification-metric--tribuo y-true y-pred :fn :macro)
          (col-metric/classification-metric--tribuo y-true y-pred :fp :macro)
          (col-metric/classification-metric--tribuo y-true y-pred :tp :macro)
          (col-metric/classification-metric--tribuo y-true y-pred :tn :macro)
          (col-metric/classification-metric--tribuo y-true y-pred :fscore :macro {:beta 1.0})
          (col-metric/classification-metric--tribuo y-true y-pred :precision :macro)
          (col-metric/classification-metric--tribuo y-true y-pred :recall :macro)]))))



(deftest roc-auc-score

  (let [r
        (py/run-simple-string
         "
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
target_names = iris.target_names
X, y = iris.data, iris.target
y = iris.target_names[y]

random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
n_classes = len(np.unique(y))
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)
(
    X_train,
    X_test,
    y_train,
    y_test,
) = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)                      

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

from sklearn.metrics import roc_auc_score

roc_auc_ovr_none = roc_auc_score(
    y_test,
    y_score,
    multi_class='ovr',
    average=None,
    labels=['setosa','versicolor','virginica']
    
)

roc_auc_ovr_macro = roc_auc_score(
    y_test,
    y_score,
    multi_class='ovr',
    average='macro',
    labels=['setosa','versicolor','virginica']
    
)

                       ")

        y-true (ds/new-dataset [(ds/new-column :species (-> r :globals (get "y_test") py/->jvm) {:inference-target? true})])
        y-score (->
                 (tc/dataset (-> r :globals (get "y_score") py/->jvm))

                 (tc/rename-columns {0 "setosa"
                                     1 "versicolor"
                                     2 "virginica"})
                 (ds/assoc-metadata ["setosa" "versicolor" "virginica"] :column-type :probability-distribution))]

    (is (v/delta-eq
         (-> r :globals (get "roc_auc_ovr_none") py/->jvm)
         (col-metric/roc_auc-score y-true y-score :ovr nil)))


    (is (m/delta-eq
         (-> r :globals (get "roc_auc_ovr_macro") py/->jvm)
         (col-metric/roc_auc-score y-true y-score :ovr :macro)))))


(deftest regression-metric
  (is (= 0.375
         (col-metric/regression-metric--fastmath
          (ds/new-dataset [(col/new-column :my-target-1 [3 -0.5 2 7] {:inference-target? true})])
          (ds/new-dataset [(col/new-column :pred [2.5 0.0 2 8] {:column-type :prediction})])
          :mse)))
  (is (= 0.6123724356957945
         (col-metric/regression-metric--fastmath
          (ds/new-dataset [(col/new-column :my-target-1 [3 -0.5 2 7] {:inference-target? true})])
          (ds/new-dataset [(col/new-column :pred [2.5 0.0 2 8] {:column-type :prediction})])
          :rmse))))

(deftest regression-metric-invalid
  (is (thrown? AssertionError
               (col-metric/regression-metric--fastmath
                (ds/new-dataset [(col/new-column :my-target-1 [1 2 3] {:inference-target? true})])
                (ds/new-dataset [(col/new-column :pred [1 2 3] {:column-type :prediction})])
                :rme)))


  (is (thrown? AssertionError
               (col-metric/regression-metric--fastmath
                (ds/new-dataset [(col/new-column :my-target-1 [1.0] {:inference-target? true})])
                (ds/new-dataset [(col/new-column :pred [3.0] {:column-type :prediction})])
                :non-existing-fm-fn))))


(deftest classification--cat-maps
  (is (= 1.0
         (col-metric/classification-metric-fastmath
          (ds/new-dataset [(col/new-column :x [1 2 3 4] {:inference-target? true})])
          (ds/new-dataset [(col/new-column :y [1 2 3 4] {:column-type :prediction})])
          :accuracy
          :macro)))

  (is (= 1.0
         (col-metric/classification-metric-fastmath
          (ds/new-dataset [(col/new-column :x [3 2 1 0] ; :b :a :c :d
                                           {:inference-target? true
                                            :categorical-map {:lookup-table {:d 0, :c 1, :a 2, :b 3},
                                                              :src-column :x, :result-datatype :int64}
                                            :categorical? true})])
          (ds/new-dataset [(col/new-column :y [0 2 1 3]
                                           {:column-type :prediction
                                            :categorical-map {:lookup-table {:d 3, :c 1, :a 2, :b 0},
                                                              :src-column :y, :result-datatype :int64}
                                            :categorical? true})])
          :accuracy
          :macro)))
  (is (= 1.0
         (col-metric/classification-metric-fastmath
          (ds/new-dataset [(col/new-column :x [3 2 1 0] ; :b :a :c :d
                                           {:inference-target? true
                                            :categorical-map {:lookup-table {:d 0 :c 1 :a 2 :b 3}
                                                              :src-column :x, :result-datatype :int64}
                                            :categorical? true})])
          (ds/new-dataset [(col/new-column :y [0.0 2.0 1.0 3.0] ; :b :a :c :d
                                           {:column-type :prediction
                                            :categorical-map {:lookup-table {:d 3 :c 1 :a 2 :b 0}
                                                              :src-column :y :result-datatype :float64}
                                            :categorical? true})])
          :accuracy
          :macro))))






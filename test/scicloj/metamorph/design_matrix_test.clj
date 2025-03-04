(ns scicloj.metamorph.design-matrix-test
  (:require  [clojure.test :as t :refer [deftest is]]
             [tablecloth.api :as tc]
             [tablecloth.column.api]
             [tech.v3.dataset.tensor :as tensor]
             [scicloj.metamorph.ml.design-matrix :as dm]))



(defn interactions [a b]
  (vector (tablecloth.column.api/* a b)
          (tablecloth.column.api/+ a b)))

(def ds
  (tc/dataset

   {"a" [7 3 2 3 4 5 6 7 8]              ;will be removed from dm
    :b [1 2 3 2 3 4 3 2 1]               ;stays as is
    'c [3 1 2 4 2 1 3 2 4]               ;will be remove from ds
    :x ["a" "b" "c" "d" "e" "f" "g" "h" "i"]
    :y [3 1 2 4 2 1 3 2 4]}))

(deftest design-matrix
  (let [my-dm

        (dm/create-design-matrix
         ds
         [:y]                                  ;becomes "target column"
         [[:sum '(+ "a" :b c)]
          [:b '(clojure.core/identity :b)]                   ; stays as is
          [:a** '(tablecloth.column.api/pow "a" 2)]
          [:a-str '(clojure.core/str "a")]
          [nil '(clojure.string/upper-case (str :x "a"))] ;autogenerated colum name
          [:a+c '(clojure.core/+ "a" c)]
          ['a+c*b '(clojure.core/* :a+c :b)]
          [:interaction '(scicloj.metamorph.design-matrix-test/interactions :b c)] ;will be split, as it returns seq in mapping,
          [nil '(tablecloth.column.api/+ "a" :b c :sum :a+c a+c*b)]])

        t (tensor/dataset->tensor my-dm)]

    (is (=

         [:b
          :y
          :sum
          :a**
          :a-str
          "(clojure.string/upper-case (str :x \"a\"))"
          :a+c
          'a+c*b
          "(tablecloth.column.api/+ \"a\" :b c :sum :a+c a+c*b)"
          :interaction-0
          :interaction-1]
         (tc/column-names my-dm)))

    (is (=
         [[1.000 3.000 11.00 49.00 5.000 1.000 10.00 10.00 42.00 3.000 4.000]
          [2.000 1.000 6.000 9.000 1.000 0.000 4.000 8.000 24.00 2.000 3.000]
          [3.000 2.000 7.000 4.000 0.000 2.000 4.000 12.00 30.00 6.000 5.000]
          [2.000 4.000 9.000 9.000 1.000 3.000 7.000 14.00 39.00 8.000 6.000]
          [3.000 2.000 9.000 16.00 2.000 4.000 6.000 18.00 42.00 6.000 5.000]
          [4.000 1.000 10.00 25.00 3.000 5.000 6.000 24.00 50.00 4.000 5.000]
          [3.000 3.000 12.00 36.00 4.000 6.000 9.000 27.00 60.00 9.000 6.000]
          [2.000 2.000 11.00 49.00 5.000 7.000 9.000 18.00 49.00 4.000 4.000]
          [1.000 4.000 13.00 64.00 6.000 8.000 12.00 12.00 50.00 4.000 5.000]]
         t))))


(deftest all-col-names-varints
  (let [ds
        (tc/dataset
         {:a [0 1 2]
          'a [2 3 4]
          "a" [4 5 6]
          :y [0 0 0]})
        dm
        (dm/create-design-matrix
         ds
         [:y]
         [[:sum '(clojure.core/+ :a a "a")]])]

    (is (= [6 9 12]
           (:sum dm)))))

(deftest dm-mutiple-returns
  (is (=
       [{:y 2, :p-0 2, :p-1 0, :q-a 2, :q-b 0, :r-0 2, :r-1 0}
        {:y 1, :p-0 2, :p-1 1, :q-a 2, :q-b 1, :r-0 2, :r-1 1}
        {:y 0, :p-0 2, :p-1 0, :q-a 2, :q-b 0, :r-0 2, :r-1 0}]

       (->
        (dm/create-design-matrix (tc/dataset {"v" [4 5 6]
                                              :w [:A :B :C]
                                              :x (range 3)
                                              :y (reverse (range 3))})
                                 [:y]
                                 [[:p '((juxt tablecloth.column.api/+ tablecloth.column.api/*) :x :y)]
                                  [:q '{:a (tablecloth.column.api/+ :x :y) :b (tablecloth.column.api/* :x :y)}]
                                  [:r '[(tablecloth.column.api/+ :x :y)
                                        (tcc/* :x :y)]]])
        (tc/rows :as-maps)))))



(deftest test-map-column->columns--keyword
  (is (= [:a-x :a-y]
         (tc/column-names
          (dm/map-column->columns
           (tc/dataset
            {:a [{:x 1 :y 2}
                 {:x 3 :y 4}]}) :a))))


  (is (= ["a-x" "a-y"]
         (tc/column-names
          (dm/map-column->columns
           (tc/dataset
            {"a" [{:x 1 :y 2}
                  {:x 3 :y 4}]}) "a"))))

  (is (= [:a-x :a-y]
         (tc/column-names
          (dm/map-column->columns
           (tc/dataset
            {:a [{:x 1 "y" 2}
                 {:x 3 "y" 4}]}) :a)))))





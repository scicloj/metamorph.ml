(ns scicloj.metamorph.categorical-test
  (:require  [clojure.test :refer [deftest is ]]
             [scicloj.metamorph.ml.categorical :as cat]
             [scicloj.metamorph.core :as mm]
             [tablecloth.api :as tc]))




(deftest independent
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :independent))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z]})
          :metamorph/mode :fit})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx :yy :zz]})
                 :metamorph/mode :transform}))]
    (is (= #{:x :y :z} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:xx :yy :zz} (-> transformed-ctx :metamorph/data :a-xx meta :one-hot-map :one-hot-table keys set)))))


(deftest train-to-test
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :fit))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z :xx]})
          :metamorph/mode :fit})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx]})
                 :metamorph/mode :transform}))]

    (is (= #{:xx :x :y :z} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:xx :x :y :z} (-> transformed-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))))



(deftest global-ds
  (let [pipe-fn
        (mm/pipeline
         (cat/transform-one-hot :a :full {:result-datatype :int}))

        fitted-ctx
        (pipe-fn
         {:metamorph/data (tc/dataset {:a [:x :y :z]})
          :metamorph/mode :fit
          :metamorph.ml/full-ds (tc/dataset {:a [:x :y :z :xx :yy :zz]})})

        transformed-ctx
        (pipe-fn
         (merge fitted-ctx
                {:metamorph/data (tc/dataset {:a [:xx :yy :zz]})
                 :metamorph/mode :transform}))]

    (is (= #{:x :y :z :xx :yy :zz} (-> fitted-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))
    (is (= #{:x :y :z :xx :yy :zz} (-> transformed-ctx :metamorph/data :a-x meta :one-hot-map :one-hot-table keys set)))))




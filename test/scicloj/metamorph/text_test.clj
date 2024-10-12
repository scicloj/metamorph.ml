(ns scicloj.metamorph.text-test
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.text :as text]
            [tablecloth.api :as tc])
  )



(defn- parse-review-line [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     (dec (Integer/parseInt (second splitted)))]))

(deftest ->tidy-text

  (let [text

        (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                              parse-review-line
                              #(str/split % #" ")
                              :max-lines 5
                              :skip-lines 1)
            (tc/drop-missing)
            (tc/drop-rows #(empty? (:word %)))
            (tc/drop-rows #(nil? (:word %)))
            (tc/drop-rows #(= "" (:word %))))
        tf (text/->term-frequency text)
        tf-with-index (text/add-word-idx tf)
        ]
    (is (= 576
           (tc/row-count text)))

    (is (= '(:word :word-index :document :meta)
           (tc/column-names text)))

    (is (= [["Is" 0 0 3] ["it" 1 0 3] ["a" 2 0 3] ["great" 3 0 3] ["product" 4 0 3]]
           (-> text
               (tc/head)
               (tc/rows :maps))))
    (is (= ["Is" "it" "a" "great" "product"] (take 5 (-> tf :word))))
    (is (= {0 68, 1 24, 2 136, 3 135, 4 63}
           (-> tf :document frequencies)))
    (is (= {1 355, 2 36, 4 7, 6 3, 3 18, 5 2, 7 4, 11 1}
           (-> tf :tf frequencies)))
    
    (is ( = [1 2 3 4 5 6 7 8 9 10]
          (-> tf-with-index (tc/head 10) :word-idx ))
        )
    ))





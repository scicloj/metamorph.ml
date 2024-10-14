(ns scicloj.metamorph.text-test
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.text :as text]
            [tech.v3.dataset.base :as ds-base]
            [tablecloth.api :as tc]))



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
            (tc/drop-rows #(empty? (:term %)))
            (tc/drop-rows #(nil? (:term %)))
            (tc/drop-rows #(= "" (:term %))))
        tf
        (->
         (text/->term-frequency text))
        tf-with-index
        (->
         (text/add-word-idx tf))]

    (is (= 576
           (tc/row-count text)))

    (is (= '(:term :term-index :document :meta)
           (tc/column-names text)))

    (is (= [["Is" 0 0 3] ["it" 1 0 3] ["a" 2 0 3] ["great" 3 0 3] ["product" 4 0 3]]
           (-> text
               (tc/head)
               (tc/rows :maps))))
    (is (= ["Is" "Is" "it" "it" "it"] (take 5 (-> tf :term))))
    (is (= {0 68, 1 24, 2 136, 3 135, 4 63}
           (-> tf :document frequencies)))
    (is (= {1 355, 2 36, 4 7, 6 3, 3 18, 5 2, 7 4, 11 1}
           (-> tf :term-count frequencies)))

    (is (= [1 1 2 2 2 3 3 3 3 4]
           (-> tf-with-index (tc/head 10) :term-idx)))))

(deftest tfidf
  (let [tidy
        (->
         (text/->tidy-text
          (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
           (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
      ;(io/reader "test/data/reviews.csv")
          parse-review-line
          #(str/split % #" ")
          :max-lines 5
          :skip-lines 0)
         (tc/drop-missing)
         (tc/drop-rows #(empty? (:term %)))
         (tc/drop-rows #(nil? (:term %)))
         (tc/drop-rows #(= "" (:term %)))
         (tc/rename-columns {:meta :label}))


        tf (text/->term-frequency tidy)]

    (is (= '("this" "this" "is" "is" "a" "sample" "another" "example")
           (-> tf :term seq)))

    (is (=
         ["0.20000000298023224"
          "0.1428571492433548"
          "0.20000000298023224"
          "0.1428571492433548"
          "0.4000000059604645"
          "0.20000000298023224"
          "0.2857142984867096"
          "0.4285714328289032"]
         (map str (-> tf :tf))))

    (is (= '("0.0" "0.0" "0.0" "0.0" "0.12041200005987107" "0.060206000029935536" "0.08600857403459163" "0.12901285656619094")
           (map str (-> tf :tfidf))))))


(comment

  (def tidy
    (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                          parse-review-line
                          #(str/split % #" ")
                        ;:max-lines 5
                          :skip-lines 1)
        (tc/drop-missing)
        (tc/drop-rows #(empty? (:term %)))
        (tc/drop-rows #(nil? (:term %)))
        (tc/drop-rows #(= "" (:term %)))))
  
  (def tf
    (->
     (text/->term-frequency tidy)))
  

  (def
    tf-with-index
    (->
     (text/add-word-idx tf)))
  )

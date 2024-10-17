(ns scicloj.metamorph.text-test
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.text :as text]
            [tablecloth.api :as tc]
            [tech.v3.datatype :as dt]
            [tech.v3.dataset :as ds]

            )
  (:import [tech.v3.datatype.native_buffer NativeBuffer])
  )


(defn- parse-review-line [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     (dec (Integer/parseInt (second splitted)))]))

(defn- parse-review-line-as-maps [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     {:label 
      (dec (Integer. (second splitted)))}]))

(deftest containertype

  (let [{:keys [dataset]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          parse-review-line
                          #(str/split % #" ")
                          :max-lines 5
                          :skip-lines 1
                          :container-type :native-heap
                          :datatype-metas :int16)]

    (is  (instance? NativeBuffer
                    (-> dataset :document .data)))
    (is (= 3
           (-> dataset :meta first)))))


(deftest ->tidy-text-with-object

  (let [{:keys [dataset]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          parse-review-line-as-maps
                          #(str/split % #" ")
                          :max-lines 5
                          :skip-lines 1
                          :container-type :jvm-heap
                          :datatype-metas :object)]
    (is (not (instance? NativeBuffer (-> dataset :document .data))))

    (is (= {:label 3}
           (-> dataset :meta first)))))

(deftest ->tidy-text
  (let [tidy
        (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                              parse-review-line
                              #(str/split % #" ")
                              :max-lines 5
                              :skip-lines 1))

        text (:dataset tidy)
        int->str (:int->str tidy)
        tf (text/->term-frequency text)]
    (is (= 596
           (tc/row-count text)))

    (is (= '(:term-idx :term-pos :document :meta)
           (tc/column-names text)))

    (is (not (instance? NativeBuffer (-> text :term-idx .data))))
    (is (not (instance? NativeBuffer (-> text :term-pos .data))))
    (is (not (instance? NativeBuffer (-> text :document .data))))
    (is (not (instance? NativeBuffer (-> text :meta .data))))

    (is (= :int16 (-> text :term-idx meta :datatype)))
    (is (= :int16 (-> text :term-pos meta :datatype)))
    (is (= :int32 (-> text :document meta :datatype)))
    (is (= :int8 (-> text :meta meta :datatype)))

    (is (= [["Is" 0 0 3] ["it" 1 0 3] ["a" 2 0 3] ["great" 3 0 3] ["product" 4 0 3]]
           (->>
            (-> text
                (tc/head)
                (tc/rows :maps))
            (map (fn [[term-index a b c]]
                   [(int->str term-index) a b c])))))


    (is (=
         {0 68, 3 136, 4 64, 2 137, 1 24}
         (-> tf :document frequencies)))
    (is (=
         {7 4, 1 356, 4 7, 13 1, 6 4, 3 18, 2 36, 11 1, 5 2}
         (-> tf :term-count frequencies)))

    (is (= [1 1 2 2 2 3 3 3 3 4]
           (-> tf (tc/head 10) :term-idx)))))

(deftest tfidf
  (let [ds-and-st

        (text/->tidy-text
         (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
      ;(io/reader "test/data/reviews.csv")
         parse-review-line
         #(str/split % #" ")
         :max-lines 5
         :skip-lines 0)

       text 
        (-> (:dataset ds-and-st)
        (tc/rename-columns {:meta :label}))


        tf (text/->term-frequency text)]

    (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
           '(1 1 2 2 3 4 5 6)
           (-> tf :term-idx seq)))

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


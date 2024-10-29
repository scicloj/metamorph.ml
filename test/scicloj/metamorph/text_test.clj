(ns scicloj.metamorph.text-test
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.set :as c-set]
   [clojure.string :as str]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]
   [criterium.core :as criterium]
   [tech.v3.dataset :as ds])
  (:import
   [org.mapdb DBMaker]
   [tech.v3.datatype.native_buffer NativeBuffer]))


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


  (let [{:keys [datasets]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          line-seq
                          parse-review-line
                          #(str/split % #" ")
                          :max-lines 5
                          :skip-lines 1
                          :container-type :native-heap
                          :datatype-metas :int16)
        dataset (first datasets)]


    (is (str/includes?
         (-> dataset :document .data class .getName)
         "native_buffer"))
    (is (= 3
           (-> dataset :meta first)))))

(deftest ->tidy-text2--document-distinct

  (let [{:keys [datasets]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          line-seq
                          parse-review-line-as-maps
                          #(str/split % #" ")
                          :max-lines 7
                          :skip-lines 1
                          :container-type :jvm-heap
                          :datatype-metas :object
                          :compacting-document-intervall 10)
        df (first datasets)]
    
    (is (= (range 7)
           (-> df :document distinct)))

    ))

(deftest ->tidy-text2--document-distinct-2

  (let [{:keys [datasets]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          line-seq
                           parse-review-line-as-maps
                           #(str/split % #" ")
                           :max-lines 7
                           :skip-lines 1
                           :container-type :jvm-heap
                           :datatype-metas :object
                           :compacting-document-intervall 3)
        df (first datasets)]

    (def df df)
    (is (= (range 7)
           (-> df :document distinct)))))



(defn reviews->tidy [tidy-text-fn combine-method]
  (-> (tidy-text-fn (io/reader "test/data/reviews.csv")
                    line-seq
                    parse-review-line
                    #(str/split % #" ")
                    :max-lines 5
                    :skip-lines 1
                    :combine-method combine-method)))

(defn- validate-tidy-and-tf [tidy]
  (let [
        text (first (:datasets tidy))
        token->long (:token->long tidy)
        int->str (c-set/map-invert token->long)
        tf (->
            (text/->tfidf text)
            (tc/order-by [:document :term-idx]))]

    (is (= 596
           (tc/row-count text)))

    (is (= '(:term-idx :term-pos :document :meta)
           (tc/column-names text)))

    (is (not (instance? NativeBuffer (-> text :term-idx .data))))
    (is (not (instance? NativeBuffer (-> text :term-pos .data))))
    (is (not (instance? NativeBuffer (-> text :document .data))))
    (is (not (instance? NativeBuffer (-> text :meta .data))))

    (is (= :int32 (-> text :term-idx meta :datatype)))
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

    (is (= '("50-60" "Did" "Don't" "Is" "It" "Just" "No." "So," "Yes." "a" "and" "anyway?" "around" "box" "could" "difficult" "do" "gift" "gifts." "girlfriend's" "go" "goodies." "great" "has" "how" "if" "is." "it" "it's" "it." "know" "like" "list" "love" "make" "my" "of" "offending" "old" "on" "or" "parents" "people" "person" "product" "receiving" "recipient" "seems" "since" "someone" "specific" "sure" "sweet" "than" "that's" "the" "there's" "this" "thoughtful" "to" "tooth," "value?" "want" "with" "worse" "years" "you" "your")
           (->>
            (map int->str
                 (-> tf
                     (tc/select-rows (fn [row] (= 0 (:document row))))
                     :term-idx))
            sort)))))

(defn tidy-text-test [tidy-text-fn combine-method]
  (let [tidy (reviews->tidy tidy-text-fn combine-method)]
    (validate-tidy-and-tf tidy)))




(deftest tidy-text-test--from-df
  (let [tidy
        (text/->tidy-text
         (tc/dataset "test/data/reviews.csv")
         (fn [df] (map str (-> df (get "Text"))))
         (fn [line] [line 3])
         #(str/split % #" ")
                              ;:max-lines 100000
         :skip-lines 0
         :max-lines 5)]
    (validate-tidy-and-tf tidy)))
        




(deftest tidy-text--coalesce-blocks
  (tidy-text-test text/->tidy-text :coalesce-blocks!))

(deftest tidy-text--concat-buffers
  (tidy-text-test text/->tidy-text :concat-buffers))




(defn validate-tfidf [tidy->text-fn]
  (let [ds-and-st

        (tidy->text-fn
         (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
         line-seq
         parse-review-line
         #(str/split % #" ")
         :max-lines 5
         :skip-lines 0)

        text
        (-> (first (:datasets ds-and-st))
            (tc/rename-columns {:meta :label}))

        tfidfs
        (->
         (text/->tfidf text)
         (tc/order-by [:term-idx :document :label :term-count :tf :idf :tfidf]))

        tfidfs-native-heap
        (->
         (text/->tfidf text :container-type :native-heap)
         (tc/order-by [:term-idx :document :label :term-count :tf :idf :tfidf]))]

    
    (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
         '(1 1 2 2 3 4 5 6)

         (-> tfidfs
             (tc/order-by :term-idx)
             :term-idx seq)))

    (is (=
         ["0.2"
          "0.14285715"
          "0.2"
          "0.14285715"
          "0.4"
          "0.2"
          "0.2857143"
          "0.42857143"]
         (map str (-> tfidfs :tf))))

    (is (=
         ["0.20000000298023224"
          "0.1428571492433548"
          "0.20000000298023224"
          "0.1428571492433548"
          "0.4000000059604645"
          "0.20000000298023224"
          "0.2857142984867096"
          "0.4285714328289032"]
         (map str (-> tfidfs-native-heap :tf))))

    (is (= '("0.0" "0.0" "0.0" "0.0"
                   "0.12041201"
                   "0.060206003"
                   "0.08600857"
                   "0.12901287")
           (map str (-> tfidfs :tfidf))))))

(defn- text->string-table [db]
  (let [string-table (text/mapdb-string-table db)]
    (text/fill-string-table!
     (io/reader "test/data/reviews.csv")
     string-table
     parse-review-line
     #(str/split % #" ")
     10
     1)
    string-table))



(deftest test-tidy-df2
  (validate-tfidf text/->tidy-text))

(deftest fill-string-table-memory-db!
  (let [memory-db
        (.. DBMaker
            memoryDB
            make)]
    (is (= ["Is" "it" "a" "great" "product" "or" "great" "value?"]
           (take 8 (text->string-table memory-db))))))


(deftest fill-string-table--temp-file!
  (let [temp-file-db
        (.. DBMaker
            (tempFileDB)
            fileMmapEnable
            make)]
    (is (= ["Is" "it" "a" "great" "product" "or" "great" "value?"]
           (take 8 (text->string-table temp-file-db))))))





(ns scicloj.metamorph.text-test
  (:require
   [clojure.data.csv :as csv]
   [clojure.edn :as edn]
   [clojure.java.io :as io]
   [clojure.set :as c-set]
   [clojure.string :as str]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds])
  (:import
   [org.mapdb DBMaker]
   [tech.v3.datatype.native_buffer NativeBuffer]))

(defonce reviews-7l
  (edn/read-string
   (slurp "test/data/reviews_7l.edn")
  ))

(defn pr-edn-str [& xs]
  (binding [*print-length* nil
            *print-dup* nil
            *print-level* nil
            *print-readably* true]
    (apply pr-str xs)))

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
                          :datatype-meta :int16)
        dataset (first datasets)]


    (is (str/includes?
         (-> dataset :document .data class .getName)
         "native_buffer"))
    (is (= 3
           (-> dataset :meta first)))))

(defn- validate-df-when-compating [compacting-document-intervall]
  (let [{:keys [datasets]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          line-seq
                          parse-review-line-as-maps
                          #(str/split % #" ")
                          :max-lines 7
                          :skip-lines 1
                          :container-type :jvm-heap
                          :datatype-meta :object
                          :compacting-document-intervall compacting-document-intervall)
        df (first datasets)]

    
    (is (= reviews-7l
           (ds/rowvecs df)))
    (is (= (range 7)
           (-> df :document distinct)))))


(deftest ->tidy-text--compact
  (validate-df-when-compating 1)
  (validate-df-when-compating 2)
  (validate-df-when-compating 10)
  (validate-df-when-compating 1000))



(defn reviews->tidy [combine-method compacting-document-intervall]
  (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                    line-seq
                    parse-review-line
                    #(str/split % #" ")
                    :datatype-meta :int16    
                    :max-lines 5
                    :skip-lines 1
                    :compacting-document-intervall compacting-document-intervall    
                    :combine-method combine-method)))

(defn- validate-tidy-and-tf [tidy expected-meta]
  (def tidy tidy)
  (def expected-meta expected-meta)
  (let [
        text (first (:datasets tidy))
        token-lookup-table (:token-lookup-table tidy)
        int->str (c-set/map-invert token-lookup-table)
        _ (def text text)
        tf (->
            (text/->tfidf text)
            (tc/order-by [:document :token-idx]))]

    (is (= 596
           (tc/row-count text)))

    (is (= 
         (if  expected-meta
           '(:token-idx :token-pos :document :meta)
           '(:token-idx :token-pos :document))
           (tc/column-names text)))

    (is (not (instance? NativeBuffer (-> text :token-idx .data))))
    (is (not (instance? NativeBuffer (-> text :token-pos .data))))
    (is (not (instance? NativeBuffer (-> text :document .data))))
    (is 
     (if expected-meta
       (not (instance? NativeBuffer (-> text :meta .data)))
       true))

    (is (= :int16 (-> text :token-idx meta :datatype)))
    (is (= :int16 (-> text :token-pos meta :datatype)))
    (is (= :int16 (-> text :document meta :datatype)))
    (def text text)
    (def expected-meta expected-meta)
    (def tf tf)

    (is (= (if expected-meta 
             :int16
             nil) 
           (-> text :meta meta :datatype)))

    (is (= [["Is" 0 0 expected-meta] 
            ["it" 1 0 expected-meta] 
            ["a" 2 0 expected-meta] 
            ["great" 3 0 expected-meta] 
            ["product" 4 0 expected-meta]]
           (->>
            (-> text
                (tc/head)
                (tc/rows :maps))
            (map (fn [[token-index a b c]]
                   [(int->str token-index) a b c])))))



    (is (=
         {0 68, 3 136, 4 64, 2 137, 1 24}
         (-> tf :document frequencies)))
    (is (=
         {7 4, 1 356, 4 7, 13 1, 6 4, 3 18, 2 36, 11 1, 5 2}
         (-> tf :token-count frequencies)))

    (is (= '("50-60" "Did" "Don't" "Is" "It" "Just" "No." "So," "Yes." "a" "and" "anyway?" "around" "box" "could" "difficult" "do" "gift" "gifts." "girlfriend's" "go" "goodies." "great" "has" "how" "if" "is." "it" "it's" "it." "know" "like" "list" "love" "make" "my" "of" "offending" "old" "on" "or" "parents" "people" "person" "product" "receiving" "recipient" "seems" "since" "someone" "specific" "sure" "sweet" "than" "that's" "the" "there's" "this" "thoughtful" "to" "tooth," "value?" "want" "with" "worse" "years" "you" "your")
           (->>
            (map int->str
                 (-> tf
                     (tc/select-rows (fn [row] (= 0 (:document row))))
                     :token-idx))
            sort)))))

(defn tidy-text-test [combine-method compacting-document-intervall]
  (let [tidy (reviews->tidy combine-method compacting-document-intervall)]
    (validate-tidy-and-tf tidy 3)))

(deftest tidy-text-test--from-df
  (let [tidy
        (text/->tidy-text
         (tc/dataset "test/data/reviews.csv")
         (fn [df] (map str (-> df (get "Text"))))
         (fn [line] [line 3])
         #(str/split % #" ")
                      
         :datatype-meta :int16
         :skip-lines 0
         :max-lines 5)]
    (validate-tidy-and-tf tidy 3)))
        

(deftest tidy-text-test--from-df-no-meta
  (let [tidy
        (text/->tidy-text
         (tc/dataset "test/data/reviews.csv")
         (fn [df] (map str (-> df (get "Text"))))
         (fn [line] [line nil])
         #(str/split % #" ")
                              ;:max-lines 100000
         :skip-lines 0
         :max-lines 5)]
    (validate-tidy-and-tf tidy nil)))


(deftest tidy-text--coalesce-blocks
  (tidy-text-test :coalesce-blocks! 10000)
  (tidy-text-test :concat-buffers 10000)
  (tidy-text-test :coalesce-blocks! 2)
  (tidy-text-test :concat-buffers 2)
  )


(deftest toke->index-map
  (let [token->index-map
        (.. DBMaker
            memoryDB
            make
            (hashMap "map")
            counterEnable
            createOrOpen)]

    (text/->tidy-text
     (io/reader
      (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
     line-seq
     parse-review-line
     #(str/split % #" ")
     :token->index-map token->index-map)

    (is (=
         {"" 0, "a" 3, "another" 5, "is" 2, "this" 1, "sample" 4, "example" 6}
         token->index-map))))


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
         (tc/order-by [:token-idx :document :label :token-count :tf :idf :tfidf]))

        tfidfs-native-heap
        (->
         (text/->tfidf text :container-type :native-heap)
         (tc/order-by [:token-idx :document :label :token-count :tf :idf :tfidf]))]

    
    (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
         '(1 1 2 2 3 4 5 6)

         (-> tfidfs
             (tc/order-by :token-idx)
             :token-idx seq)))

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





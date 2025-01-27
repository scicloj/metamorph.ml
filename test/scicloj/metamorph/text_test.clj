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
   [tech.v3.dataset :as ds]
   [fastmath.vector :as fmv])
  (:import
   [org.mapdb DBMaker]
   [tech.v3.datatype.native_buffer NativeBuffer]))



(def reviews-7l
  (edn/read-string
   (slurp "test/data/reviews_7l.edn")))

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

(defn validate-containertype [column-container-type]
  (let [{:keys [datasets]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          line-seq
                          parse-review-line
                          #(str/split % #" ")
                          :max-lines 5
                          :skip-lines 1
                          :column-container-type :native-heap
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
  (let [text (first (:datasets tidy))
        token-lookup-table (:token-lookup-table tidy)
        int->str (c-set/map-invert token-lookup-table)
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
  (tidy-text-test :concat-buffers 2))



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
         {"[UNKNOWN]" 0, "a" 3, "another" 5, "is" 2, "this" 1, "sample" 4, "example" 6}
         token->index-map))))


(defn validate-tfidf [tidy-column-container-type
                      tfidf-column-container-type
                      tidy-container-type
                      tfidf-container-type
                      tidy-combine-method
                      tfidf-combine-method
                      compacting-document-intervall]
  (let [ds-and-st

        (text/->tidy-text
         (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
         line-seq
         parse-review-line
         #(str/split % #" ")
         :column-container-type tidy-column-container-type
         :container-type tidy-container-type
         :combine-method tidy-combine-method
         :compacting-document-intervall compacting-document-intervall)

        text
        (-> (first (:datasets ds-and-st))
            (tc/rename-columns {:meta :label}))

        tfidfs
        (->
         (text/->tfidf text
                       :column-container-type tfidf-column-container-type
                       :container-type tfidf-container-type
                       :combine-method tfidf-combine-method)
         (tc/order-by [:token-idx :document :label :token-count :tf :idf :tfidf]))]


    (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
         '(1 1 2 2 3 4 5 6)

         (-> tfidfs
             (tc/order-by :token-idx)
             :token-idx seq)))

    (is (fmv/edelta-eq
         [0.2
          0.1429
          0.2
          0.1429
          0.4
          0.2
          0.2857
          0.4286]
         (-> tfidfs :tf)
         0.001))



    (is (fmv/edelta-eq
         '(0.0  0.0  0.0  0.0
                0.12041201
                0.060206003
                0.08600858
                0.12901287)
         (-> tfidfs :tfidf)
         0.001))))


(defmacro cart [& lists]
  (let [syms (for [_ lists] (gensym))]
    `(for [~@(mapcat list syms lists)]
       (list ~@syms))))


(deftest test-tidy-tfidf
  (->>
   (cart
    [:jvm-heap :native-heap :mmap]
    [:jvm-heap :native-heap :mmap]
    [:jvm-heap :native-heap :mmap]
    [:jvm-heap :native-heap :mmap]
    [:coalesce-blocks! :concat-buffers]
    [:coalesce-blocks! :concat-buffers]
    [1 2 10])
   (run! (partial apply validate-tfidf))))

(deftest meta-is-present-in-tfidf
  (let [tfidf
        (->
         (text/->tidy-text (io/reader "test/data/reviews.csv")
                           line-seq
                           parse-review-line
                           #(str/split % #" ")
                           :max-lines 5
                           :skip-lines 1
                           :container-type :native-heap
                           :column-container-type :native-heap)
         :datasets
         first
         (text/->tfidf))]
    (is (= {3 68, 4 225, 2 136}
           (frequencies (:meta tfidf))))
    (is (= '(:document  :tfidf :tf :token-idx :token-count :meta)
           (tc/column-names tfidf)))))
(defn- tidy-wiki []
  (text/->tidy-text
   (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
   line-seq
   parse-review-line
   #(str/split % #" ")))


(deftest unknow-maps-to-0
  (let [first-tidy-result (tidy-wiki)
        token-lookup-table (:token-lookup-table first-tidy-result)

        second-tidy-result
        (text/->tidy-text
         (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2\nEXTRA,2"))
         line-seq
         parse-review-line
         #(str/split % #" ")
         :new-token-behaviour :as-unknown
         :token->index-map token-lookup-table)]

    (is (= 0 (last (-> second-tidy-result :datasets first :token-idx))))))



(deftest new-index-on-:store
  (let [first-tidy-result
        (tidy-wiki)
        mutable-token-lookup-table (java.util.HashMap. (:token-lookup-table first-tidy-result))
        second-tidy-result
        (text/->tidy-text
         (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
          (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2\nEXTRA,2"))
         line-seq
         parse-review-line
         #(str/split % #" ")
         :new-token-behaviour :store
         :token->index-map mutable-token-lookup-table)]

    (is (= 7 (last (-> second-tidy-result :datasets first :token-idx))))))


(deftest ->svmlib
  (let [f (java.io.File/createTempFile "tfidf" ".txt")]
    (.deleteOnExit f)
    (-> (tidy-wiki)
        :datasets
        first
        (text/->tfidf)
        (text/tidy->libsvm! (io/writer f) :tfidf))
    (is (= "0 1:0.0 2:0.0 3:0.12041201 4:0.060206003\n1 1:0.0 2:0.0 5:0.08600858 6:0.12901287\n"
           (slurp f)))))


(deftest libsvm->tidy
  (is (=
       [{:instance 0, :index 1, :value -0.555556, :label 1} {:instance 0, :index 2, :value 0.25, :label 1}]
       (->
        (text/libsvm->tidy (io/reader "test/data/iris.libsvm.txt"))
        (tc/head 2)
        (tc/rows :as-maps)))))
  


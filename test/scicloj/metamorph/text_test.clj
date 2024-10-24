(ns scicloj.metamorph.text-test
   (:require
    [clojure.data.csv :as csv]
    [clojure.java.io :as io]
    [clojure.string :as str]
    [clojure.test :refer [deftest is]]
    [scicloj.metamorph.ml.text :as text]
    [tablecloth.api :as tc]
    [tech.v3.dataset.string-table :as st])
   (:import
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
                           parse-review-line
                           #(str/split % #" ")
                           (st/make-string-table)
                           :max-lines 5
                           :skip-lines 1
                           :container-type :native-heap
                           :datatype-metas :int16)
         dataset (first datasets)]

     
     (is (str/includes? 
                        (-> dataset :document .data class .getName)
          "concat"))
     (is (= 3
            (-> dataset :meta first)))))


 (deftest ->tidy-text-with-object

   (let [{:keys [datasets]}
         (text/->tidy-text (io/reader "test/data/reviews.csv")
                           parse-review-line-as-maps
                           #(str/split % #" ")
                           (st/make-string-table)
                           :max-lines 5
                           :skip-lines 1
                           :container-type :jvm-heap
                           :datatype-metas :object)
         dataset (first datasets)]
     (is (not (instance? NativeBuffer (-> dataset :document .data))))

     (is (= {:label 3}
            (-> dataset :meta first)))))

 (deftest ->tidy-text
   (let [tidy
         (-> (text/->tidy-text (io/reader "test/data/reviews.csv")
                               parse-review-line
                               #(str/split % #" ")
                               (st/make-string-table)
                               :max-lines 5
                               :skip-lines 1))

         text (first (:datasets tidy))
         int->str (:int->str tidy)
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

     (is (= '("50-60" "Did" "Don't" "Is" "It" "Just" "No." "So," "Yes." "a" "and" "anyway?" "around" "box" "could" "difficult" "do" "gift" "gifts." "girlfriend's" "go" "goodies." "great" "has" "how" "if" "is." "it" "it's" "it." "know" "like" "list" "love" "make" "my" "of" "offending" "old" "on" "or" "parents" "people" "person" "product" "receiving" "recipient" "seems" "since" "someone" "specific" "sure" "sweet" "than" "that's" "the" "there's" "this" "thoughtful" "to" "tooth," "value?" "want" "with" "worse" "years" "you" "your")
            (->>
             (map int->str
                  (-> tf
                      (tc/select-rows (fn [row] (= 0 (:document row))))
                      :term-idx))
             sort)))

     ))

 (deftest tfidf
   (let [ds-and-st

         (text/->tidy-text
          (io/reader
      ;;https://en.wikipedia.org/wiki/Tf%E2%80%93idf
           (java.io.StringReader. "this is a a sample,1\nthis is another another example example example,2"))
      ;(io/reader "test/data/reviews.csv")
          parse-review-line
          #(str/split % #" ")
          (st/make-string-table)
          :max-lines 5
          :skip-lines 0)

         text
         (-> (first (:datasets ds-and-st))
             (tc/rename-columns {:meta :label}))

         tfidfs 
         (->
          (text/->tfidf text)
          (tc/order-by [:term-idx :document :label :term-count :tf :idf :tfidf])
          )

         tfidfs-native-heap
         (->
          (text/->tfidf text :container-type :native-heap)
          (tc/order-by [:term-idx :document :label :term-count :tf :idf :tfidf])
          )
         
         tfidfs-native-buffer
         (->
          (text/->tfidf text :container-type :native-buffer)
          (tc/order-by [:term-idx :document :label :term-count :tf :idf :tfidf])
          )
         ]


     
     
     (-> tfidfs :tfidf .data)
     ;;=> tech.v3.datatype.io_indexed_buffer$indexed_buffer$reify__16993
     

     (-> tfidfs-native-buffer :tfidf .data class)
     ;;=> tech.v3.datatype.io_indexed_buffer$indexed_buffer$reify__16993

     (-> tfidfs-native-heap
         :tf .data
         class)

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
                    "0.08600858"
                    "0.12901287")
            (map str (-> tfidfs :tfidf))))))




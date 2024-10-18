(ns scicloj.metamorph.text-test
   (:require [clojure.data.csv :as csv]
             [clojure.java.io :as io]
             [clojure.string :as str]
             [clojure.test :refer [deftest is]]
             [scicloj.metamorph.ml.text :as text]
             [tech.v3.datatype.functional :as fun]
             [tablecloth.api :as tc]
             [tech.v3.datatype :as dt]
             [tech.v3.dataset :as ds]
             [ham-fisted.api :as hf])
   (:import [tech.v3.datatype.native_buffer NativeBuffer]))


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


         
         tfidf 
         (->
          (text/->tfidf text)
          (tc/order-by [ :term-idx :document :label :term-count :tf :idf :tfidf ])
          )

         ]

     (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
          '(1 1 2 2 3 4 5 6)

          (-> tfidf :term-idx seq)))

     (def tfidf tfidf)
     (is (=
          ["0.2"
           "0.14285714285714285"
           "0.2"
           "0.14285714285714285"
           "0.4"
           "0.2"
           "0.2857142857142857"
           "0.42857142857142855"]
          (map str (-> tfidf :tf))))

     (is (= '("0.0" "0.0" "0.0" "0.0" 
              "0.12041199826559248" 
              "0.06020599913279624" 
              "0.08600857018970891" 
              "0.12901285528456335")
            (map str (-> tfidf :tfidf))))))

 (comment

   (require '[clj-memory-meter.core :as mm])
   (defn load-reviews []
     (-> (text/->tidy-text
          (io/reader "repeatedAbstrcats_3.7m_.txt")
          (fn [line] [line
                      (rand-int 6)])
          #(str/split % #" ")
          :max-lines 10000
          :skip-lines 1)))


   (def reviews (load-reviews))
   (def reviews-text (:dataset reviews))


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
      (text/add-word-idx tf))))


 (comment
   (require '[tech.v3.dataset :as ds])

   (require '[ham-fisted.reduce :as hf-reduce]
            '[ham-fisted.api :as hf])



   (defn ->tf [text]
     (->>
      (hf-reduce/preduce (fn [] (hf/object-array-list))
                         (fn [l [document-idx row-indices]]
                           (let [terms
                                 (ds/select-rows (:term text) row-indices)
                                 freqs (hf/frequencies terms)
                                 n-terms (hf/constant-count row-indices)]
                             
                             
                             (-> l
                                 (hf/conj!
                                  (hf/hash-map :document (hf/repeat (hf/constant-count freqs) document-idx)
                                               :term (hf/keys freqs)
                                               :freq (hf/vals freqs)
                                               :n-terms (hf/repeat (hf/constant-count freqs) n-terms))))))
                         (fn [list-1 list-2]
                           (hf/add-all! list-1 list-2))
                         (ds/group-by-column->indexes text :document))
      (hf/union-reduce-maps
       (fn [m-1 m-2]
         (hf/concatv m-1 m-2)))
      (ds/->>dataset)))


   (time
    (def tf
      (->tf (ds/->dataset
       ;     text
             {:document   [0     0      0    0  0   1   1     1      1   1     1     1]
              :term       ["I" "like" "fish" "fish" "fish"      "fish" "is" "fish" "and" "I" "like" "it"]}))))
   )
 


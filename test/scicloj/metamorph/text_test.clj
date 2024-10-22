(ns scicloj.metamorph.text-test
   (:require [clojure.data.csv :as csv]
             [clojure.java.io :as io]
             [clojure.string :as str]
             [clojure.test :refer [deftest is]]
             [scicloj.metamorph.ml.text :as text]
             [tablecloth.api :as tc]
             [tech.v3.dataset :as ds]
             [tech.v3.dataset.string-table :as st])
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
                           (st/make-string-table)
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
                           (st/make-string-table)
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
                               (st/make-string-table)
                               :max-lines 5
                               :skip-lines 1))

         text (:dataset tidy)
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


     (def tf tf)
     (is (=
          {0 68, 3 136, 4 64, 2 137, 1 24}
          (-> tf :document frequencies)))
     (is (=
          {7 4, 1 356, 4 7, 13 1, 6 4, 3 18, 2 36, 11 1, 5 2}
          (-> tf :term-count frequencies)))

     (is (= [1 2 3 4 5 6 7 8 9 10]
            (-> tf
                (tc/order-by :term-idx)
                (tc/head 10) :term-idx)))))

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
         (-> (:dataset ds-and-st)
             (tc/rename-columns {:meta :label}))

         tfidfs 
         (->
          (text/->tfidf text)
          (tc/order-by [ :term-idx :document :label :term-count :tf :idf :tfidf ])
          )

         ]

    (def text text)
    (def tfidfs tfidfs)
    (ds/rows text)   
    ;;=> [{:term-idx 1, :term-pos 0, :document 0, :label 0} {:term-idx 2, :term-pos 1, :document 0, :label 0} {:term-idx 3, :term-pos 2, :document 0, :label 0} {:term-idx 3, :term-pos 3, :document 0, :label 0} {:term-idx 4, :term-pos 4, :document 0, :label 0} {:term-idx 1, :term-pos 0, :document 1, :label 1} {:term-idx 2, :term-pos 1, :document 1, :label 1} {:term-idx 5, :term-pos 2, :document 1, :label 1} {:term-idx 5, :term-pos 3, :document 1, :label 1} {:term-idx 6, :term-pos 4, :document 1, :label 1} {:term-idx 6, :term-pos 5, :document 1, :label 1} {:term-idx 6, :term-pos 6, :document 1, :label 1}]

     (is (= ;;'("this" "this" "is" "is" "a" "sample" "another" "example")
          '(1 1 2 2 3 4 5 6)

          (-> tfidfs :term-idx seq)))

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

     (is (= '("0.0" "0.0" "0.0" "0.0" 
              "0.12041201" 
              "0.060206003" 
              "0.08600858" 
              "0.12901287")
            (map str (-> tfidfs :tfidf))))))



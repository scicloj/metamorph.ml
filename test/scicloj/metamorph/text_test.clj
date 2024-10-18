(ns scicloj.metamorph.text-test
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.text :as text]
            [tech.v3.dataset.string-table :as string-table]
            [tablecloth.api :as tc]
            [tech.v3.dataset.string-table :as st]))

(defn- parse-review-line [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     (dec (Integer/parseInt (second splitted)))]))

(defn- parse-review-line-as-mape [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     {:label 
      (dec (Integer. (second splitted)))}]))

(deftest ->tidy-text-with-objec-d

  (let [{:keys [dataset]}
        (text/->tidy-text (io/reader "test/data/reviews.csv")
                          parse-review-line-as-mape
                          #(str/split % #" ")
                          :max-lines 5
                          :skip-lines 1
                          :datatype-metas :object
                          
                          )]
    (is ( = {:label 3}
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
        tf (text/->term-frequency text)
        ]
    (is (= 596
           (tc/row-count text)))

    (is (= '(:term-idx :term-pos :document :meta)
           (tc/column-names text)))

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

(comment


  (require '[clj-memory-meter.core :as mm])
  



  (defn load-reviews [] 
    (-> (text/->tidy-text 
         (io/reader "repeatedAbstrcats_3.7m_.txt")
         (fn [line] [line 
                     (rand-int 6)])
         #(str/split % #" ")
         :max-lines 10000
         :skip-lines 1) ))
  
  

  (def reviews (load-reviews))
  
  (def reviews-text (:dataset reviews))
  
  




  (mm/measure
   (:term-idx reviews-text))
  
;;=> "318.8 KiB"
  
  (mm/measure
   (meta (:term-pos reviews-text)))
  
;;=> "1.0 MiB"
  
  (mm/measure
   (:document reviews-text))
  
;;=> "45.7 KiB"
  
  (mm/measure
   (:meta reviews-text)
   )
  
  
;;=> "1.0 MiB"
  
;;=> "2.4 MiB"
  
  (println 
   :tc-row-count
   (tc/row-count (:dataset reviews)))
  

  (println
   :reviews-text
   (mm/measure reviews-text))
  )

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


(require '[tech.v3.dataset :as ds])
(require '[ham-fisted.reduce :as hf-reduce]
         '[ham-fisted.api :as hf])


(defn ->tf [text]
  (hf-reduce/preduce (fn [] (hf/object-array-list))
                     (fn [l [document-idx row-indices]]
                       (let [terms
                             (ds/select-rows (:term text) row-indices)
                             freqs (hf/frequencies terms)]

                         (-> l
                             (hf/conj! (hf/repeat (hf/constant-count freqs) document-idx))
                             (hf/conj! (hf/keys freqs))
                             (hf/conj! (hf/vals freqs)))


                         ))
                     (fn [list-1 list-2]
                       (hf/add-all! list-1 list-2)
                       )
                     (ds/group-by-column->indexes text :document)))
(time
 (def tf
   (->tf (ds/->dataset
          text
       ;{:document   [0     0      0    0  0   1   1     1      1   1     1     1]
       ; :term       ["I" "like" "fish" "fish" "fish"      "fish" "is" "fish" "and" "I" "like" "it"]}
          ))))
  ;;=> [[0 0 0] (I like fish) (1 1 3) [1 1 1 1 1 1] (like and fish I is it) (1 1 2 1 1 1)]
  
(class
 (nth tf 2))
(ns measure-tfidf
  (:require
   [clojure.data.csv :as csv]
   [clojure.java.io :as io]
   [clojure.string :as str]
   [criterium.core :as crit]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]))

(defn- parse-review-line [line]
  (let [splitted (first
                  (csv/read-csv line))]
    [(first splitted)
     (dec (Integer/parseInt (second splitted)))]))


(defn do-tfidf-reviews [tidy-column-container-type
                              tfidf-column-container-type
                              tidy-container-type
                              tfidf-container-type
                              tidy-combine-method
                              tfidf-combine-method
                              compacting-document-intervall]
  (let [ds-and-st

        (text/->tidy-text
         (io/reader "test/data/reviews.csv")
         line-seq
         parse-review-line
         #(str/split % #" ")
         :skip-lines 1
         ;:max-line 1000

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
                       :combine-method tfidf-combine-method))]
    tfidfs))


(defmacro cart [& lists]
  (let [syms (for [_ lists] (gensym))]
    `(for [~@(mapcat list syms lists)]
       (list ~@syms))))


(def parameters
  (cart
   [:jvm-heap :native-heap :mmap]
   [:jvm-heap :native-heap :mmap]
   [:jvm-heap :native-heap :mmap]
   [:jvm-heap :native-heap :mmap]
   [:coalesce-blocks! :concat-buffers]
   [:coalesce-blocks! :concat-buffers]
   [10 20 500])

  )

(def test-tidy-tfidf-crit
  (->>
   parameters   
   (map-indexed (fn [index [& args]]
           (println index "/" (count parameters) ":" args)
           {args
            (/
             (first

              (crit/time-body
               (apply do-tfidf-reviews  args)))
             1000000.0)}))
   (apply merge)))

(def measures
  (->
   (tc/dataset
    (keys test-tidy-tfidf-crit))
   (tc/add-column :time (vals test-tidy-tfidf-crit))))

(tc/write-csv! measures "measures.csv")
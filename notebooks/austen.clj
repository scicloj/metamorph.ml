(ns austen
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.metamorph.ml.text :as text]
   [tablecloth.api :as tc]))

;; The Jane Austen books are available here.

(def austen-books
  (->>
   {
    "Lady Susan" "https://raw.githubusercontent.com/GITenberg/Lady-Susan_946/refs/heads/master/946.txt"
    "Sense and Snsibility" "https://raw.githubusercontent.com/GITenberg/Sense-and-Sensibility_161/refs/heads/master/161.txt"
    "Emma" "https://raw.githubusercontent.com/GITenberg/Emma_158/refs/heads/master/158.txt"
    "Northanger Abbey" "https://raw.githubusercontent.com/GITenberg/Northanger-Abbey_121/refs/heads/master/121.txt"
    "Persuasion" "https://raw.githubusercontent.com/GITenberg/Persuasion_105/refs/heads/master/105.txt"
    "Mansfield Park" "https://github.com/GITenberg/Mansfield-Park_141/raw/refs/heads/master/141.txt"
    "pride and justice" "https://github.com/GITenberg/Pride-and-Prejudice_1342/blob/master/1342.txt"
    }
   (map (fn [[title link]]
          (hash-map :title title
                    :text (slurp (io/reader link)))
          )
        )
   (tc/dataset)
   ))


(def tidy-austen
  (text/->tidy-text
   austen-books
   (fn [df] (map str (-> df :text)))
   (fn [line] [line nil])
   #(map str/lower-case (str/split % #"\W+"))
   :datatype-token-pos :int32                       
   :datatype-token-idx :int32 
   :datatype-document :int32 
   ))

(def token-table
  (tc/dataset
   {:token (-> tidy-austen :token-lookup-table keys)
    :token-idx (-> tidy-austen :token-lookup-table vals)}))

(def austen-tfidf
  (-> tidy-austen
      :datasets
      first
      text/->tfidf))

(def groups
  (-> austen-tfidf
      (tc/group-by :document { :result-type :as-seq})
      ))

(map (fn [df]
       (-> df
           (tc/order-by :tfidf :desc)
           (tc/head 10)
           (tc/left-join token-table :token-idx)
           (tc/select-columns [:token :tfidf])
           (tc/order-by :tfidf :desc)
           ))
     groups)
(ns tidytext
  (:require
   [clojure.java.io :as io]
   [clojure.string :as str]
   [scicloj.clay.v2.api :as clay]
   [scicloj.metamorph.ml.text :as text]
   [tech.v3.dataset.reductions :as ds-reductions]
   [tablecloth.api :as tc]
   [clojure.set :as set]
   [scicloj.tableplot.v1.hanami :as hanami]
   [aerial.hanami.templates :as ht]
   [scicloj.ml.smile.nlp :as nlp]
   [scicloj.kindly.v4.kind :as kind]))

(comment
  (clay/stop!)
  (clay/start!)
  (clay/make! {:source-path "notebooks/tidytext.clj"
               :show true}))


(def stopword
  (tc/dataset {:token
               (nlp/resolve-stopwords :default)}))

(defn parse-f [f]
  (let [path-components (->> (.toPath f) (.iterator) iterator-seq)
        type (str (nth path-components 2))
        label (str (nth path-components 3))
        id (str (nth path-components 4))
        texts (line-seq (io/reader f))]
    (map
     #(hash-map  :label %1
                 :text %2
                 :id %3
                 :type %4)
     (repeat (count texts) label)
     texts
     (repeat (count texts) id)
     (repeat (count texts) type))))



(def raw-text
  (tc/dataset
   (flatten
    (->>
     (file-seq (io/file "bigdata/20news-by-date"))
     (filter #(.isFile %))
     (remove #(=  ".keep" (.getName %)))
     (map parse-f)))))


(assert (pos? (tc/row-count raw-text)) "raw-text is empty")
(def raw-text-cleaned
  (->
   raw-text
   (tc/drop-rows (fn [{:keys [text]}]
                   (empty? text)))
   (tc/drop-rows (fn [{:keys [text]}] (nil? (re-matches #"^[^>]+[A-Za-z\\d]" text))))
   (tc/add-column :line (range))))


(tc/shape raw-text)

^kind/table
(tc/head
 raw-text)

(def category-counts
  (->> raw-text
       (ds-reductions/group-by-column-agg
        :label
        {:messages (ds-reductions/row-count)})))
(->
 category-counts
 (hanami/plot
  ht/bar-chart {:Y :label
                :X :messages
                :YTYPE :nominal
                })
 )


(->
 category-counts
 (hanami/plot
  hanami/bar-chart
  {:=y :label
   :=x :messages
   })
 )



(defn tokenize-fn [s]
  (if (empty? s)
    []
    (map 
     str/lower-case
     (str/split s #"\W+"))))


(def tidy-result
  (->
   (text/->tidy-text
    raw-text-cleaned
    (fn [df] (:text df))
    (fn [line] [line nil])
    tokenize-fn
    :datatype-token-pos :int32
    :datatype-document :int32
    :datatype-token-idx :int32))
  )

(def usenet-words (-> tidy-result :datasets first))

(def token-word-table
  (tc/dataset
   {:token (->> tidy-result :token-lookup-table keys)
    :token-idx (->> tidy-result :token-lookup-table vals)}))

(def stopword-table
  (-> stopword
      (tc/left-join token-word-table :token)
      (tc/drop-columns [:right.token])))


(def cleaned-usenet-words
  (-> usenet-words
      (tc/anti-join stopword-table :token-idx)))

(->
 (ds-reductions/group-by-column-agg
  :token-idx
  {:n (ds-reductions/row-count)}
  cleaned-usenet-words)
 (tc/order-by :n :desc)
 (tc/head 100)

 (tc/left-join token-word-table [:token-idx])
 (tc/drop-rows (fn [{:keys [token]}]
                 (empty? token)))

 (tc/drop-rows (fn [{:keys [token]}]
                 (some? (re-matches #"[a-z']$" token))))

 (tc/order-by :n :desc)
 (tc/head 20))


 

;(tc/shape usenet_words)
;^kind/table
;(tc/head usenet_words)



(-> raw-text-cleaned
    (tc/select-rows (fn [row] (= "sci.space" (:label row))))
    (tc/select-columns [:id :line])
    (tc/left-join cleaned-usenet-words {:left :line  :right :document})
    (tc/drop-missing)
    (tc/drop-columns [:line :document])
    (tc/rename-columns {:id :document})
    (tc/add-column :document #(map Integer/parseInt (:document %)))
    (text/->tfidf)
    (tc/order-by :tfidf :desc)
    (tc/head 10)
    (tc/left-join token-word-table :token-idx)
    (tc/order-by :tfidf :desc))





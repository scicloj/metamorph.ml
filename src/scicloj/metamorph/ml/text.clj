(ns scicloj.metamorph.ml.text
  (:require [clojure.string :as str]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.string-table :as st]
            [tech.v3.datatype :as dt])
  (:import [java.io BufferedReader FileReader]))




(defn- process-file [file-name line-func line-acc max-lines skip-lines]
  (with-open [rdr (BufferedReader. (FileReader. file-name))]
    (reduce line-func line-acc 
            (take max-lines 
                  (drop skip-lines
                        (line-seq rdr))))))


(defn- process-line [string-table text-label-split-fn acc line]
  (let [[text label](text-label-split-fn line) 
        words (str/split text #" ")
        index-count (count words)]
    (.addAllReducible string-table words)
    (let [new-acc (conj acc 
                        {:index-count index-count
                         :label label})]
      (when (zero? (rem (count new-acc) 10000))
        (println (count new-acc)))
      new-acc))
  )

(defn ->tidy-text [text-file-name text-label-split-fn 
                   & {:keys [skip-lines max-lines] 
                      :or {skip-lines  0
                           max-lines Integer/MAX_VALUE
                                                  }}

                   ]

  (let [string-table (st/string-table-from-strings [])
        index-counts-and-label
        (process-file text-file-name 
                      (partial process-line string-table text-label-split-fn)
                      [] max-lines skip-lines)

        

        
        line-idx
        (->
         (map-indexed
          (fn [idx count]
            (dt/const-reader idx count)
            )
          (map :index-count index-counts-and-label))
         (dt/concat-buffers))

        word-pos
        (flatten
         (map
          #(range (:index-count %))
          index-counts-and-label))
        
        labels 
        (flatten
         (map
          #(repeat (:index-count %) (:label %) )
          index-counts-and-label))
        ]

    {:ds
     (->
      (ds/new-dataset
       [(ds/new-column :word string-table {:datatype :string} [])
        (ds/new-column :word-index word-pos nil [])
        (ds/new-column :document line-idx nil [])
        (ds/new-column :label labels nil [])]
       )
    ;; drops empty string 
    ;;https://clojurians.zulipchat.com/#narrow/stream/236259-tech.2Eml.2Edataset.2Edev/topic/is.20empty.20string.20a.20.22missing.22.20.3F
      (tc/drop-missing) 
      (tc/drop-rows #(empty? (:word %)))
      (tc/drop-rows #(nil? (:word %)))
      (tc/drop-rows #(= "" (:word %)))
      
      )
     :st string-table}
     
    )
)

(defn ->term-frequency [tidy-text-ds]
  (-> tidy-text-ds
      (tc/group-by  [:word :document :label])
      (tc/aggregate #(hash-map :tf (ds/row-count %)))
      (tc/rename-columns {:summary-tf :tf})))


(defn add-word-idx [tidy-text-ds]
  (let [word->int-table
        (zipmap
         (-> tidy-text-ds :word .data st/int->string)
         (range))]
    (-> tidy-text-ds
        (tc/add-column
         :word-idx
         #(map word->int-table (:word %))))))









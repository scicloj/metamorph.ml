(ns gen-datasets-ns
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.java.io :as io]
   [clojure.java.shell :as sh]
   [clojure.string :as str]
   [tablecloth.api :as tc])
  
  (:import
   [java.io Writer]))

(def datasets-info
  (->
   (tc/dataset "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/refs/heads/master/datasets.csv" {:key-fn csk/->kebab-case-keyword})
   (tc/select-columns [:package :item :doc :csv])))



(defn- writeln!
  ^Writer [^Writer writer strdata & strdatas]
  (.append writer (str strdata))
  (doseq [data strdatas]
    (when data
      (.append writer (str data))))
  (.append writer "\n")
  writer)



(defn clean-R-relevant [s]
  (->
   (str/replace
    s
    #"\n\n### Examples(?s).*```"
    "")
   (str/replace
    ":::: container\n::: container\n"
    "")
   (str/replace
    #"### Usage(?s).*```"
    "")
   (str/replace
    "R Documentation"
    "Documentation")))


(defn doc-url->md [html-file md-file doc]
  (sh/sh "wget" "-O" html-file doc)
  (sh/sh "pandoc" "--from" "html" "--to" "markdown" "--no-highlight"
         html-file
         "-o" md-file)
  (sh/sh "rm" html-file)
  (let [md-text
        (if (.exists (io/file md-file))
          (clean-R-relevant
           (slurp md-file))
          "NO-DOC")]
    (sh/sh "rm" "-f" md-file)
    md-text))

(with-open [writer (io/writer "src/scicloj/metamorph/ml/datasets.clj")]

  (writeln! writer (str
                    '(ns scicloj.metamorph.ml.datasets
                       (:require [tablecloth.api :as tc]


                                 [camel-snake-kebab.core :as csk]))))

  (writeln! writer   
            "
;;    Using data documentation from 
;;    @Manual{,
;;    title = {Rdatasets: A collection of datasets originally distributed in various R packages},
;;    author = {Vincent Arel-Bundock},
;;    year = {2024},
;;    note = {R package version 1.0.0},
;;    url = {https://vincentarelbundock.github.io/Rdatasets},
;;   }
    ")

  (writeln! writer


            (str
             '(defn _fetch-dataset [csv]
                (-> csv
                    
                    (tc/dataset {:key-fn 
                                 csk/->kebab-case-keyword
                                 })
                    ))))
  (writeln! writer
            
            (str
             '(def fetch-dataset (memoize _fetch-dataset))))

  (run!
   (fn [{:keys [package item doc csv]}]

     (let [clean-item (str/replace item #"[ ()]" "")
           md-file (format "%s.md" clean-item)
           html-file (format "%s.html" clean-item)
           md-text (doc-url->md html-file md-file doc)]
       (println :package (str package "/" clean-item))

       (writeln! writer (str (list
                              'defn
                              (symbol (str  package "-"
                                            clean-item
                                            nil))
                              md-text
                              []
                              (list 'fetch-dataset csv))))))
   
   (take 10000
         (tc/rows datasets-info :as-maps))))


(when (not clojure.core/*repl*)
  (shutdown-agents))


(comment

  (println
   (-> datasets-info
       (tc/select-rows (fn [row]
                         (and
                          (= (row :package) "AER")
                          (= (row :item) "CPSSW8"))))
       :doc
       first
       (doc-url->md "a.html" "a.md"))))

(comment
  (clean-R-relevant "
 ### Usage
 
 ``` R
 data(\"CPSSW9204\")
 data(\"CPSSW9298\")
 data(\"CPSSW04\")
 data(\"CPSSW3\")
 data(\"CPSSW8\")
 data(\"CPSSWEducation\")
 ```
 
 ### Format                  

                  "))



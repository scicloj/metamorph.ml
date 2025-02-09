(ns gen-datasets-ns
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.java.io :as io]
   [clojure.pprint :as pprint]
   [clojure.string :as str]
   [tablecloth.api :as tc] ;[scicloj.metamorph.ml.rdatasets :as rdatasets]
)
  
  (:import
   [com.vladsch.flexmark.html2md.converter FlexmarkHtmlConverter]
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
    "Documentation")
   (str/replace "|--------------|-----------------|" "") 
   ))


(defn doc-url->md [doc]
  (clean-R-relevant
   (.. (FlexmarkHtmlConverter/builder) build (convert (slurp doc)))))

(def num-packages 10000)

(defn format-code [o]
  (with-out-str
    (binding [pprint/*print-right-margin* 100
              pprint/*print-miser-width* 60]
      (pprint/with-pprint-dispatch pprint/code-dispatch
        (pprint/pprint o)))))

(with-open [writer (io/writer "src/scicloj/metamorph/ml/rdatasets.clj")]

  (writeln! writer (format-code
                    '(ns scicloj.metamorph.ml.rdatasets
                       (:require [tablecloth.api :as tc]
                                 [clojure.string :as str]


                                 [camel-snake-kebab.core :as csk])

                       (:import
                        [com.vladsch.flexmark.html2md.converter FlexmarkHtmlConverter]))))

  (writeln! writer
            "
;;    Based on
;;    @Manual{,
;;    title = {Rdatasets: A collection of datasets originally distributed in various R packages},
;;    author = {Vincent Arel-Bundock},
;;    year = {2024},
;;    note = {R package version 1.0.0},
;;    url = {https://vincentarelbundock.github.io/Rdatasets},
;;   }
    ")

  (writeln! writer

            (format-code
             '(defn clean-R-relevant [s]
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
                  "Documentation")
                 (str/replace #"\|(\-*)\|(\-*)\|"  ""))))

            (format-code
             '(defn doc-url->md [doc]
                (clean-R-relevant
                 (.. (FlexmarkHtmlConverter/builder) build (convert (slurp doc))))))


            (format-code
             '(defn _fetch-dataset [csv]
                (-> csv

                    (tc/dataset {:key-fn
                                 (fn [s]
                                   (-> s
                                       (str/replace
                                        #"\." "-")
                                       csk/->kebab-case-keyword))})))))
  (writeln! writer

            (format-code
             '(def fetch-dataset (memoize _fetch-dataset))))

  (run!
   (fn [{:keys [package item doc csv]}]

     (let [clean-item (str/replace item #"[ ()]" "")
           sym (symbol (str  package "-"
                             clean-item
                             nil))]
       (println :package (str package "/" clean-item))

       (writeln! writer (format-code (list
                                      'defn sym
                                      (format "Fetch the dataset '%s' from rdatasets. 
                                               
                                               Data description: %s" 
                                              (str package "-" item)
                                              doc)
                                      {:doc-link doc}
                              ;md-text

                                      []
                                      (list 'fetch-dataset csv))))))
   (take num-packages
         (tc/rows datasets-info :as-maps)))



  (writeln! writer
            (format-code
             '(defn dataset-descriptions->doc-strings! 
                "Run this function to attach the dataset descriptions as doc string to the fetch fns"
                []
                (run!
                 (fn [v] (->
                          (symbol
                           "scicloj.metamorph.ml.rdatasets"
                           (name v))
                          (find-var)
                          (alter-meta!  (fn [var-m]
                                          (if (:doc-link var-m)
                                            (do (println :fetch (:doc-link var-m))
                                                (assoc var-m :doc (doc-url->md (:doc-link var-m))))
                                            var-m)))))



                 (keys (ns-publics (find-ns (symbol "scicloj.metamorph.ml.rdatasets"))))))))

  )





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


(comment 

  (intern *ns*
          (with-meta
            'my-fn
            {:doc (reify Object
                    (toString [this] "dynamic doc"))})
          (fn
            [a b]
            (+ a b)))
  )


(comment
  (alter-meta! #'scicloj.metamorph.ml.rdatasets/AER-Affairs
               (fn [var-m]
                 (def var-m var-m)
                 (println :var-m var-m)
                 (assoc var-m :doc (slurp (:doc-link var-m)))
                 ))
  
  )




(comment
  (defn AER-CigarettesSW
    {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CigarettesSW.html"}
    []
    (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CigarettesSW.csv"))
  
  (alter-meta! (var AER-CigarettesSW) (fn [var-m] (assoc var-m :doc (slurp (:doc-link var-m)))))

  )


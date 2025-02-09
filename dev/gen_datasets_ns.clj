(ns gen-datasets-ns
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.java.io :as io]
   [clojure.java.shell :as sh]
   [clojure.string :as str]
   [tablecloth.api :as tc]
   [scicloj.metamorph.ml.rdatasets :as rdatasets])
  
  (:import
   [java.io Writer]
   [com.vladsch.flexmark.html2md.converter FlexmarkHtmlConverter]
   ))



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

(with-open [writer (io/writer "src/scicloj/metamorph/ml/rdatasets.clj")]

  (writeln! writer (str
                    '(ns scicloj.metamorph.ml.rdatasets
                       (:require [tablecloth.api :as tc]
                                 [clojure.string :as str]


                                 [camel-snake-kebab.core :as csk])
                       
                       (:import
                        [com.vladsch.flexmark.html2md.converter FlexmarkHtmlConverter])
                       )
                    
                    
                    ))

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
                 (str/replace "|--------------|-----------------|" ""))))
            
            (str 
             '(defn doc-url->md [doc]
                (clean-R-relevant
                 (.. (FlexmarkHtmlConverter/builder) build (convert (slurp doc))))))


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
           ;md-text (doc-url->md doc)
           sym (symbol (str  package "-"
                             clean-item
                             nil))]
       (println :package (str package "/" clean-item))

       (writeln! writer (str (list
                              'defn
                              sym
                              
                              
                              
                              (format "Data description: %s" doc)
                              {:doc-link doc}
                              ;md-text
                              
                              []
                              (list 'fetch-dataset csv))))
       
       ;(writeln! writer (format "(def ^{:a 1} a 1)"))

       (writeln! writer (str (list
                              'alter-meta!
                              (list 'var sym)
                              '(fn [var-m]
                                 (assoc var-m :doc 
                                        (doc-url->md (:doc-link var-m))
                                        ;; (reify Object
                                        ;;   (toString [this]
                                        ;;     ;(slurp (:doc-link var-m))
                                        ;;     (doc-url->md (:doc-link var-m))
                                        ;;     )
                                        ;;   )
                                        ))))
  )
       ))
   
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

(require '[scicloj.metamorph.ml.rdatasets :as rdatasets] :reload)

()

(clojure.repl/doc rdatasets/datasets-airmiles)
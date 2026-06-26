(ns scicloj.metamorph.ml.r
  (:require [cemerick.pomegranate :as pom]
            [cemerick.pomegranate.aether :as aether]
            [scicloj.metamorph.ml.impl.r :as impl-r]
            [metadoc.examples :refer [example-session]]))

(defn add-clojisr-dependency
  "Adds dynamically `clojisr` to classpath using pomegranate.
   This might not work in all situations"
  [classloader-or-nil]
  (pom/add-dependencies
   :classloader classloader-or-nil
   :coordinates '[[scicloj/clojisr "1.1.0"]]
   :repositories (merge cemerick.pomegranate.aether/maven-central
                        {"clojars" "https://clojars.org/repo"})))

(defn add-opencpu-dependency
  "Adds dynamically `opencpu-clj` to classpath using pomegranate.
     This might not work in all situations"
  [classloader-or-nil]
  (pom/add-dependencies
   :classloader classloader-or-nil
   :coordinates '[[opencpu-clj/opencpu-clj "0.3.1"]]
   :repositories (merge cemerick.pomegranate.aether/maven-central
                        {"clojars" "https://clojars.org/repo"})))

(defn add-renjin-deps
  "Adds dynamically `renjin` to classpath using pomegranate.
       This might not work in all situations"
  [classloader-or-nil]
  (pom/add-dependencies
   :classloader classloader-or-nil
   :coordinates '[[org.renjin/renjin-script-engine "3.5-beta76"]]
   :repositories (merge aether/maven-central
                        {"bedatadriven-public" "https://nexus.bedatadriven.com/content/groups/public/"})))




(defn pretty
  "Compute pretty breaks using R function base::pretty.

   Parameters:
    
   - `s`         sequence of values
   - `opts`      options for pretty, 
        - `:n` number of breask is supported by all implementations
   
   - `impl`       An implementation keyword, either

       - `:ocpu`    Uses an online service https://www.opencpu.org/api.html (server: cloud.opencpu.org)
       - `:renjine` Uses https://renjin.org/   
       - `:clojisr` Uses https://github.com/scicloj/clojisr, which requires a local R installation 
    
   Each implementation requires dependencies to be added:
    
   - `:ocpu` :  [opencpu-clj/opencpu-clj \"0.3.1\"] 
   - `:renjin` : [org.renjin/renjin-script-engine \"3.5-beta76\"]
   - `:clojisr` : [scicloj/clojisr \"1.1.0\"]


   Returns seq of the breaks, which R considers 'pretty'
    
    "

  {:metadoc/examples
   [(example-session "Use wit rejin"
                     (pretty (range 0 11) {:n 7} :renjin))
    (example-session "Use wit opencpu"
                     (pretty (range 0 1 0.1) {:n 5} :ocpu))]} 
  [s opts impl]

  (let [result
        (case impl
          :ocpu (impl-r/pretty--ocpu s opts)
          :renjin (impl-r/pretty--renjine s opts)
          :clojisr (impl-r/pretty--clojisr s opts))]
    (map 
     #(if (= 0.0 (mod % 1))
        (int %)
        %
        )
     result)
    ))
 



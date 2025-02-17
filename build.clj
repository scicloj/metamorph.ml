(ns build
  (:refer-clojure :exclude [test])
  (:require
   [camel-snake-kebab.core :as csk]
   [clj-yaml.core :as yaml]
   [clojure.java.io :as io]
   [clojure.pprint :as pp]
   [clojure.tools.build.api :as b] ; for b/git-count-revs
   [org.corfield.build :as bb]
   [clojure.tools.deps :as t]
   ))

(def lib 'org.scicloj/metamorph.ml)
; alternatively, use MAJOR.MINOR.COMMITS:
;; (def version (format "6.2.%s" (b/git-count-revs nil)))
(def version "1.2beta1")
(def class-dir "target/classes")
(def basis (b/create-basis {:project "deps.edn"}))
(def jar-file (format "target/%s-%s.jar" (name lib) version))



(defn test "Run the tests." [opts]
  (-> opts
      (assoc :aliases [:runner])
      (bb/run-tests)))

(defn- pom-template [version]
  [[:description "Machine learning functions for tech.ml.dataset"]
   [:url "https://github.com/scicloj/metamorph.ml"]
   [:licenses
    [:license
     [:name "Eclipse Public License"]
     [:url "http://www.eclipse.org/legal/epl-v10.html"]]]
   [:developers
    [:developer
     [:name "Carsten Behring"]]]
   [:scm
    [:url "https://github.com/scicloj/metamorph.ml"]
    [:connection "scm:git:https://github.com/scicloj/metamorph.ml.git"]
    [:developerConnection "scm:git:https://github.com/scicloj/metamorph.ml.git"]

    [:tag (str version)]]])


(defn jar [_]
  (b/write-pom {:class-dir class-dir
                :lib lib
                :version version
                :basis basis
                :pom-data (pom-template version)
                :src-dirs ["src"]})
  (b/copy-dir {:src-dirs ["src" "resources"]
               :target-dir class-dir})
  (b/jar {:class-dir class-dir
          :jar-file jar-file}))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version
             :aliases [:runner :dev :test])
      (bb/run-tests)
      (bb/clean)
      (jar)))


(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

(defn deploy "Deploy the JAR to Clojars." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/deploy)))

(defn build-glance-columns [ops]
  (with-open [w (io/writer "resources/columms-glance.edn")]
    (-> (slurp "https://raw.githubusercontent.com/alexpghayes/modeltests/main/data-raw/columns_glance.yaml")
        (yaml/parse-string
         :key-fn #(-> % :key  csk/->kebab-case-keyword))
        (pp/pprint w))))


(defn build-tidy-columns [opts]
  (with-open [w (io/writer "resources/columms-tidy.edn")]
    (-> (slurp "https://raw.githubusercontent.com/alexpghayes/modeltests/main/data-raw/columns_tidy.yaml")
        (yaml/parse-string
         :key-fn #(-> % :key  csk/->kebab-case-keyword))
        (pp/pprint w))))

(defn build-augment-columns [ops]
  (with-open [w (io/writer "resources/columms-augment.edn")]
    (-> (slurp "https://raw.githubusercontent.com/alexpghayes/modeltests/main/data-raw/columns_augment.yaml")
        (yaml/parse-string
         :key-fn #(-> % :key  csk/->kebab-case-keyword))
        (pp/pprint w))))


(defn render-notebooks [opts]
  (let [opts (update opts :aliases conj :dev)
        aliases    [:dev]  
        basis      (b/create-basis opts) ; primarily using :aliases here
        
        alias-data (t/combine-aliases basis aliases)
        cmd-opts   (merge {:basis     basis
                           :main      'clojure.main
                           :main-args ["notebooks/render.clj"]}
                          opts
                          alias-data)
        cmd        (b/java-command cmd-opts)]
    (when-not (zero? (:exit (b/process cmd)))
      (throw (ex-info (str "run failed for " aliases) opts)))
    opts))



(ns scicloj.metamorph.explore-test
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.explore :refer [explore-all]]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml.tools :as tools]
            [hiccup2.core :as h]))

(def pinguins
  (->
   (rdatasets/palmerpenguins-penguins)
   (tc/drop-columns [:rownames])
   (tc/drop-missing [:flipper-length-mm])
   (tc/replace-missing [:sex] :value "__NA__")
   (tc/add-column :year #(map str (:year %)))))


(deftest explore-all-test
  (let [svgs
        (->
         (edn/read (java.io.PushbackReader. (io/reader "test/data/svgs.edn"))))]
    (run!
     (fn [[code expected-svg]]
       (let [my-fn (ns-resolve *ns* (symbol (first code)))
             opts (second (rest code))
             drawn-svg (pj/plot (my-fn pinguins opts))]

         (spit (io/file "expected.svg") (str (h/html expected-svg)))
         (spit (io/file "drawn.svg")  (str (h/html drawn-svg)))
         (is (= expected-svg drawn-svg) "not equals")
         ))
     svgs)))


(comment
  (run!
   (fn [code]
     (let [result
           (->
            (eval code)
            (pj/plot {:format :svg}))]
       (tools/pretty-spit "test/data/svgs.edn" (hash-map code result))))

   ['(scicloj.metamorph.ml.explore/explore-all pinguins {:color "skyblue"})
    '(scicloj.metamorph.ml.explore/explore-all pinguins {:target :species})]))



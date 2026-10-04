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
        (edn/read (java.io.PushbackReader. (io/reader "test/data/svgs.edn")))]
    (run!
     (fn [index]
       (let [code (first (nth (seq svgs) index))
             expected-svg (second (nth (seq svgs) index))
             my-fn (ns-resolve *ns* (symbol (first code)))
             opts (second (rest code))
             drawn-svg (pj/plot (my-fn pinguins opts))]

         (spit (io/file (format "/tmp/expected_%s.svg" index)) (str (h/html expected-svg)))
         (spit (io/file (format "/tmp/drawn_%s.svg" index))  (str (h/html drawn-svg)))
         (is (= expected-svg drawn-svg) "not equals")
         ))
     (range (count (seq svgs))))))


(comment
  (let [results
        (mapv
         (fn [code]
           (let [result
                 (->
                  (eval code)
                  (pj/plot {:format :svg}))]

             (hash-map code result)))

         [
          '(scicloj.metamorph.ml.explore/assocation-plot
                pinguins
                {:width-per-col 30
                 :height-per-col 40
                 :x-tick-angle 45
                 :association-text-font-size 5
                 :association-text-color "green"
                 :label-font-size 10
                 :association-visble? true})
          '(scicloj.metamorph.ml.explore/assocation-plot pinguins)
          '(scicloj.metamorph.ml.explore/explore-all pinguins {:color "skyblue"})
          '(scicloj.metamorph.ml.explore/explore-all pinguins {:target :species})])]
    (def results results)
    (tools/pretty-spit "test/data/svgs.edn" 
                       (apply merge results))          
    )
  )


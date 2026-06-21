(ns scicloj.metamorph.explore-test
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.explore :refer [explore-all]]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml.tools :as tools]
            [hiccup2.core :as h]
            )
  )



(deftest explore-all-test
  (let [pinguins
        (->
         (rdatasets/palmerpenguins-penguins)
         (tc/drop-columns [:rownames])
         (tc/drop-missing [:flipper-length-mm])
         (tc/replace-missing [:sex] :value "__NA__")
         (tc/add-column :year #(map str (:year %))))
        svgs
        (->
         (edn/read (java.io.PushbackReader. (io/reader "test/data/svgs.edn"))))]
    (is (nil?
         (some false?
               (map
                (fn [[code expected-svg]]
                  (let [fn (ns-resolve *ns* (symbol (first code)))
                        opts (first (rest code))
                        drawn-svg (pj/plot (apply fn pinguins opts))]

                    (spit (io/file "/tmp/expected.svg") (str (h/html expected-svg)))
                    (spit (io/file "/tmp/drawn.svg")  (str (h/html drawn-svg)))
                    (is (= expected-svg drawn-svg))))
                svgs))))))


(comment
  (def pinguins
    (-> 
     (rdatasets/palmerpenguins-penguins)
     (tc/drop-columns [:rownames])
     (tc/drop-missing [:flipper-length-mm])
     (tc/replace-missing [:sex] :value "__NA__")
     (tc/add-column :year #(map str (:year %)))
     ))
  (map 
   (fn [code]
     (def code code)
   (let [result (eval code)
         
         ]
     
     )
     (pr-str (hash-map code result))
     )
   
   ['(scicloj.metamorph.ml.explore/explore-all pinguins {:color "skyblue"})])

  )
  
  

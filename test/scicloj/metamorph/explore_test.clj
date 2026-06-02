(ns scicloj.metamorph.explore-test
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.explore :refer [explore-all]]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc])
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
                  (let [
                        fn (ns-resolve *ns* (symbol (first code)) )
                        opts (first (rest code))
                        drawn-svg (pj/plot (apply fn pinguins opts))]
                    (is (= expected-svg drawn-svg))
                    ))
                svgs))))))


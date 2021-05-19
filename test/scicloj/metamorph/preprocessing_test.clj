(ns scicloj.metamorph.preprocessing-test
  (:require  [clojure.test :refer [deftest is]]
             [scicloj.metamorph.ml.preprocessing :refer :all]
             [scicloj.metamorph.core :as mm]
             [tablecloth.api :refer [dataset] :as tc]
             [tech.v3.dataset.math :as std-math]
             ))

(deftest test-std-scale

  (let [data
        (dataset
         [
          [100, 0.001] ,
          [8, 0.05] ,
          [50, 0.005] ,
          [88, 0.07] ,
          [4, 0.1]]
         {:layout :as-row})

        data-2
        (dataset  {0 [60 80]
                   1 [1 2]})
        pipe
        (std-scale [0 1] {})


        fitted-ctx
        (pipe
         {:metamorph/data data
          :metamorph/mode :fit}

         )

        transformed-ctx
        (pipe
         (merge fitted-ctx
                {:metamorph/data data-2
                 :metamorph/mode :transform}
                )

         )]


    (is (=  [ 1.1305390791153365 -0.9496528264568825 0.0 0.8592097001276556 -1.0400959527861096 ]
            (get-in fitted-ctx [:metamorph/data 0])))

    (is (=  [0.22610781582306727 0.6783234474692018]
            (get-in transformed-ctx [:metamorph/data 0]))))
   )



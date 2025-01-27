(ns scicloj.metamorph.preprocessing-test
  (:require  [clojure.test :refer [deftest is]]
             [scicloj.metamorph.ml.preprocessing :refer [min-max-scale std-scale]]
             [scicloj.metamorph.core :as mm]
             [tablecloth.api :refer [dataset] :as tc]
             ))

(def data
  (dataset
   [
    [100 0.001]
    [8   0.05]
    [50  0.005]
    [88  0.07]
    [4   0.1]]
   {:layout :as-row}))

(def data-2
  (dataset  {0 [60 80]
             1 [1 2]}))




(deftest test-min-max []
  (let [pipe
        (mm/pipeline
         (min-max-scale [0] {})
         )

        fitted
        (pipe
         {:metamorph/data data
          :metamorph/mode :fit}

         )]
    (is (= [0.5
            -0.4583333333333333
            -0.020833333333333315
            0.375
            -0.5]
           (seq (get-in fitted [:metamorph/data 0]))))
    )

  )




(deftest test-std-scale

  (let [pipe
        (mm/pipeline
         (std-scale [0 1] {}))


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

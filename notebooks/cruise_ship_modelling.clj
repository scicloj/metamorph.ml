(ns cruise-ship-modelling
  (:require [camel-snake-kebab.core :as csk]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [tablecloth.api :as tc]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.metamorph.ml.explore :as explore]))



(def cruise-ships
  (tc/dataset "test/data/cruise_ship_info.csv" {:key-fn csk/->kebab-case-keyword}))


(def numeric-col-names
  (-> cruise-ships
      (tc/drop-columns [:ship-name :cruise-line])
      (tc/column-names)))

(def cruise-ships--numeric
  (-> cruise-ships
      (tc/select-columns numeric-col-names)))


;# cruiseships
(explore/pair-plot cruise-ships--numeric)
(explore/assocations-plot cruise-ships--numeric)

; # Iris
(->
 (rdatasets/datasets-iris)
 (tc/drop-columns [:species :rownames])
 explore/pair-plot)


; # wooldridge-cement
(->
 (rdatasets/wooldridge-cement)
 (tc/drop-columns [:rownames])
 (tc/drop-missing)
 (explore/assocations-plot 
  ))
 
 
; # openintro-bdims
(->
 (rdatasets/openintro-bdims)
 (tc/drop-columns [:rownames])
 (tc/drop-missing)
 explore/assocations-plot)

(-> cruise-ships
    (explore/assocations-plot))

(-> (rdatasets/datasets-iris)
    (tc/drop-columns [:rownames])
    (explore/assocations-plot))

(-> (rdatasets/datasets-iris)
    (tc/drop-columns [:rownames])
    (cf/numeric)
    (explore/correlation-plot))


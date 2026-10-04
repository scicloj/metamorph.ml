(ns explore
  (:require
   [scicloj.metamorph.ml.explore :as explore]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [scicloj.metamorph.ml.impl.dsutils :as dsutils]
   [tablecloth.api :as tc]
   [tech.v3.dataset.column-filters :as cf]
   [scicloj.plotje.api :as pj]))


(def pinguins
  
  (->
   (rdatasets/palmerpenguins-penguins)
   (tc/drop-columns [:rownames])
   (tc/drop-missing [:flipper-length-mm])
   (tc/replace-missing [:sex] :value "__NA__")
   (tc/add-column :year #(map str (:year %)))))


; # Pinguins 
; ## Explore variables

(explore/explore-all pinguins
                     {:color "blue"})

; ## Explore variables vs target  
(explore/explore-all pinguins
                     {:target :species})

; ## pair plot

(-> pinguins
    (cf/numeric)
    (explore/pair-plot 
     {:size-per-col 200}))

; ## association plot
(-> pinguins
    (explore/assocation-plot
     {:x-tick-angle 45}
     ))



(def epa2021
  
  (-> 
   (rdatasets/openintro-epa2021)
   (tc/drop-columns [:rownames :release-date])
   (tc/replace-missing)
   (dsutils/cast-cols-to-categorical-string [:no-cylinders :no-gears :model-yr])
   (dsutils/lump-categories [:carline :division :transmission-speed :mfr-code :mfr-name])
   
   
   
   ))


(tc/info epa2021 :columns)
(explore/explore-all epa2021
                     {:height 10000
                      :width 1000
                      :color "blue"})



; # pair plots

; ## Iris
(->
 (rdatasets/datasets-iris)
 (tc/drop-columns [:species :rownames])
 explore/pair-plot)


; # association plots



; ## wooldridge-cement
(->
 (rdatasets/wooldridge-cement)
 (tc/drop-columns [:rownames])
 (tc/drop-missing)
 (explore/assocation-plot 
  ))
 
 
; ## openintro-bdims
(->
 (rdatasets/openintro-bdims)
 (tc/drop-columns [:rownames])
 (tc/drop-missing)
 (explore/assocation-plot
  {:width-per-col 30
   :height-per-col 30
   :x-tick-angle 45
   :association-font-size 8
   :label-font-size 10})
 )
;## iris


(-> (rdatasets/datasets-iris)
    (tc/drop-columns [:rownames])
    (explore/assocation-plot
     {:x-tick-angle 45}))

; ## schooling

(-> (rdatasets/camerondata-schooling)
    (tc/drop-columns [:rownames])
    (tc/drop-missing)
    (explore/assocation-plot

     {:width-per-col 15
      :height-per-col 15
      :x-tick-angle 45
      :association-text-font-size 5
      :association-text-color "green"
      :label-font-size 8
      :association-visble? true
      }))

(-> pinguins
    (tc/drop-columns [:rownames])
    (tc/drop-missing)
    (explore/assocation-plot

     {:width-per-col 20
      :height-per-col 30
      :x-tick-angle 45
      :association-text-font-size 5
      :association-text-color "green"
      :label-font-size 10
      :association-visble? true}))


(ns explore
  (:require
   [scicloj.metamorph.ml.explore :as explore]
   [scicloj.metamorph.ml.impl.dsutils :as dsutils]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [scicloj.plotje.api :as pj]
   [tablecloth.api :as tc]))


(def pinguins
  
  (->
   (rdatasets/palmerpenguins-penguins)
   (tc/drop-columns [:rownames])
   (tc/drop-missing [:flipper-length-mm])
   (tc/replace-missing [:sex] :value "__NA__")
   (tc/add-column :year #(map str (:year %)))))

; # Explore variables

(pj/with-config {:theme {:bg "#ffffff" :grid "#555252" :font-size 8}}
  (explore/explore-all pinguins
                       {:color "lightblue"}))




; # Explore variables vs target  
(explore/explore-all pinguins
                     {:target :species})

(pj/with-config {:theme {:bg "#ffffff" :grid "#555252" :font-size 8}}
  (explore/explore-all pinguins
                       {:target :species})

  )


(def epa2021 
  
  (->
   (rdatasets/openintro-epa2021)
   (tc/drop-columns [:rownames :release-date])
   (tc/replace-missing)
   (dsutils/cast-cols-to-categorical-string [:no-cylinders :no-gears :model-yr])
   (dsutils/lump-categories [:carline :division :transmission-speed :mfr-code :mfr-name])))


(tc/info epa2021 :columns)
(explore/explore-all epa2021
                     {:height 10000
                      :width 1000
                      :color "blue"})




(ns explore
  (:require
   [scicloj.metamorph.ml.explore :as explore]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]))


(def pinguins
  
  (->
   (rdatasets/palmerpenguins-penguins)


   (tc/drop-columns [:rownames])
   (tc/drop-missing [:flipper-length-mm])
   (tc/replace-missing [:sex] :value "__NA__")
   (tc/add-column :year #(map str (:year %)))))

; # Explore variables

(explore/explore-all pinguins
                     {:color "blue"})


; # Explore variables vs target  
(explore/explore-all pinguins
                     {:target :species})

(ns scicloj.metamorph.explore
  (:require [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [tablecloth.api :as tc]
            [fastmath.stats :as stats]

            [scicloj.plotje.api :as pj]
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset :as dataset]))





(defn round-to-precision
  "Round a double to the given precision (number of significant digits)"
  [precision d]
  (let [factor (Math/pow 10 precision)]
    (/ (Math/round (* d factor)) factor)))


(defn explore-categorical-var [data variable]
  (let [freqs (-> data variable frequencies)
        n-missing (-> data variable tc/select-missing tc/row-count)
        n-unique (count freqs)

        plot-data
        (tc/dataset
         (map
          (fn [[category count]]
            ;(print :category category)
            {variable (or category "_NA_")
             :% (round-to-precision 2 (float (/ count (tc/row-count data))))})
          freqs))]
    (->
     plot-data
     (pj/pose variable :%)
     (pj/lay-value-bar  {:color "skyblue" :alpha 0.7})
     (pj/lay-label  {:text :% :color "black"})
     (pj/options {:title (name variable)
                  :subtitle (format "na = %d, unique = %d" n-missing n-unique)
                  :x-label ""
                  })
     (pj/coord :flip)
     ;
     )))



(defn explore-continous-var [data variable]
  (let [
        qq-2 (stats/quantile (get data variable) 0.02)
        qq-98 (stats/quantile (get data variable) 0.98)
        n-missing (-> data variable tc/select-missing tc/row-count)
        min (apply  tcc/min (get data variable))
        max (apply  tcc/max (get data variable))
        ]
    (-> data
        (tc/select-rows (fn [row]
                          (and
                           (>= (get row variable) qq-2)
                           (<= (get row variable) qq-98))))
        (pj/pose variable)
        (pj/lay-density)
        (pj/options {:title (name variable)
                     :subtitle (format "na = %d, min = %.2f, max = %.2f" n-missing (double min) (double max))
                     :x-label ""
                     }
                    
                    )
        )))

(defn explore-all [data]
  (pj/arrange
   (->>
    (map
     (fn [col]
       (if (:categorical? (meta col))
         (explore-categorical-var data (:name (meta col)))
         (explore-continous-var data (:name (meta col)))))
     (->
      (tc/columns data)))
    (partition-all 2))
   {:height 1000
    :width 800}))





(-> 
    (rdatasets/palmerpenguins-penguins)
    
    (tc/drop-columns [:rownames])
    (tc/drop-missing [:flipper-length-mm])
    (tc/add-column :year #(map str (:year %)))
    (explore-all)
    )  



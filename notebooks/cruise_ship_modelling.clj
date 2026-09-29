(ns cruise-ship-modelling
  (:require [camel-snake-kebab.core :as csk]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [fastmath.stats :as stats]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.metamorph.ml.explore :as explore]
            [tablecloth.api.utils :as tc-utils]
            
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset :as ds]))






(def cruise-ships
  (tc/dataset "test/data/cruise_ship_info.csv" {:key-fn csk/->kebab-case-keyword}))

cruise-ships

(tc/info cruise-ships)

(def numeric-col-names
  (-> cruise-ships
      (tc/drop-columns [:ship-name :cruise-line])
      (tc/column-names)))

(def cruise-ships--numeric
  (-> cruise-ships
      (tc/select-columns numeric-col-names)))


;# cruiseships
(explore/pair-plot cruise-ships--numeric)
(explore/correlation-plot cruise-ships--numeric)

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
 (explore/correlation-plot 
  ;{:measure-fn (fn [s1 s1] 1.0)}
  )
 
 ;(cf/numeric)
 ;explore/correlation-plot
 )

; # openintro-bdims
(->
 (rdatasets/openintro-bdims)
 (tc/drop-columns [:rownames])
 (tc/drop-missing)
 (cf/numeric)
 explore/correlation-plot)

(def columns
  (-> 
   (rdatasets/datasets-iris)
   
   ))

(-> columns first meta)


(defn my-stat [categories measurements]
  (let [{:keys [SSt SSe]} (stats/one-way-anova-test
                           [measurements
                            categories
                            ])]
    (/ SSt (+ SSt SSe))))

(defn correlation-ratio [categories measurements]
  (->>
   (mapv vector
         measurements
         categories)
   (group-by second)
   (mapv
    (fn [[k v]]
      [k (mapv first v)]))
   (mapv
    second)
   stats/anova-eta-sq
   tcc/sqrt))


(defn associations [ds]

  (->>
   (let [columns (tc/columns ds)]
     (for [c-1 columns c-2 columns]
       (let [meta-c-1 (meta c-1)
             meta-c-2 (meta c-2)
             assoc (cond
                     (and
                      (contains? (tc-utils/->general-types (tcc/typeof c-1)) :numerical)
                      (contains? (tc-utils/->general-types (tcc/typeof c-2)) :numerical))
                     {:value (stats/pearson-correlation c-1 c-2)
                      :method :pearson-correlation}

                     (and
                      (contains? (tc-utils/->general-types (tcc/typeof c-1)) :textual)
                      (contains? (tc-utils/->general-types (tcc/typeof c-2)) :textual))
                     {:value (stats/cramers-v c-1 c-2)
                      :method :cramers-v}

                     (and
                      (contains? (tc-utils/->general-types (tcc/typeof c-1)) :textual)
                      (contains? (tc-utils/->general-types (tcc/typeof c-2)) :numerical))
                     {:value (correlation-ratio c-1 c-2)
                      :method :coorelation-ratio}

                     (and
                      (contains? (tc-utils/->general-types (tcc/typeof c-1)) :numerical)
                      (contains? (tc-utils/->general-types (tcc/typeof c-2)) :textual))
                     {:value (correlation-ratio c-2 c-1)
                      :method :coorelation-ratio}



                     )]
         {:c-1-name (:name meta-c-1)
          :c-2-name (:name meta-c-2)
          :assoc assoc})))
   (remove #(nil? (:assoc %)))
   ))


(def ds
  (-> (rdatasets/datasets-iris)
      (tc/drop-columns  [:rownames]) 
      ;(ds/categorical->number [:species])
      )
  )


(def assocs
  (->>  ds
        
        associations
        (map #(hash-map :assoc-str (->> % :assoc :value (format "%.2f"))
                        :assoc (->> % :assoc :value )
                        :x (-> % :c-1-name)
                        :y (-> % :c-2-name)))
        tc/dataset
        )  
  
  )

(def columns
  (-> ds tc/column-names reverse))

(def column-indexes (range (count columns)))
(def index-col-name-map (zipmap column-indexes columns))
(def num-cols (count column-indexes))

(def x
  (map vector
       (map index-col-name-map (range num-cols))
       (range num-cols)))
(def sorted
  (sort-by (fn [[id _]]
             (.indexOf (tc/column-names ds) id))
           x))

(def tick-labels (map first sorted))
(def breaks (map second sorted))

(->  assocs
     (tc/add-columns {:x-indexed (map
                                  (clojure.set/map-invert index-col-name-map)
                                  (:x assocs))
                      :y-indexed (map
                                  (clojure.set/map-invert index-col-name-map)
                                  (:y assocs))})
     
     (pj/lay-tile :x-indexed :y-indexed {;:text :assoc-str 
                                         :fill :assoc
                                         
                                         })
     (pj/lay-text :x-indexed :y-indexed {:text :assoc-str
                                         :align-x :center
                                         :align-y :center
                                        :color "white" 
                                         })
     (pj/scale :x {
                   :tick-labels tick-labels
                   :breaks breaks
                   :domain [num-cols -1]
                   })
     (pj/scale :y {:breaks breaks
                   :tick-labels tick-labels
                   
                   })
     (pj/scale :fill {:range :grDevices/Berlin
                      :domain [-1 1]})
     (pj/options {
                  :x-tick-angle 45
                  })

     )




(tcc/sqrt (stats/anova-eta-sq [[0.7] [0.2 0.5] [0.3]]))

(defn by
  ([data f]
   (mapv second (sort-by first (group-by f (data)))))
  ([data f selector]
   (vec (for [group (by data f)]
          (map selector group)))))





(-> (rdatasets/datasets-iris)
    (pj/lay-tile :sepal-length :sepal-width)
    (pj/scale :x {:domain [4.5 8]})
    )
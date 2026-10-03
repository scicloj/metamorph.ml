(ns scicloj.metamorph.ml.explore
  (:require [clojure.set :as c-set]
            [fastmath.stats :as stats]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [tablecloth.api.utils :as tc-utils]
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset.column :as ds-col]))


(defn- round-to-precision
  "Round a double to the given precision (number of significant digits)"
  [precision d]
  (let [factor (Math/pow 10 precision)]
    (/ (Math/round (* d factor)) factor)))


(defn- explore-categorical-var [data variable {:keys [color target]}]

  (let [group (if (some? target) target color)
        freqs (-> data variable frequencies)
        n-missing (-> data variable tc/select-missing tc/row-count)
        n-unique (count freqs)
        subtitle (if (some? target)
                   ""
                   (format "na = %d, unique = %d" n-missing n-unique))


        plot-data
        (-> data
            (tc/group-by [target variable])
            (tc/aggregate (fn [ds]
                            
                            (round-to-precision 2
                                                (* 100.0
                                                   (/
                                                    (tc/row-count ds)
                                                    (if (some? target)
                                                      (get freqs (first (get ds variable)))
                                                      (tc/row-count data))
                                                    
                                                    ))))
                          {:default-column-name-prefix :%}))]
    (->
     plot-data
     (pj/lay-bar variable :% {:color group
                                    :alpha 0.7})
     ((fn [pose]

        (if (some? target)
          (pj/lay-text pose   {:text :% 
                               :group group 
                               :color "black"
                               :align-x :right 
                               })
          (pj/lay-text pose variable :% {:text :%
                                         :color "black"
                                         :align-y :center
                                         :align-x :right}))))
     (pj/options {:title (str variable)
                  :subtitle subtitle
                  :x-label ""})
     (pj/coord :flip))))



(defn- explore-continous-var [data variable {:keys [color target]}]
  (let [group (if (some? target) target color)
        qq-2 (stats/quantile (get data variable) 0.02)
        qq-98 (stats/quantile (get data variable) 0.98)
        n-missing (-> data variable tc/select-missing tc/row-count)
        min (tcc/reduce-min (get data variable))
        max (tcc/reduce-max  (get data variable))
        mean (tcc/mean  (get data variable))
        subtitle (if (some? target)
                   ""
                   (format "na = %d, min = %.2f, max = %.2f" n-missing (double min) (double max)))
        rule-fn (if (some? target)
                  (fn [pose] pose)
                  (fn [pose] (pj/lay-rule-v pose {:x-intercept mean :color "grey" :alpha 0.5})))
        rug-fn (if (some? target)
                 (fn [pose] pose)
                 (fn [pose] (pj/lay-rug pose)))
        ]
    (-> data
        ;; not doing anything
        ;; (tc/select-rows (fn [row]
        ;;                   (and
        ;;                    (>= (get row variable) qq-2)
        ;;                    (<= (get row variable) qq-98))))

        (pj/lay-density variable  {:color group})

        (rug-fn)
        (rule-fn)
        ;TODO: https://github.com/scicloj/plotje/issues/23

        ;(pj/scale :x {:domain [qq-2 qq-98]})
        (pj/options {:title (str variable)
                     :subtitle subtitle
                     :x-label ""}))))

(defn explore-all
  "Create a faceted overview plot for all columns in a Tablecloth dataset.

  For each non-target column:
  - Categorical columns (meta :categorical? true) -> percentage bar charts.
    If :target is provided, bars are grouped/colored by target and percentages
    are computed per category; otherwise a single-color bar chart with value
    labels, plus subtitle showing NA count and number of unique values.
  - Continuous columns -> density plots.
    If :target is provided, densities are colored by target; otherwise a rug
    and a vertical rule at the mean are added, plus a subtitle with NA/min/max.

  Options (merged with defaults):
  - :target  keyword or nil (default nil) — column name used for grouping/coloring.
  - :color   string/color when no target is given (default \"skyblue\").
  - :height  int canvas height (default 1000).
  - :width   int canvas width (default 800).

  Returns a plotje pose with the individual plots arranged in a 2-column grid."
  [data opts]
  (let  [defaults {:height 1000
                   :width 800
                   :color "skyblue" :target nil}
         defaulted-opts (merge defaults opts)]

    (pj/arrange
     (->>
      (map
       (fn [col]
         (when (not (= (ds-col/column-name col) (:target opts)))
           (if (:categorical? (meta col))
             (explore-categorical-var data (:name (meta col)) defaulted-opts)
             (explore-continous-var data (:name (meta col)) defaulted-opts))))
       (->
        (tc/columns data)))
      (remove nil?)
      (partition-all 2))
     (select-keys defaulted-opts [:height
                                  :width]))))



(defn pair-plot
  "Create a pair plot with one panel for each pair of dataset columns.

  Options:
  - `:size-per-col` — panel size in pixels, used for both plot width and
    height (default 100).

  Returns a Plotje pose."
  [dataset & {:keys [size-per-col]
              :or {size-per-col 100}}]
  (let [col-names (tc/column-names dataset)
        num-cols (count col-names)]
    (-> dataset
        (pj/pose
         (pj/cross col-names col-names))
        (pj/options {:width (* size-per-col num-cols)
                     :height (* size-per-col num-cols)}))))



(defn- correlation-ratio [categories measurements]
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


(defn- calc-associations [ds]

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
                      :method :coorelation-ratio})]
         {:c-1-name (:name meta-c-1)
          :c-2-name (:name meta-c-2)
          :assoc assoc})))
   (remove #(nil? (:assoc %)))))


(defn assocation-plot
  "Create a heatmap of pairwise associations between dataset columns.

  Numeric-numeric pairs use Pearson correlation, textual-textual pairs use
  Cramer's V, and numeric-textual pairs use the correlation ratio. The
  association value is shown in each tile when `:association-visible?` is
  true.
   
   Expects a tech.ml dataset without missing values.

  Options:
  - `:width-per-col` — tile width in pixels (default 60).
  - `:height-per-col` — tile height in pixels (default 60).
  - `:x-tick-angle` — angle for x-axis tick labels (default 0).
  - `:association-text-font-size` — value-label font size (default 11).
  - `:label-font-size` — axis-label font size (default 11).
  - `:association-visible?` — whether to show values in tiles (default true).
  - `:association-text-color` — value-label color (default \"black\").
  - `:gradient` — fill color scale (default `:grDevices/Blue-Red`).

  Returns a Plotje pose."
  [ds & {:keys [width-per-col
                                     height-per-col
                                     x-tick-angle
                                     association-text-font-size
                                     label-font-size
                                     association-visible?
                                     association-text-color
                                     gradient
                                     ]
                              :or {width-per-col 60
                                   height-per-col 60
                                   x-tick-angle 0
                                   association-text-font-size 11
                                   label-font-size 11
                                   association-visible? true
                                   gradient :grDevices/Blue-Red
                                   association-text-color "black"}}]

  (let [columns
        (-> ds tc/column-names reverse)

        column-indexes (range (count columns))
        index-col-name-map (zipmap column-indexes columns)
        num-cols (count column-indexes)

        column-index-map
        (map vector
             (map index-col-name-map (range num-cols))
             (range num-cols))

        sorted
        (sort-by (fn [[id _]]
                   (.indexOf (tc/column-names ds) id))
                 column-index-map)

        tick-labels (map first sorted)
        breaks (map second sorted)


        assocs
        (->>  ds

              calc-associations
              (map #(hash-map :assoc-str (->> % :assoc :value (format "%.2f"))
                              :assoc (->> % :assoc :value)
                              :x (-> % :c-1-name)
                              :y (-> % :c-2-name)))
              
              tc/dataset)
        
        association-lay-fn (if association-visible?
                             (fn [pose]
                               (pj/lay-text pose
                                            :x-indexed :y-indexed
                                            {:text :assoc-str
                                             :align-x :center
                                             :align-y :center
                                             :color association-text-color
                                             :font-size association-text-font-size}))
                             (fn [pose] pose))
        ]
    
    (->  assocs
         (tc/add-columns {:x-indexed (map
                                      (c-set/map-invert index-col-name-map)
                                      (:x assocs)) 
                          :y-indexed (map
                                      (c-set/map-invert index-col-name-map)
                                      (:y assocs))})

         (pj/lay-tile :x-indexed :y-indexed {;:text :assoc-str 
                                             :fill :assoc})
         association-lay-fn
         
         (pj/scale :x {:tick-labels (reverse tick-labels)
                       :breaks (reverse breaks)
                       :domain  [num-cols -1]})
         (pj/scale :y {:breaks breaks
                       :tick-labels tick-labels
                       :domain [num-cols -1]
                       })
         (pj/scale :fill {:range gradient
                          :domain [-1 1]})
         (pj/options {:width (+ 200 (* width-per-col num-cols))
                      :height (* height-per-col num-cols)
                      :x-label ""
                      :y-label ""
                      :theme {:font-size label-font-size}
                      

                      :x-tick-angle x-tick-angle
                      :color-label "Association"}))))

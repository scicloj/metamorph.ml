(ns scicloj.metamorph.ml.explore
  (:require [fastmath.stats :as stats]
            [scicloj.plotje.api :as pj]
            [tablecloth.api :as tc]
            [tablecloth.column.api :as tcc]
            [tech.v3.dataset.column :as ds-col]))





(defn- round-to-precision
  "Round a double to the given precision (number of significant digits)"
  [precision d]
  (let [factor (Math/pow 10 precision)]
    (/ (Math/round (* d factor)) factor)))


(defn- explore-categorical-var [data variable {:keys [color target]}]
  ;; (def data data)
  ;; (def variable variable)
  ;; (def color color)
  ;; (def target target)

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
    ;; (def plot-data plot-data)
    ;; (def group group)
    ;; (def variable variable)
    ;; (def variable variable)
    ;; (def n-missing n-missing)
    ;; (def n-unique n-unique)
    ;; (def color color)
    (->
     plot-data
     (pj/lay-value-bar variable :% {:color group
                                    :alpha 0.7})
     ((fn [pose]

        (if (some? target)
          pose
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
    ;; (def data data)
    ;; (def qq-2 qq-2)
    ;; (def qq-98  qq-98)
    ;; (def group group)
    ;; (def variable variable)
    ;; (def subtitle subtitle)
    ;; (def rule-fn rule-fn)
    (-> data
        ;; not doing anything
        ;; (tc/select-rows (fn [row]
        ;;                   (and
        ;;                    (>= (get row variable) qq-2)
        ;;                    (<= (get row variable) qq-98))))
    
        (pj/lay-density variable  {:color group}  )

        (rug-fn
         )
        (rule-fn)
        ;TODO: https://github.com/scicloj/plotje/issues/23

        ;(pj/scale :x {:domain [qq-2 qq-98]})
        (pj/options {:title (str variable)
                     :subtitle subtitle
                     :x-label ""}))))

(defn explore-all [data opts]
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




(ns scicloj.metamorph.ml.explore
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


(defn explore-categorical-var [data variable {:keys [color target]}]
  (let [;group (if (some? target) target color)
        freqs (-> data variable frequencies)
        n-missing (-> data variable tc/select-missing tc/row-count)
        n-unique (count freqs)

        plot-data
        (-> data
            (tc/group-by [target variable])
            (tc/aggregate (fn [ds]
                            (round-to-precision 2
                                                (* 100.0
                                                   (/
                                                    (tc/row-count ds)
                                                    (tc/row-count data)))))
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
     (pj/lay-value-bar variable :% {
                                    :color color
                                    :alpha 0.7})
      (pj/lay-text variable :% {:text :%
                                 :color "black"
                                 :align-y :bottom
                                 :align-x :right
                                 })
     (pj/options {:title (name variable)
                  :subtitle (format "na = %d, unique = %d" n-missing n-unique)
                  :x-label ""})
     (pj/coord :flip)
     )))



(defn explore-continous-var [data variable {:keys [color target]}]
  (let [
        ;group (if (some? target) target color)
        qq-2 (stats/quantile (get data variable) 0.02)
        qq-98 (stats/quantile (get data variable) 0.98)
        n-missing (-> data variable tc/select-missing tc/row-count)
        min (tcc/reduce-min (get data variable))
        max (tcc/reduce-max  (get data variable))
        mean (tcc/mean  (get data variable))]
    (-> data
        (tc/select-rows (fn [row]
                          (and
                           (>= (get row variable) qq-2)
                           (<= (get row variable) qq-98))))
        (pj/pose variable)
        (pj/lay-density  {:color color}  )
        (pj/lay-rule-v {:x-intercept mean :color "grey" :alpha 0.5})
        (pj/options {:title (name variable)
                     :subtitle (format "na = %d, min = %.2f, max = %.2f" n-missing (double min) (double max))
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
         (if (:categorical? (meta col))
           (explore-categorical-var data (:name (meta col)) defaulted-opts)
           (explore-continous-var data (:name (meta col)) defaulted-opts)))
       (->
        (tc/columns data)))
      (partition-all 2))
     (select-keys defaulted-opts [:height
                                  :width]))))




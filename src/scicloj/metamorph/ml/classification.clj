(ns scicloj.metamorph.ml.classification
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype.pprint :as dtype-pp]
            [scicloj.metamorph.ml :as ml]))
            
            
(defn- safe-inc
    [item]
    (if item
      (inc item)
      1))

(defn confusion-map
  "Creates a confusion-matrix in map form. Can be either as raw counts or normalized.
  `normalized` when :all (default) it is normalized
                    :none otherwise
   "
  ([predicted-labels labels normalize]

   (let [answer-counts (frequencies labels)]
     (->> (map vector predicted-labels labels)
          (reduce (fn [total-map [pred actual]]
                    (update-in total-map [actual pred]
                               safe-inc))
                  {})
          (map (fn [[k v]]
                 [k (->> v
                         (map (fn [[guess v]]
                                [guess
                                 (case normalize
                                   :all  (double (/ v (get answer-counts k)))
                                   :none v)]))
                                   

                                 
                         (into (sorted-map)))]))
          (into (sorted-map)))))
  ([predicted-labels labels]
   (confusion-map predicted-labels labels :all)))
  


(defn confusion-map->ds
  "Converts teh confusion-matrix map obtained via `confusion-mao` into a dataset representation"
  [conf-matrix-map]
  (let [
        conf-matrix-map conf-matrix-map
        all-counts (flatten (map vals (vals conf-matrix-map)))
        _ (assert (or
                   (every? float? all-counts)
                   (every? int? all-counts))
                  (str "All counts need to be either int? or float?, but are: " all-counts))
        is-integer (integer? (first all-counts))
        all-labels (->> (keys conf-matrix-map)
                        sort)
        column-names (concat [:column-name]
                             all-labels)]
    (->> all-labels
         (map (fn [label-name]
                (let [entry (get conf-matrix-map label-name)]
                  (merge {:column-name label-name}
                         (->> all-labels
                              (map (fn [entry-name]
                                     [entry-name (dtype-pp/format-object
                                                  (get entry entry-name
                                                       (if is-integer
                                                         0
                                                         0.0)))]))
                              (into {}))))))
         (ds/->>dataset)
         (#(ds/select-columns % column-names)))))
  

(defn- get-majority-class [target-ds]
  (let [target-column-name (first
                            (ds-mod/inference-target-column-names target-ds))
        target-column (get target-ds target-column-name)
        freqs (frequencies target-column)
        ]
    ;; (println  :target-column target-column)
    ;; (println  :target-column--meta (meta target-column))
    ;; (println  :target-column--freq  freqs)
    (->>
     freqs
     (sort-by second)
     reverse
     first
     first)))

(ml/define-model! :metamorph.ml/dummy-classifier
  (fn [feature-ds target-ds options]
    (let [target-column-name (first
                              (ds-mod/inference-target-column-names target-ds))]
      {:majority-class (get-majority-class target-ds)
       :distinct-labels (-> target-ds (get target-column-name) distinct)}))

  (fn [feature-ds thawed-model {:keys [options model-data target-categorical-maps] :as model}]
    (let  [ target-column-name (-> model :target-columns first)
           dummy-labels
           (take (ds/row-count feature-ds)
                 (case (get options :dummy-strategy :majority-class)
                   :majority-class (repeat (:majority-class model-data))
                   :fixed-class (repeat (:fixed-class options))
                   :random-class (repeatedly
                                  (fn [] (rand-nth (:distinct-labels model-data))))))]

      (ds/new-dataset [(ds/new-column target-column-name dummy-labels {:column-type :prediction
                                                                       :categorical-map (get target-categorical-maps target-column-name)})])))
     
  {:glance-fn (fn [_] (ds/->dataset {:npar 0}))
   :options
   [:map {:closed true}
    [:dummy-strategy
     {:optional true
      :description "The strategy to use for the dummy classifier"}
     [:enum :majority-class :fixed-class :random-class]]
    [:fixed-class {:optional true
                   :description "The fixed class to use when strategy is :fixed-class"}
     number?]]})



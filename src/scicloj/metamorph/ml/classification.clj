(ns scicloj.metamorph.ml.classification
  (:require [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.modelling :as ds-cat]
            [tech.v3.datatype.pprint :as dtype-pp]
            [scicloj.metamorph.ml :as ml]))
            
            
(defn- safe-inc
    [item]
    (if item
      (inc item)
      1))

(defn confusion-map
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
  ([conf-matrix-map normalize]
   (let [all-labels (->> (keys conf-matrix-map)
                         sort)
         header-column (merge {:column-name "column-name"}
                              (->> all-labels
                                   (map #(vector % %))
                                   (into {})))
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
                                                        (case normalize
                                                          :none 0
                                                          :all 0.0)))]))
                               (into {}))))))
          (concat [header-column])
          (ds/->>dataset)
          ;;Ensure order is consistent
          (#(ds/select-columns % column-names)))))
  ([conf-matrix-map]
   (confusion-map->ds conf-matrix-map :all)))





#_(defn confusion-ds
    [model test-ds]
    (let [predictions (ml/predict model test-ds)
          answers (ds/labels test-ds)]
      (-> (probability-distributions->labels predictions)
          (confusion-map (ds/labels test-ds))
          (confusion-map->ds))))
(comment
  (confusion-map [:a :b :c :a] [:a :c :c :a] :all))
  

(defn- get-majority-class [target-ds]
  (let [target-column-name (first
                            (ds-mod/inference-target-column-names target-ds))]
    (->>
     (-> target-ds (get target-column-name) frequencies)
     (sort-by :second)
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
     
  {})

(ns scicloj.metamorph.ml.ensemble
  (:require
   [scicloj.metamorph.core :as morph]
   [scicloj.metamorph.ml :as ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.column-filters :as cf]))

(defn majority [l]
  (->>
   (frequencies l)
   seq
   (sort-by second)
   reverse
   first
   first))


(defn  ensemble-pipe [pipes]
  "Creates an ensemble pipeline function out of various pipelines. The different predictions
   get combined via majority voting.
   Can be used in the same way as any other pipeline.
"
  (def pipes pipes)
  (morph/pipeline
   {:metamorph/id :ensemble}
   (fn [{:metamorph/keys [id data mode] :as ctx}]
     (let [
           pipe-keys (map-indexed (fn [index _] (keyword (str "pipe-" index))) pipes)]

       (case mode
         :fit
         (let [fitted-ctxs
               (apply merge
                      (map
                       (fn [pipe-key pipe]
                         (hash-map pipe-key
                                   (morph/fit-pipe (:metamorph/data ctx) pipe)))

                       pipe-keys
                       pipes))]
               

           (assoc ctx id {
                          ;; :target-column (-> fitted-ctxs :pipe-0 :model :target-columns first)
                          :fitted-ctxs fitted-ctxs}))
         :transform
         (let [

               _ (def ctx ctx)
               _ (def id id)


               target-column (-> ctx (get id) :fitted-ctxs :pipe-0 :model :target-columns first)
               target-categorical-map  (-> ctx (get id) :fitted-ctxs :pipe-0 :model :target-categorical-maps)

               _ (def target-column target-column)
               _ (def target-categorical-map target-categorical-map)
               
               transformed-ctxs
               (map
                (fn [pipe-key pipe] (morph/transform-pipe data pipe (-> ctx (get id) :fitted-ctxs pipe-key)))
                pipe-keys
                pipes)
               _ (def transformed-ctxs transformed-ctxs)


               predictions
               (map
                #(cf/prediction (get % :metamorph/data))
                transformed-ctxs)
               _ (def predictions predictions)


               columns
               (map-indexed
                (fn [index prediction]
                  (ds/new-column (keyword (str "model-" index)) (get prediction target-column)))
                predictions)
               _ (def columns columns)


               target-ds (-> transformed-ctxs first :model :scicloj.metamorph.ml/target-ds)
               _ (def target-ds target-ds)



               prediction-ds (-> (ds/new-dataset columns)

                                 (tc/add-column target-column
                                                (fn [ds]
                                                  (->> ds
                                                       tc/rows
                                                       (map majority))))

                                 (ds/assoc-metadata [target-column]
                                                    :column-type :prediction
                                        ;:categorical? (target-ds-meta :categorical?)
                                                    :categorical-map (get target-categorical-map target-column)))]


           ;; (ds-cat/fit-categorical-map)
           (def prediction-ds prediction-ds)
           (assoc ctx
                  ;; :scicloj.metamorph.ml/feature-ds (-> transformed-ctxs first :model :scicloj.metamorph.ml/feature-ds)
                  :model {:scicloj.metamorph.ml/target-ds target-ds}

                  :metamorph/data prediction-ds
                  id
                  (->>
                   (map
                    (fn [pipe-key ctx]
                      (hash-map pipe-key ctx))
                    pipe-keys
                    transformed-ctxs)
                   (apply merge)))))))))

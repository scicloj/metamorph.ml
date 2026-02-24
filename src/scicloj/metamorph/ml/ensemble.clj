(ns scicloj.metamorph.ml.ensemble
  (:require
   [scicloj.metamorph.core :as morph]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]))

(defn- majority [l]
  (->>
   (frequencies l)
   seq
   (sort-by second)
   reverse
   first
   first))


(defn  ensemble-pipe
  "Creates an ensemble pipeline from multiple pipelines using majority voting.

  `pipes` - Sequence of metamorph pipeline functions

  Returns a single metamorph pipeline function that trains all sub-pipelines
  in :fit mode and combines their predictions via majority voting in :transform
  mode. Each pipeline is trained independently on the same data.

  In :fit mode, stores all fitted pipeline contexts. In :transform mode, runs
  predictions from all pipelines and selects the most common prediction for
  each observation.

  The ensemble pipeline can be used anywhere a regular pipeline is accepted
  (e.g., in `evaluate-pipelines`).

  metamorph                            | .
  -------------------------------------|----------------------------------------------------------------------------
  Behaviour in mode :fit               | Fits all sub-pipelines and stores their contexts
  Behaviour in mode :transform         | Runs all sub-pipelines and combines predictions by majority vote
  Reads keys from ctx                  | In `:transform`: reads fitted sub-pipeline contexts
  Writes keys to ctx                   | In `:fit`: stores all fitted contexts; In `:transform`: writes final prediction

  See also: `scicloj.metamorph.ml/evaluate-pipelines`"
  [pipes]
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
                          :fitted-ctxs fitted-ctxs}))
         :transform
         (let [
               target-column (-> ctx (get id) :fitted-ctxs :pipe-0 :model :target-columns first)
               target-categorical-map  (-> ctx (get id) :fitted-ctxs :pipe-0 :model :target-categorical-maps)


               transformed-ctxs
               (map
                (fn [pipe-key pipe] (morph/transform-pipe data pipe (-> ctx (get id) :fitted-ctxs pipe-key)))
                pipe-keys
                pipes)


               predictions
               (map
                #(cf/prediction (get % :metamorph/data))
                transformed-ctxs)


               columns
               (map-indexed
                (fn [index prediction]
                  (ds/new-column (keyword (str "model-" index)) (get prediction target-column)))
                predictions)


               target-ds (-> transformed-ctxs first :model :scicloj.metamorph.ml/target-ds)

               prediction-ds (-> (ds/new-dataset columns)

                                 (tc/add-column target-column
                                                (fn [ds]
                                                  (->> ds
                                                       tc/rows
                                                       (map majority))))

                                 (ds/assoc-metadata [target-column]
                                                    :column-type :prediction
                                                    :categorical-map (get target-categorical-map target-column)))]


           (assoc ctx
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

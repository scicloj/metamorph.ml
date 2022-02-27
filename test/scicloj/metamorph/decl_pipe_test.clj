(ns scicloj.metamorph.decl-pipe-test
  (:require  [clojure.test :refer [deftest is] :as t]
             [scicloj.metamorph.core :as mm]
             [scicloj.metamorph.ml :as ml]
             [scicloj.metamorph.ml.loss :as loss]
             [tech.v3.dataset.metamorph :as ds-mm]
             [tablecloth.api :as tc]
             [tech.v3.dataset :as ds]))

(defonce data
  (tc/dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))



(defn is-thrown [decl-pipe]

  (is (thrown? IllegalArgumentException
               (->
                (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
                (nth 0) (nth 0) :train-transform :metric))))





(defn eval-pipe [decl-pipe]
  (def decl-pipe decl-pipe)

  (->
   (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
   first first :train-transform :metric))

(defn is-pos-metric [decl-pipe]
  (def decl-pipe decl-pipe)
  (t/is (pos?
         (->
          (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
          first first :train-transform :metric))))
         

(defn identity-1 []
  (fn [ctx] ctx))


(defn identity-2 []
  (fn [ctx]
    ( assoc ctx :metamorph/data
     (tech.v3.dataset/update-column (:metamorph/data ctx) :species identity))))

(defn identity-3 []
  (fn [ctx]
    ( update ctx :metamorph/data
     #(tech.v3.dataset/update-column % :species identity))))



(defn update-species [f]

  (fn [ctx]
    (update ctx :metamorph/data
            tech.v3.dataset/update-column :species f)))


(defn duplicate-columns
  [column-selector]
  (mm/lift
   (fn [ds]
     (let [column-names (tc/column-names ds column-selector)]
       (reduce (fn [d n]
                 (tc/add-column d (str n "-copy") (d n))) ds column-names)))))


(defn upper-case-col [col]
  (map clojure.string/upper-case col))

(ml/define-model! :test-model
  (fn train
    [feature-ds label-ds options]
    {})
  (fn predict
    [feature-ds thawed-model {:keys [target-columns
                                     target-categorical-maps
                                     top-k
                                     options]}]

    (ds/new-dataset [(ds/new-column :species
                                    (repeat (tc/row-count feature-ds) 1)
                                    {:column-type :prediction})]))
  {})






(deftest test-decl-1
  (is-pos-metric [[::identity-1]
                  [::identity-2]
                  [::identity-3]
                  [::update-species identity]
                  [::update-species :clojure.core/identity]
                  [::update-species upper-case-col]
                  [::update-species ::upper-case-col]
                  [:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}
                  [:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))




(deftest test-decl-2
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                  [::duplicate-columns :type/numerical]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model} [:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))

(deftest test-decl-3
  (is-pos-metric [[::update-species (fn [col] (map  clojure.string/upper-case col))]
                  [:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                  [::identity-1]
                  [::identity-2]
                  [::identity-3]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))


(deftest test-decl-4
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species] {} :int64]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))

(deftest test-decl-5
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                  [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))




(deftest test-decl-1

  (is-thrown [[::update-species :upper-case-col]]))


(t/deftest x
  (t/is (pos?
         (eval-pipe [[::identity-1]
                     [::update-species identity]
                     [::update-species :clojure.core/identity]

                     [::update-species upper-case-col]
                     [::update-species ::upper-case-col]
                     [:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                     [:tech.v3.dataset.metamorph/categorical->number [:species ] {} :int64]
                     [:tech.v3.dataset.metamorph/set-inference-target :species]
                     {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))))


(comment

  ((update-species identity)
   {:metamorph/data data})
  ((update-species clojure.core/identity)
   {:metamorph/data data})
  ((update-species :clojure.core/identity)
   {:metamorph/data data})


  ((update-species :identity)
   {:metamorph/data data})

  ((update-species 'identity)
   {:metamorph/data data}))
(comment
  (eval-pipe [[:not-there]])

  (eval-pipe [[::not-there]])

  (eval-pipe [[:clojure.string/upper-case "hello"]]))





(comment


  (->
   (mm/->pipeline [[:ds-mm/update-column :a upper-case-col]])
   (apply [{:metamorph/data (tech.v3.dataset/->dataset {:a "small"})}]))


  (->
   (mm/->pipeline [[:ds-mm/update-column :a ::upper-case-col]])
   (apply [{:metamorph/data (tech.v3.dataset/->dataset {:a "x"})}]))

  (->
   (mm/->pipeline [[:ds-mm/update-column :a :upper-case-col]])
   (apply [{:metamorph/data (tech.v3.dataset/->dataset {:a "x"})}]))


  (tech.v3.dataset/update-column (tech.v3.dataset/->dataset {:a "x"}) :a :a-keyword)

  (upper-case-col (:species data)))

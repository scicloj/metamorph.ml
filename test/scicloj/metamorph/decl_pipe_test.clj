(ns scicloj.metamorph.decl-pipe-test
  (:require
   [clojure.string :as str]
   [clojure.test :refer [deftest is] :as t]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]))

(def data
  (rdatasets/datasets-iris))

(def iris-target-values (-> data :species distinct sort))
(def iris-target-values-capital
  (map str/upper-case
       (-> data :species distinct sort)))


(defn is-thrown [decl-pipe]

  (is (thrown? RuntimeException
               (->
                (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
                (nth 0) (nth 0) :train-transform :metric))))



(defn eval-pipe [decl-pipe]
  (->
   (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
   first first :train-transform :metric))

(defn is-pos-metric [decl-pipe]
  (t/is (pos?
         (->
          (ml/evaluate-pipelines [decl-pipe] (tc/split->seq data :holdout) loss/classification-accuracy :accuracy)
          first first :train-transform :metric))))

(defn is-zero-metric [decl-pipe]
  (t/is (zero?
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
                 (tc/add-column d (str n "-copy") (d n))) 
               ds column-names)))))


(defn upper-case-col [col]
  (map clojure.string/upper-case col))

(defn do-define-model []
  (ml/define-model! :test-model
    (fn train
      [feature-ds label-ds options]
      {})
    (fn predict
      [feature-ds thawed-model {:keys [target-columns
                                       target-categorical-maps
                                       top-k
                                       options]}]

      (let [target-column-name (first target-columns)]
        (ds/new-dataset [(ds/new-column target-column-name
                                        (repeat (tc/row-count feature-ds) 0)
                                        {:column-type :prediction
                                         :categorical-map (get target-categorical-maps target-column-name)})])))

      
    {}))






(deftest test-decl-1
  (do-define-model)
  (is-pos-metric [[::identity-1]
                  [::identity-2]
                  [::identity-3]
                  [::update-species identity]
                  [::update-species :clojure.core/identity]
                  ;; [::update-species upper-case-col]
                  ;; [::update-species ::upper-case-col]
                  [:tech.v3.dataset.metamorph/categorical->number [:species] iris-target-values]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}
                  [:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))




(deftest test-decl-2
  (do-define-model)
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species ] iris-target-values]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  [::duplicate-columns :type/feature]
                  
                  {:metamorph/id :model} [:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))

(deftest test-decl-3
  (do-define-model)
  (is-zero-metric [[::update-species (fn [col] (map  clojure.string/upper-case col))]
                   [:tech.v3.dataset.metamorph/categorical->number [:species ] iris-target-values :int64]
                   [::identity-1]
                   [::identity-2]
                   [::identity-3]
                   [:tech.v3.dataset.metamorph/set-inference-target :species]
                   {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))


(deftest test-decl-4
  (do-define-model)
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species]  iris-target-values :int64]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))

(deftest test-decl-5
  (do-define-model)
  (is-pos-metric [[:tech.v3.dataset.metamorph/categorical->number [:species ] iris-target-values :int64]
                  [:tech.v3.dataset.metamorph/update-column :species :clojure.core/identity]
                  [:tech.v3.dataset.metamorph/set-inference-target :species]
                  {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))




(deftest test-decl-throws
  (do-define-model)
  (is-thrown [[::update-species :upper-case-col]]))


(t/deftest x
  (do-define-model)
  (t/is (pos?
         (eval-pipe [[::identity-1]
                     [::update-species identity]
                     [::update-species :clojure.core/identity]

                     [::update-species upper-case-col]
                     [::update-species ::upper-case-col]
                     [:tech.v3.dataset.metamorph/categorical->number [:species ] iris-target-values-capital :int64]
                     [:tech.v3.dataset.metamorph/set-inference-target :species]
                     {:metamorph/id :model}[:scicloj.metamorph.ml/model (merge {:model-type :test-model})]]))))

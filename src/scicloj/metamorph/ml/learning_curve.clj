(ns scicloj.metamorph.ml.learning-curve
  (:require  [clojure.test :as t]
             [tech.v3.dataset]
             [scicloj.metamorph.core :as mm]
             [tech.v3.dataset.metamorph :as mds]
             [tablecloth.api :as tc]
             [tablecloth.pipeline :as tc-mm]
             [scicloj.metamorph.ml]
             [scicloj.metamorph.ml.loss]
             [scicloj.metamorph.ml.toydata]
             [tech.v3.datatype.functional :as fun]
             [scicloj.ml.smile.classification]))




(defn rounded-mean [coll]
  (Math/round (fun/mean coll)))


(defn mean+std [col]
  (+
   (fun/mean col)
   (fun/standard-deviation col)))

(defn mean-std [col]
  (-
   (fun/mean col)
   (fun/standard-deviation col)))


(defn learning-curve [ds pipe-fn train-sizes k]
  (def ds ds)
  (def pipe-fn pipe-fn)
  (def train-sizes train-sizes)
  (def k k)

  (let [splits (tc/split->seq ds :kfold {:k k})
        _ (def splits splits)
        metrices (->>
                  (mapv (fn [{:keys [train test]}]
                          (let [train-test-seq
                                (map-indexed
                                 (fn [index train-size]
                                   (let [train-subset (tc/head train (Math/round (* train-size (tc/row-count train))))]
                                     {:split-uid (str index)
                                      :train train-subset
                                      :test test}))
                                 train-sizes)
                                _ (def train-test-seq train-test-seq)
                                eval-results
                                (scicloj.metamorph.ml/evaluate-pipelines
                                 [pipe-fn]
                                 train-test-seq
                                 scicloj.metamorph.ml.loss/classification-accuracy
                                 :accuracy
                                 {:evaluation-handler-fn identity
                                  :return-best-pipeline-only false
                                  :return-best-crossvalidation-only false})]
                            (map  (fn [result]
                                    (def result result)
                                    (hash-map
                                     :train-size-index (:split-uid result)
                                     ;; :train-size train-size
                                     :train-ds-size (-> result :fit-ctx :metamorph/data tc/row-count)
                                     :test-ds-size (-> result :test-transform :ctx :model :scicloj.metamorph.ml/target-ds tc/row-count)
                                     :metric-test (get-in result [:test-transform :metric])
                                     :metric-train (get-in result [:train-transform :metric])))

                                  ;; train-sizes
                                  (flatten eval-results))))
                        splits)
                  flatten
                  (tc/dataset))]
    (def metrices metrices)
    (-> metrices
        (tc/group-by :train-size-index)

        (tc/aggregate {:metric-test      #(fun/mean (:metric-test %))
                       :metric-test-min  #(mean-std (:metric-test %))
                       :metric-test-max  #(mean+std (:metric-test %))
                       :metric-train     #(fun/mean (:metric-train %))
                       :metric-train-min #(mean-std (:metric-train %))
                       :metric-train-max #(mean+std (:metric-train %))
                       :train-ds-size    #(rounded-mean (:train-ds-size %))
                       :test-ds-size     #(rounded-mean (:test-ds-size %))}))))


        ;; (tc/aggregate {:metric-test      #(fun/mean (:metric-test %))
        ;;                ;;
        ;;                :metric-train     #(fun/mean (:metric-train %))}))))

        
        ;; (tc/drop-columns [:train-size-index :$group-name])

        ;; (tc/order-by :train-ds-size)




(comment
  (def splits (tc/split->seq ds :kfold {:k k}))

  (def pairs
    (flatten
     (mapv (fn [{:keys [train test]}]
             (let [train-test-seq
                   (map
                    (fn [train-size]
                      (let [train-subset (tc/head train (Math/round (* train-size (tc/row-count train))))]
                        {:train train-subset
                         :test test}))
                    train-sizes)]
               train-test-seq))
           (butlast splits))))

  (frequencies
   (map #(-> %  :train tc/row-count) pairs))

  (frequencies
   (map #(-> %  :train tc/row-count) splits))

  (def eval-results
    (flatten
     (scicloj.metamorph.ml/evaluate-pipelines
      [pipe-fn]
      pairs
      scicloj.metamorph.ml.loss/classification-accuracy
      :accuracy
      {:evaluation-handler-fn identity
       :return-best-pipeline-only false
       :return-best-crossvalidation-only false})))


  (def metrices
    (map  (fn [result]
            (def result result)
            (hash-map
             ;; :index index
             ;; :train-size train-size
             :train-ds-size (-> result :fit-ctx :metamorph/data tc/row-count)
             :test-ds-size (-> result :test-transform :ctx :model :scicloj.metamorph.ml/target-ds tc/row-count)
             :metric-test (get-in result [:test-transform :metric])
             :metric-train (get-in result [:train-transform :metric])))
          ;; (range)
          eval-results))

  (-> metrices
      tc/dataset
      (tc/group-by :train-ds-size)
      (tc/aggregate {:metric-test      #(fun/mean (:metric-test %))
                     :metric-test-min  #(fun/min (:metric-test %))
                     :metric-test-max  #(fun/max (:metric-test %))
                     :metric-train     #(fun/mean (:metric-train %))
                     :metric-train-min #(fun/min (:metric-train %))
                     :metric-train-max #(fun/max (:metric-train %))
                     :train-ds-size    #(rounded-mean (:train-ds-size %))
                     :test-ds-size     #(rounded-mean (:test-ds-size %))})


      [:metric-test
       :metric-test-min
       :metric-test-max
       :metric-train
       :metric-train-min
       :metric-train-max
       :train-ds-size
       :test-ds-size]
      [fun/mean

       fun/mean
       first
       first]
      (tc/rename-columns {:$group-name :train-ds-size})
      (tc/order-by :train-ds-size))



  (->
   (map :train-ds-size metrices)
   frequencies)


  (def splits (tc/split->seq ds :kfold {:k k}))

  (map
   #(-> % :train tc/row-count)
   splits)

  (map
   #(-> % :test tc/row-count)
   splits))

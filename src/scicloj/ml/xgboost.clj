;; (ns scicloj.ml.xgboost
;;   "Require this namespace to get xgboost support for classification and regression.
;;   Defines a full range of xgboost model definitions and supports xgboost explain
;;   functionality."
;;   (:require [clojure.set :as set]
;;             [clojure.tools.logging :as log]
;;             [scicloj.metamorph.ml :as ml]
;;             [scicloj.metamorph.ml.gridsearch :as ml-gs]
;;             [scicloj.ml.xgboost.model :as model]
;;             [tablecloth.api :as tc]
;;             [tech.v3.dataset :as ds]
;;             [tech.v3.dataset.modelling :as ds-mod]
;;             [tech.v3.dataset.tensor :as ds-tens]
;;             [tech.v3.dataset.utils :as ds-utils]
;;             [tech.v3.datatype :as dtype]
;;             [tech.v3.datatype.errors :as errors]
;;             [tech.v3.tensor :as dtt]
;;             [scicloj.ml.xgboost.csr :as csr]
;;             [camel-snake-kebab.core :as csk])
;;   (:import [java.io ByteArrayInputStream ByteArrayOutputStream]
;;            [java.util LinkedHashMap Map]
;;            [ml.dmlc.xgboost4j LabeledPoint]
;;            [ml.dmlc.xgboost4j.java Booster DMatrix XGBoost DMatrix$SparseType IObjective IEvaluation]
;;            [smile.util SparseArray SparseArray$Entry]))



;; (def objective-types
;;   {:linear-regression
;;    {:objective "reg:linear"
;;     :options [{:name :eta
;;                :description "Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative."}
;;               {:name :gamma
;;                :description "Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be."}
;;               {:name :max-depth
;;                :description "Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit. 0 is only accepted in lossguide growing policy when tree_method is set as hist or gpu_hist and it indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree."}
;;               {:name :min-child-weight
;;                :description "Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be."}
;;               {:name "max_delta_step "
;;                :description "Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update."}
;;               {:name "subsample"
;;                :description "Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration."}
;;               {:name "sampling_method"
;;                :description "The method to use to sample the training instances.\nuniform: each training instance has an equal probability of being selected. Typically set subsample >= 0.5 for good results.\n
;; gradient_based: the selection probability for each training instance is proportional to the regularized absolute value of gradients (more specifically, ).
;; subsample may be set to as low as 0.1 without loss of model accuracy. Note that this sampling method is only supported when tree_method is set to gpu_hist; other tree methods only support uniform sampling.
;; "}
;;               {:name "colsample_bytree"
;;                :description ""}
;;               {:name "colsample_bylevel"
;;                :description ""}
;;               {:name "colsample_bynode"
;;                :description ""}
;;               {:name "lambda"
;;                :description "L2 regularization term on weights. Increasing this value will make model more conservative."}
;;               {:name "alpha"
;;                :description "L1 regularization term on weights. Increasing this value will make model more conservative."}
;;               {:name "tree_method"
;;                :description ""}
;;               {:name "sketch_eps"
;;                :description ""}
;;               {:name "scale_pos_weight"
;;                :description ""}
;;               {:name "updater"
;;                :description ""}
;;               {:name "refresh_leaf"
;;                :description ""}
;;               {:name "process_type"
;;                :description ""}
;;               {:name "grow_policy"
;;                :description ""}
;;               {:name "max_leaves"
;;                :description ""}
;;               {:name "max_bin"
;;                :description ""}
;;               {:name "predictor"
;;                :description ""}
;;               {:name "num_parallel_tree"
;;                :description ""}
;;               {:name "monotone_constraints"
;;                :description ""}
;;               {:name "interaction_constraints"
;;                :description ""}]}







;;    :squared-error-regression {:objective "reg:squarederror"}


;;    :logistic-regression {:objective  "reg:logistic"}
;;    ;;logistic regression for binary classification
;;    :logistic-binary-classification {:objective "binary:logistic"}
;;    ;; logistic regression for binary classification, output score before logistic
;;    ;; transformation
;;    :logistic-binary-raw-classification {:objective "binary:logitraw"}
;;    ;;hinge loss for binary classification. This makes predictions of 0 or 1, rather
;;    ;;than producing probabilities.
;;    :binary-hinge-loss {:objective "binary:hinge"}
;;    ;; versions of the corresponding objective functions evaluated on the GPU; note that
;;    ;; like the GPU histogram algorithm, they can only be used when the entire training
;;    ;; session uses the same dataset
;;    :gpu-linear-regression {:objective "gpu:reg:linear"}
;;    :gpu-logistic-regression {:objective "gpu:reg:logistic"}
;;    :gpu-binary-logistic-classification {:objective "gpu:binary:logistic"}
;;    :gpu-binary-logistic-raw-classification {:objective "gpu:binary:logitraw"}

;;    ;; poisson regression for count data, output mean of poisson distribution
;;    ;; max_delta_step is set to 0.7 by default in poisson regression (used to safeguard
;;    ;; optimization)
;;    :count-poisson {:objective "count:poisson"}

;;    ;; Cox regression for right censored survival time data (negative values are
;;    ;; considered right censored). Note that predictions are returned on the hazard ratio
;;    ;; scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function
;;    ;; h(t) = h0(t) * HR).
;;    :survival-cox {:objective "survival:cox"}
;;    ;; set XGBoost to do multiclass classification using the softmax objective, you also
;;    ;; need to set num_class(number of classes)
;;    :multiclass-softmax {:objective "multi:softmax"}
;;    ;; same as softmax, but output a vector of ndata * nclass, which can be further
;;    ;; reshaped to ndata * nclass matrix. The result contains predicted probability of
;;    ;; each data point belonging to each class.
;;    :multiclass-softprob {:objective "multi:softprob"}
;;    ;; Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
;;    :rank-pairwise {:objective "rank:pairwise"}
;;    ;; Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative
;;    ;; Gain (NDCG) is maximized
;;    :rank-ndcg {:objective "rank:ndcg"}
;;    ;; Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is
;;    ;; maximized
;;    :rank-map {:objective "rank:map"}
;;    ;; gamma regression with log-link. Output is a mean of gamma distribution. It might
;;    ;; be useful, e.g., for modeling insurance claims severity, or for any outcome that
;;    ;; might be gamma-distributed.
;;    :gamma-regression {:objective "reg:gamma"}
;;    ;; Tweedie regression with log-link. It might be useful, e.g., for modeling total
;;    ;; loss in insurance, or for any outcome that might be Tweedie-distributed.
;;    :tweedie-regression {:objective "reg:tweedie"}})




;; (defmulti ^:private model-type->xgboost-objective
;;   (fn [model-type]
;;     model-type))


;; (defmethod model-type->xgboost-objective :default
;;   [model-type]
;;   (if-let [retval (get-in objective-types [model-type :objective])]
;;     retval
;;     (throw (ex-info "Unrecognized xgboost model type"
;;                     {:model-type model-type
;;                      :possible-types (keys objective-types)}))))


;; (defmethod model-type->xgboost-objective :regression
;;   [_]
;;   (model-type->xgboost-objective :squared-error-regression))


;; (defmethod model-type->xgboost-objective :binary-classification
;;   [_]
;;   (model-type->xgboost-objective :logistic-binary-classification))


;; (defmethod model-type->xgboost-objective :classification
;;   [_]
;;   (model-type->xgboost-objective :multiclass-softprob))



;; (defn- sparse->labeled-point [^SparseArray sparse target weight n-sparse-columns]
;;   (let [x-i-s
;;         (mapv
;;          #(hash-map :i (.i ^SparseArray$Entry %)
;;                     :x (.x ^SparseArray$Entry %))
;;          (iterator-seq
;;           (.iterator sparse)))]
;;     (LabeledPoint. target
;;                    n-sparse-columns
;;                    (into-array Integer/TYPE (map :i x-i-s))
;;                    (into-array Float/TYPE (map :x x-i-s))
;;                    (float weight)
;;                    -1
;;                    Float/NaN)))

;; (defn sparse-feature->dmatrix
;;   "converts columns containing smile.util.SparseArray to a sparse dmatrix"
;;   [feature-ds target-ds weight-ds sparse-column n-sparse-columns]
;;   {:dmatrix
;;    (DMatrix.
;;     (.iterator
;;      ^Iterable (map
;;                 (fn [features target weight] (sparse->labeled-point features target weight n-sparse-columns))
;;                 (get feature-ds sparse-column)
;;                 (or  (get target-ds (first (ds-mod/inference-target-column-names target-ds)))
;;                      (repeat 0.0))
;;                 (if-not weight-ds
;;                   (repeat 1.0)
;;                   (dtype/->reader (ds-tens/dataset->tensor weight-ds :float32)))))
;;     nil)})


;; (defn tidy-text-bow-ds->dmatrix [feature-ds target-ds text-feature-column n-col]
;;   (let [ds (if (seq target-ds)
;;              (assoc feature-ds :label (:label target-ds))
;;              feature-ds)

;;         zero-baseddocs-map
;;         (zipmap
;;          (-> ds :document distinct)
;;          (range))

;;         bow-zeroed
;;         (-> ds
;;             (tc/select-columns [:document :token-idx text-feature-column])
;;             (tc/add-or-replace-column
;;              :document-zero-based
;;              #(map zero-baseddocs-map (:document %))))

;;         sparse-features
;;         (-> bow-zeroed
;;             (tc/select-columns [:document-zero-based :token-idx text-feature-column])
;;             (tc/order-by [:document-zero-based :token-idx])
;;             (tc/rows))


;;         csr  (csr/->csr sparse-features)

;;         labels
;;         (->
;;          ds
;;          (tc/group-by :document)
;;          (tc/aggregate #(-> % :label first))
;;          (tc/column "summary"))

;;         m
;;         (DMatrix.
;;          (long-array (:row-pointers csr))
;;          (int-array (:column-indices csr))
;;          (float-array (:values csr))
;;          DMatrix$SparseType/CSR
;;          n-col)]

;;     (when (seq target-ds)
;;       (.setLabel m (float-array labels)))

;;     {:dmatrix m
;;      :dmatrix-order
;;      (-> bow-zeroed
;;          (tc/select-columns [:document :document-zero-based])
;;          (tc/unique-by [:document :document-zero-based])
;;          (tc/rename-columns {:document-zero-based :row-nr}))}))


;; (defn- dataset->labeled-point-iterator
;;   "Create an iterator to labeled points from a possibly quite large
;;   sequence of maps.  Sets expected length to length of first entry"
;;   ^Iterable [feature-ds target-ds weight-ds]
;;   (let [feature-tens (ds-tens/dataset->tensor feature-ds :float32)
;;         target-tens (when target-ds
;;                       (ds-tens/dataset->tensor target-ds :float32))
;;         weight-tens (when weight-ds
;;                       (ds-tens/dataset->tensor weight-ds :float32))]
;;     (errors/when-not-errorf
;;      (or (not target-ds)
;;          (== 1 (ds/column-count target-ds)))
;;      "Multi-column regression/classification is not supported.  Target ds has %d columns"
;;      (ds/column-count target-ds))
;;     (map (fn [features target weight]
;;            (LabeledPoint. (float target)
;;                           (first (dtype/shape features))
;;                           nil
;;                           (dtype/->float-array features)
;;                           (float weight)
;;                           -1
;;                           Float/NaN))
;;          feature-tens
;;          (or (when target-tens (dtype/->reader target-tens))
;;              (repeat (float 0.0)))
;;          (or (when weight-tens (dtype/->reader weight-tens))
;;              (repeat (float 1.0))))))

;; (defn dataset->dmatrix
;;   "Dataset is a sequence of maps.  Each contains a feature key.
;;   Returns a dmatrix."
;;   ([feature-ds target-ds weights-ds]
;;    {:dmatrix
;;     (DMatrix. (.iterator (dataset->labeled-point-iterator feature-ds target-ds weights-ds))
;;               nil)})
;;   ([feature-ds target-ds]
;;    (dataset->dmatrix feature-ds target-ds nil))
;;   ([feature-ds]
;;    (dataset->dmatrix feature-ds nil nil)))


;; (defn- options->model-type
;;   [options]
;;   (or (when (:model-type options)
;;         (keyword (name (:model-type options))))
;;       :linear-regression))

;; (defn- options->objective
;;   [options]
;;   (or (:objective options)
;;       (-> options
;;           options->model-type
;;           model-type->xgboost-objective)))


;; (defn- multiclass-model-type?
;;   [model-type]
;;   (contains? #{:multiclass-softmax
;;                :multiclass-softprob
;;                :classification} model-type))


;; (def ^:private hyperparameters
;;   {:subsample (ml-gs/linear 0.7 1.0 3)
;;    :scale-pos-weight (ml-gs/linear 0.7 1.31 6)
;;    :max-depth (ml-gs/linear 1 10 10 :int64)
;;    :lambda (ml-gs/linear 0.01 0.31 30)
;;    :gamma (ml-gs/linear 0.001 1 10)
;;    :eta (ml-gs/linear 0 1 10)
;;    :round (ml-gs/linear 5 46 5 :int64)
;;    :alpha (ml-gs/linear 0.01 0.31 30)})

;; (defn ->dmatrix [feature-ds target-ds weight-ds sparse-column n-sparse-columns]
;;   (if sparse-column
;;     (if (= (-> feature-ds (get sparse-column) first class)
;;            SparseArray)
;;       (sparse-feature->dmatrix feature-ds target-ds weight-ds sparse-column n-sparse-columns)
;;       (do (assert (not weight-ds) ":sample-weights on TidyText not supported")
;;           (tidy-text-bow-ds->dmatrix feature-ds target-ds sparse-column n-sparse-columns)))

;;     (dataset->dmatrix feature-ds target-ds weight-ds)))




;; (defn- thaw-model
;;   [model-data]
;;   (-> (if (map? model-data)
;;         (:model-data model-data)
;;         model-data)
;;       (ByteArrayInputStream.)
;;       (XGBoost/loadModel)))



;; (defn train-from-dmatrix
;;   [train-dmat-map feature-cnames target-cnames options label-map objective]
;;   ;;XGBoost uses all cores so serialization here avoids over subscribing
;;   ;;the machine.
;;   (locking #'multiclass-model-type?
;;     (let [train-dmat (:dmatrix train-dmat-map)
;;           sparse-column-or-nil (:sparse-column options)
;;           base-watches (or (:watches options) {})
;;           watches (->> base-watches
;;                        (reduce (fn  [^Map watches [k v]]
;;                                  (.put watches (ds-utils/column-safe-name k)
;;                                        (:dmatrix
;;                                         (->dmatrix
;;                                          (ds/select-columns v feature-cnames)
;;                                          (ds/select-columns v target-cnames)
;;                                          nil
;;                                          sparse-column-or-nil
;;                                          (:n-sparse-columns options))))
;;                                  watches)
;;                                ;;Linked hash map to preserve order
;;                                (LinkedHashMap.)))
;;           round (or (:round options) 25)
;;           custom-obj? (fn? objective)
;;           custom-eval? (vector? (:eval-metric options))
;;           early-stopping-round (or (when (:early-stopping-round options)
;;                                      (int (:early-stopping-round options)))
;;                                    0)
;;           _ (when (and (> (count watches) 1)
;;                        (not (instance? LinkedHashMap (:watches options)))
;;                        (not (sequential? (:watches options)))
;;                        (not= 0 early-stopping-round))
;;               (log/warn "Early stopping indicated but watches has undefined iteration order.
;;   Early stopping will always use the 'last' of the watches as defined by the iteration
;;   order of the watches map.  Consider using a java.util.LinkedHashMap for watches.
;;   https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j/src/main/java/ml/dml
;;   c/xgboost4j/java/XGBoost.java#L208"))
;;           watch-names (->> base-watches
;;                            (map-indexed (fn [idx [k v]]
;;                                           [idx k]))
;;                            (into {}))
;;           cleaned-options
;;           (->
;;            (dissoc options :model-type :watches :objective :eval-metric)
;;            (cond-> (not custom-obj?) (assoc :objective objective))
;;            (cond-> (not custom-eval?) (assoc :eval-metric (:eval-metric options))))
;;           params (->>  cleaned-options
;;                        ;;Adding in some defaults
;;                        (merge
;;                         {:alpha 0.0
;;                          :eta 0.3
;;                          :lambda 1.0
;;                          :max-depth 6
;;                          :subsample 0.87}

;;                         cleaned-options
;;                         (when label-map
;;                           {:num-class (count label-map)}))
;;                        (map (fn [[k v]]
;;                               (when v
;;                                 [(csk/->snake_case_string k) v])))

;;                        (remove nil?)
;;                        (into {}))
;;           ^"[[F" metrics-data (when-not (empty? watches)
;;                                 (->> (repeatedly (count watches)
;;                                                  #(float-array round))
;;                                      (into-array)))
;;           ^Booster model (XGBoost/train train-dmat params
;;                                         (long round)
;;                                         (or watches {}) metrics-data
;;                                         (when custom-obj?
;;                                           (reify IObjective
;;                                             (getGradient [_ predicts dtrain]
;;                                               (objective predicts dtrain))))
;;                                         (when custom-eval?
;;                                           (let [[metric-name eval-fn] (:eval-metric options)]
;;                                             (reify IEvaluation
;;                                               (getMetric [_] metric-name)
;;                                               (eval [_ predicts dtrain]
;;                                                 (eval-fn predicts dtrain)))))
;;                                         (int early-stopping-round))
;;           out-s (ByteArrayOutputStream.)]
;;       (.saveModel model out-s)
;;       (merge
;;        {:model-data (.toByteArray out-s)
;;         :custom-obj? custom-obj?
;;         ;:tidy-text-dmatrix-order (:dmatrix-order train-dmat-map)
;;         }
;;        (when (seq watches)
;;          {:metrics
;;           (->> watches
;;                (map-indexed vector)
;;                (map (fn [[watch-idx [watch-name watch-data]]]
;;                       [(get watch-names watch-idx)
;;                        (aget metrics-data watch-idx)]))
;;                (into {})
;;                (ds/->>dataset {:dataset-name :metrics}))})))))


;; (defn train [data label-ds options]
;;   (if (ds/dataset? data)
;;     (let [feature-ds data
;;           sparse-column-or-nil (:sparse-column options)
;;           feature-cnames (ds/column-names feature-ds)
;;           target-cnames (ds/column-names label-ds)
;;           train-dmat (->dmatrix feature-ds label-ds (:sample-weights options) sparse-column-or-nil (:n-sparse-columns options))
;;           model-type (options->model-type options)
;;           objective (options->objective options)
          
;;           label-map (when (multiclass-model-type? model-type)
;;                       (ds-mod/inference-target-label-map label-ds))]
;;       (train-from-dmatrix train-dmat feature-cnames target-cnames options label-map objective))

;;     (train-from-dmatrix {:dmatrix data} nil nil options nil (options->objective options))
    
;;     ))

;; (defn- predict
;;   [feature-ds thawed-model {:keys [target-columns target-categorical-maps target-datatypes model-data options]}]
;;   (def feature-ds feature-ds)
;;   (def options options)
  
;;   (let [sparse-column-or-nil (:sparse-column options)
;;         _ (def sparse-column-or-nil sparse-column-or-nil)
;;         dmatrix-context 
;;         (if (ds/dataset? feature-ds)
;;           (->dmatrix feature-ds nil nil sparse-column-or-nil (:n-sparse-columns options))
;;           {:dmatrix feature-ds}
;;           )
        
;;         dmatrix  (:dmatrix dmatrix-context)
;;         prediction (.predict ^Booster thawed-model dmatrix (:custom-obj? model-data))

;;         predict-tensor
;;         (->> prediction
;;              (dtt/->tensor))
;;         target-cname (first target-columns)

;;         prediction-df
;;         (if (multiclass-model-type? (options->model-type options))
;;           (->
;;            (model/finalize-classification predict-tensor
;;                                           target-cname
;;                                           target-categorical-maps)

;;            (tech.v3.dataset.modelling/probability-distributions->label-column
;;             target-cname
;;             (get target-datatypes target-cname))
;;            (ds/update-column (first  target-columns)
;;                              #(vary-meta % assoc :column-type :prediction)))
;;           (model/finalize-regression predict-tensor target-cname))]


;;     (if (:dmatrix-order dmatrix-context)
;;       (assoc prediction-df
;;              :document
;;              (-> dmatrix-context
;;                  :dmatrix-order
;;                  (tc/order-by :row-nr)
;;                  :document))
;;       prediction-df)))




;; (defn- explain
;;   [thawed-model {:keys [feature-columns options]}
;;    {:keys [importance-type]
;;     :or {importance-type "gain"}}]
;;   (let [^Booster booster thawed-model
;;         sparse-column-or-nil (:sparse-column options)]
;;     (if sparse-column-or-nil
;;       (let [score-map (.getScore booster "" (str importance-type))]
;;         (ds/->dataset {:feature (keys score-map)
;;                        (keyword importance-type) (vals score-map)}))
;;       (let [feature-col-map (->> feature-columns
;;                                  (map (fn [name]
;;                                         [name (ds-utils/column-safe-name name)]))
;;                                  (into {}))
;;             feature-columns (into-array String (map #(get feature-col-map %)
;;                                                     feature-columns))
;;             ^Map score-map (.getScore booster
;;                                       ^"[Ljava.lang.String;" feature-columns
;;                                       ^String importance-type)
;;             col-inv-map (set/map-invert feature-col-map)]
;;         ;;It's not a great map...Something is off about iteration so I have
;;         ;;to transform it back into something sane.
;;         (->> (keys score-map)
;;              (map (fn [item-name]
;;                     {:importance-type importance-type
;;                      :colname (get col-inv-map item-name)
;;                      (keyword importance-type) (.get score-map item-name)}))
;;              (sort-by (keyword importance-type) >)
;;              (ds/->>dataset))))))

;; (defn- assoc-if
;;   [m key value]
;;   (if value (assoc m key value) m))

;; (defn- reg-def->options [reg-def]
;;   (vec
;;    (concat [:map]
;;            (mapv (fn [o]

;;                    (vector
;;                     (csk/->kebab-case-keyword (:name o))
;;                     {:optional true}
;;                     :any))
;;                  (:options reg-def)))))

;; (doseq [objective (concat [:regression :classification]
;;                           (keys objective-types))]
;;   (let [reg-def (get objective-types objective)
;;         model-meta
;;         {:thaw-fn thaw-model
;;          :explain-fn explain
;;          :hyperparameters hyperparameters
;;          :documentation {:javadoc "https://xgboost.readthedocs.io/en/latest/jvm/javadocs/index.html"
;;                          :user-guide "https://xgboost.readthedocs.io/en/latest/jvm/index.html"}}
;;         model-meta (assoc-if model-meta :options (reg-def->options reg-def))]
;;     (ml/define-model! (keyword "xgboost" (name objective))
;;       train predict model-meta)))(ns scicloj.ml.xgboost)
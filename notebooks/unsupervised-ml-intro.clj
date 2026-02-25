;; # Introduction to Unsupervised Machine Learning with metamorph.ml
;;
;; This tutorial introduces unsupervised machine learning using the metamorph.ml
;; library. We'll cover:
;;
;; - Clustering algorithms (K-means, hierarchical clustering)
;; - Dimensionality reduction (PCA)
;; - Feature scaling and preprocessing
;; - Text feature extraction with TF-IDF
;; - Evaluation techniques for unsupervised learning
;; - Building complete unsupervised ML pipelines
;;
;; Unlike supervised learning, unsupervised learning works with unlabeled data
;; to discover hidden patterns, group similar observations, or reduce
;; dimensionality for visualization and preprocessing.

;; BROKEN 

(ns unsupervised-ml-intro
  (:require
   [clojure.string :as str]
   [tablecloth.api :as tc]
   [tablecloth.pipeline :as tc-mm]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.datatype.functional :as dfn]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.preprocessing :as prep]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [scicloj.ml.smile.projections :as projections]
   [scicloj.ml.smile.clustering :as clustering]
   [scicloj.ml.smile.manifold]))

;; ## Part 1: Clustering with K-Means
;;
;; Clustering groups similar data points together without predefined labels.
;; K-means is one of the most popular clustering algorithms.

;; ### 1.1 Loading and Exploring Data
;;
;; We'll use the Iris dataset, but ignore the species labels to treat it as
;; an unsupervised problem.

(def iris-ds
  (-> (rdatasets/datasets-iris)
      (ds/drop-columns [:rownames :species])))  ; Remove labels for unsupervised learning


;; **Iris dataset (unlabeled):**

(tc/head iris-ds 5)

^{:kind/md true :kindly/hide-code true}
(str "**Dataset shape:** "
     (first (ds/shape iris-ds)) " rows × "
     (second (ds/shape iris-ds)) " columns")

;; View column statistics:

(ds/descriptive-stats iris-ds)

;; ### 1.2 Data Preprocessing
;;
;; Before clustering, we should standardize features so that features with
;; larger scales don't dominate the distance calculations.

(def numeric-cols (tc/column-names (cf/numeric iris-ds)))

(def preprocessing-pipeline
  (mm/pipeline
   (prep/std-scale numeric-cols {:mean? true :stddev? true})))

;; Apply preprocessing in fit mode:

(def fitted-preproc-ctx
  (preprocessing-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

(def scaled-iris
  (:metamorph/data fitted-preproc-ctx))

^{:kind/md true :kindly/hide-code true}
["**Scaled data (first 5 rows):**"]

(tc/head scaled-iris 5)

;; Check that scaling worked (mean ≈ 0, std ≈ 1):

^{:kind/md true :kindly/hide-code true}
["**Scaled data statistics:**"]

(ds/descriptive-stats scaled-iris)

;; ### 1.3 K-Means Clustering
;;
;; K-means partitions data into K clusters by minimizing within-cluster variance.

(def kmeans-pipeline
  (mm/pipeline
   ;; Standardize features
   (prep/std-scale numeric-cols {:mean? true :stddev? true})
   ;; K-means clustering
   {:metamorph/id :model}
   (ml/model {:model-type :fastmath.cluster/k-means
              :clustering-method-args [3 100 1e-4]})))

;; * :k 3            -> Number of clusters
;; * :max-iter 100   -> Maximum iterations
;; * :tolerance 1e-4 -> Convergence tolerance

;; Fit the clustering model:

(def kmeans-result
  (kmeans-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

;; Extract the trained model:

(def kmeans-model
  (-> kmeans-result :model :model-data))

^kind/println
(-> kmeans-model :obj str)

;; **K-means clustering complete!**

;; ### 1.4 Analyzing Cluster Assignments
;;
;; Get the cluster assignments (which cluster each point belongs to):

(def cluster-assignments
  (-> kmeans-model :clustering))


^{:kind/md true :kindly/hide-code true}
(str "**Number of unique clusters found:** "
     (count (distinct cluster-assignments)))

;; Add cluster assignments to the original data:

(def iris-with-clusters
  (tc/add-column iris-ds :cluster cluster-assignments))

^{:kind/md true :kindly/hide-code true}
["**Data with cluster assignments:**"]

(tc/head iris-with-clusters 10)

;; View cluster sizes:

(-> iris-with-clusters
    (tc/group-by [:cluster])
    (tc/aggregate {:count tc/row-count})
    (tc/order-by [:cluster]))

;; ### 1.5 Cluster Statistics
;;
;; Examine the characteristics of each cluster:

(-> iris-with-clusters
    (tc/group-by [:cluster])
    (tc/aggregate {:mean-sepal-length #(dfn/mean (% :sepal-length))
                   :mean-sepal-width #(dfn/mean (% :sepal-width))
                   :mean-petal-length #(dfn/mean (% :petal-length))
                   :mean-petal-width #(dfn/mean (% :petal-width))
                   :count tc/row-count})
    (tc/order-by [:cluster]))

;; ## Part 2: Dimensionality Reduction with PCA
;;
;; Principal Component Analysis (PCA) reduces the number of features while
;; retaining most of the variance in the data. It's useful for:
;; - Visualization (reducing to 2-3 dimensions)
;; - Preprocessing before modeling
;; - Noise reduction

;; ### 2.1 Applying PCA

(def pca-pipeline
  (mm/pipeline
   ;; Standardize features (required for PCA)
   (prep/std-scale numeric-cols {:mean? true :stddev? true})
   ;; PCA
   {:metamorph/id :pca}
   (projections/reduce-dimensions :pca-cov 2 numeric-cols {})))          ; Reduce to 2 dimensions

;; Fit PCA:

(def pca-result (mm/fit-pipe iris-ds pca-pipeline))



(def pca-transformed
  (-> pca-result
      :metamorph/data
      (tc/select-columns ["pca-cov-0" "pca-cov-1"])))


^{:kind/md true :kindly/hide-code true}
["**PCA transformation complete!**"]

^{:kind/md true :kindly/hide-code true}
(str "**Original dimensions:** " (first (ds/shape iris-ds)))

^{:kind/md true :kindly/hide-code true}
(str "**Reduced dimensions:** " (first (ds/shape pca-transformed)))

;; View the transformed data:

(tc/head pca-transformed 10)

;; ### 2.2 Explained Variance
;;
;; PCA components capture different amounts of variance. The first component
;; captures the most variance, the second captures the second most, etc.

(def pca-model
  (-> pca-result :pca :fit-result :model))

;; The PCA model contains information about explained variance and component loadings.

(-> pca-model bean keys)

;; ex. cummulative variance proportion:
(-> pca-model .getCumulativeVarianceProportion vec)

;; ### 2.3 Combining PCA with Clustering
;;
;; A common pattern is to use PCA for dimensionality reduction, then cluster
;; in the reduced space:

(def pca-kmeans-pipeline
  (mm/pipeline
   ;; Standardize
   (prep/std-scale numeric-cols {:mean? true :stddev? true})
   ;; Reduce dimensions
   {:metamorph/id :pca}
   (projections/reduce-dimensions :pca-cov 2 numeric-cols {})

   (tc-mm/select-columns ["pca-cov-0" "pca-cov-1"])

   ;; Cluster in reduced space
   {:metamorph/id :kmeans}
   (clustering/cluster :k-means [3 100 1e-4] :clustering)))


(def pca-kmeans-result
  (mm/fit-pipe iris-ds pca-kmeans-pipeline))




;; **PCA + K-means pipeline complete!**

;; Get cluster assignments from the combined pipeline:


(def pca-clusters
  (-> pca-kmeans-result :metamorph/data :clustering seq))

(def iris-pca-clusters
  (tc/add-column iris-ds :pca-cluster pca-clusters))

(-> iris-pca-clusters
    (tc/group-by [:pca-cluster])
    (tc/aggregate {:count tc/row-count})
    (tc/order-by [:pca-cluster]))

;; ## Part 3: Hierarchical Clustering
;;
;; Hierarchical clustering builds a tree (dendrogram) of clusters, allowing
;; exploration at different granularities.

;; (def hclust-pipeline
;;   (mm/pipeline
;;    (prep/std-scale cf/numeric {:mean? true :stddev? true})
;;    {:metamorph/id :model}
;;    (ml/model {:model-type :smile.clustering/hierarchical
;;               :k 3                      ; Cut tree to get 3 clusters
;;               :linkage :complete})))    ; Linkage method: :single, :complete, :average, :ward

;; (def hclust-result
;;   (hclust-pipeline
;;    {:metamorph/data iris-ds
;;     :metamorph/mode :fit}))

;; (def hclust-assignments
;;   (:cluster-id (:metamorph/data hclust-result)))

;; (def iris-hclust
;;   (tc/add-column iris-ds :hclust-cluster hclust-assignments))

;; ^{:kind/md true :kindly/hide-code true}
;; "**Hierarchical clustering results:**"

;; (-> iris-hclust
;;     (tc/group-by [:hclust-cluster])
;;     (tc/aggregate {:count tc/row-count
;;                    :mean-petal-length #(dfn/mean (% :petal-length))})
;;     (tc/order-by [:hclust-cluster]))

;; ## Part 4: Feature Engineering and Preprocessing
;;
;; Proper preprocessing is crucial for unsupervised learning.

;; ### 4.1 Standard Scaling (Z-score Normalization)

(def std-scaled-ds
  (-> (mm/pipeline
       (prep/std-scale numeric-cols {:mean? true :stddev? true}))
      (apply [{:metamorph/data iris-ds
               :metamorph/mode :fit}])
      :metamorph/data))


;; **Standard scaling:** Transforms features to have mean=0 and std=1

(ds/descriptive-stats std-scaled-ds [:mean :standard-deviation])

;; ### 4.2 Min-Max Scaling

(def minmax-scaled-ds
  (-> (mm/pipeline
       (prep/min-max-scale numeric-cols {:min -1 :max 1}))
      (apply [{:metamorph/data iris-ds
               :metamorph/mode :fit}])
      :metamorph/data))


;; **Min-max scaling:** Transforms features to a specific range (here: -1 to 1)

(ds/descriptive-stats minmax-scaled-ds [:min :max])

;; ### 4.3 Robust Scaling for Outliers
;;
;; When data has outliers, standard scaling can be affected. Consider using
;; quantile-based scaling or removing outliers first.

;; ## Part 5: Text Clustering with TF-IDF
;;
;; Unsupervised learning is commonly used for text data. Let's create a
;; simple example of text clustering.

;; Create a small text dataset:

(def documents-ds
  (tc/dataset {:doc-id (range 6)
               :text ["machine learning is fascinating"
                      "deep learning uses neural networks"
                      "I love pizza and pasta"
                      "Italian food is delicious"
                      "supervised learning needs labels"
                      "My favorite food is sushi"]}))


;; **Text documents:**

documents-ds

;; ### 5.1 Tokenization and TF-IDF
;;
;; First, we need to tokenize text and compute TF-IDF features:

(require '[scicloj.metamorph.ml.text :as text])

;; Tokenize documents:

(defn simple-tokenize [text]
  (-> text
      str/lower-case
      (str/split #"\s+")))

(def tokenized-docs
  (tc/add-column documents-ds
                 :tokens
                 (map simple-tokenize (:text documents-ds))))

^{:kind/md true :kindly/hide-code true}
["**Tokenized documents:**"]

(tc/select-columns tokenized-docs [:doc-id :tokens])

;; Convert to tidy text format (one row per token):

(def tidy-docs
  (tc/dataset
   (mapcat (fn [row]
             (map (fn [token]
                    {:doc-id (:doc-id row)
                     :token token})
                  (:tokens row)))
           (ds/mapseq-reader tokenized-docs))))


;; **Tidy text format (sample):**

(tc/head tidy-docs 10)

;; Compute term frequencies:

(def term-counts
  (-> tidy-docs
      (tc/group-by [:doc-id :token])
      (tc/aggregate {:n tc/row-count})))


;; **Term counts (sample):**

(tc/head term-counts 10)

;; ### 5.2 Document Similarity
;;
;; After TF-IDF vectorization, we can cluster documents based on their
;; semantic similarity.

^{:kind/md true :kindly/hide-code true}
["**Note:** Full TF-IDF clustering requires converting the term-document matrix
to a format suitable for clustering. The `scicloj.metamorph.ml.text` namespace
provides functions for this."]

;; ## Part 6: Evaluation Metrics for Unsupervised Learning
;;
;; Unlike supervised learning, we don't have true labels to evaluate against.
;; Instead, we use intrinsic quality measures.

;; ### 6.1 Within-Cluster Sum of Squares (WCSS)
;;
;; WCSS measures cluster compactness. Lower is better.

(defn euclidean-distance [point1 point2]
  (dfn/sqrt
   (dfn/sum
    (dfn/pow
     (dfn/- point1 point2)
     2))))

;; ### 6.2 Silhouette Score
;;
;; Silhouette score measures how similar a point is to its own cluster
;; compared to other clusters. Ranges from -1 to 1, higher is better.

^{:kind/md true :kindly/hide-code true}
["**Common evaluation approaches:**
- **Elbow method:** Plot WCSS vs. number of clusters, look for the 'elbow'
- **Silhouette analysis:** Compute silhouette score for each point
- **Gap statistic:** Compare within-cluster dispersion to null reference
- **Domain validation:** Check if clusters make sense in your domain"]

;; ## Part 7: The Elbow Method for Choosing K
;;
;; The elbow method helps determine the optimal number of clusters.

(defn fit-kmeans-for-k [ds k]
  (let [pipeline (mm/pipeline
                  (prep/std-scale numeric-cols {:mean? true :stddev? true})
                  {:metamorph/id :model}
                  (ml/model {:model-type :fastmath.cluster/k-means
                             :clustering-method-args [k 100]}))
        result (pipeline {:metamorph/data ds
                          :metamorph/mode :fit})]
    {:k k
     :model (-> result :model :model-data)
     :result result}))

;; Try different values of K:

(def k-values [2 3 4 5 6 7 8])

^{:kind/md true :kindly/hide-code true}
["**Testing different values of K...**"]

(def elbow-results
  (mapv #(fit-kmeans-for-k iris-ds %) k-values))

^{:kind/md true :kindly/hide-code true}
(str "**Tried K values:** " k-values)


;; To find the optimal K, plot WCSS vs K and look for an 'elbow' where the rate of decrease slows down.


;;elbow-results
;; ## Part 8: Complete Unsupervised Workflow
;;
;; Here's a complete workflow combining preprocessing, dimensionality
;; reduction, and clustering:

(defn unsupervised-workflow
  "Complete unsupervised learning workflow"
  [dataset n-components n-clusters]

  (let [;; Build the pipeline
        pipeline (mm/pipeline
                  ;; Step 1: Standardize features
                  (prep/std-scale numeric-cols {:mean? true :stddev? true})

                  ;; Step 2: Dimensionality reduction with PCA
                  {:metamorph/id :pca}
                  (projections/reduce-dimensions :pca-cov n-components numeric-cols {})

                  (tc-mm/drop-columns [:sepal-length :sepal-width :petal-length :petal-width])
                  ;; Step 3: Cluster in reduced space
                  {:metamorph/id :kmeans}
                  (clustering/cluster :k-means [n-clusters 100 1e-4] :clustering))

        ;; Fit the pipeline
        fitted (mm/fit-pipe dataset pipeline)


        ;; transform dataset

        pca-model (-> fitted :pca :model-data)
        kmeans-model (-> fitted :kmeans :model-data)

        cluster-assignments (-> fitted :kmeans :clustering)]



    {:pipeline pipeline
     :cluster-assignments cluster-assignments
     :pca-model pca-model
     :kmeans-model kmeans-model
     :fitted-ctx fitted}))



(def workflow-result
  (unsupervised-workflow iris-ds 2 3))

;; workflow-result

(-> workflow-result keys)
;; **Complete workflow executed!**

;; Add clusters to original data:

(def iris-final
  (-> iris-ds
      (tc/add-column :cluster (:cluster-assignments workflow-result))))


;; **Final clustered data:**

(tc/head iris-final 10)

;; Cluster statistics:

(-> iris-final
    (tc/group-by [:cluster])
    (tc/aggregate {:count tc/row-count
                   :avg-sepal-length #(dfn/mean (% :sepal-length))
                   :avg-petal-length #(dfn/mean (% :petal-length))
                   :avg-petal-width #(dfn/mean (% :petal-width))})
    (tc/order-by [:cluster]))

;; ## Part 9: Applying Models to New Data
;;
;; Once trained, unsupervised models can transform new data using the
;; learned patterns.

;; Create some new data (using a sample from the original):


(def new-data
  (tc/random iris-ds 80))


;; **New data to transform:**

(tc/head new-data 5)

;; Apply the trained pipeline:


(def new-data-transformed
  (-> (mm/transform-pipe
       new-data
       (:pipeline workflow-result)
       (:fitted-ctx workflow-result))
      :metamorph/data))


;; **Transformed new data with cluster assignments:**

(-> new-data-transformed :clustering frequencies)

;; ## Part 10: Advanced Topics

;; ### 10.1 DBSCAN (Density-Based Clustering)
;;
;; DBSCAN can find clusters of arbitrary shape and identify outliers.

;; (def dbscan-pipeline
;;   (mm/pipeline
;;    (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
;;    {:metamorph/id :model}
;;    (ml/model {:model-type :smile.clustering/dbscan
;;               :min-pts 5                ; Minimum points for a cluster
;;               :radius 0.5})))           ; Neighborhood radius

;; (def dbscan-result
;;   (dbscan-pipeline
;;    {:metamorph/data iris-ds
;;     :metamorph/mode :fit}))

^{:kind/md true :kindly/hide-code true}
["**DBSCAN clustering:** Can detect outliers (labeled as cluster -1)"]

;; (def dbscan-clusters
;;   (:cluster-id (:metamorph/data dbscan-result)))

;; (-> (tc/add-column iris-ds :dbscan-cluster dbscan-clusters)
;;     (tc/group-by [:dbscan-cluster])
;;     (tc/aggregate {:count tc/row-count})
;;     (tc/order-by [:dbscan-cluster]))

;; ### 10.2 Different Linkage Methods in Hierarchical Clustering

^{:kind/md true :kindly/hide-code true}
["**Hierarchical clustering linkage methods:**
- `:single` - Minimum distance between clusters (can create long chains)
- `:complete` - Maximum distance between clusters (creates tight clusters)
- `:average` - Average distance between all pairs
- `:ward` - Minimizes within-cluster variance (often best results)"]

;; (defn try-linkage [linkage-method]
;;   (let [pipeline (mm/pipeline
;;                   (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
;;                   {:metamorph/id :model}
;;                   (ml/model {:model-type :smile.clustering/hierarchical
;;                              :k 3
;;                              :linkage linkage-method}))
;;         result (pipeline {:metamorph/data iris-ds
;;                           :metamorph/mode :fit})]
;;     {:linkage linkage-method
;;      :clusters (:cluster-id (:metamorph/data result))}))

;; (def linkage-comparison
;;   (map try-linkage [:single :complete :average :ward]))


;; **Compared 4 different linkage methods**

;; ## Part 11: Best Practices for Unsupervised Learning

^{:kind/md true :kindly/hide-code true}
["### Best Practices

1. **Always scale your features** - Most algorithms are sensitive to feature scales
   - Use standard scaling (mean=0, std=1) for most cases
   - Use min-max scaling when you need specific ranges
   - Use robust scaling when you have outliers

2. **Try multiple algorithms** - Different algorithms work better for different data
   - K-means: Fast, works well with spherical clusters
   - Hierarchical: Good for exploring different granularities
   - DBSCAN: Can find arbitrary shapes and outliers

3. **Validate results** - Without labels, validation requires creativity
   - Visual inspection (especially after PCA to 2D/3D)
   - Domain expertise: Do the clusters make sense?
   - Stability: Do results change much with different random seeds?
   - Multiple metrics: Use several quality measures

4. **Use dimensionality reduction carefully**
   - PCA is great for visualization and noise reduction
   - But it can remove important information
   - Try clustering with and without PCA

5. **Preprocess appropriately for your data type**
   - Numerical: Scaling, outlier handling
   - Categorical: One-hot encoding
   - Text: TF-IDF, embeddings
   - Mixed: Handle each type appropriately

6. **Experiment with hyperparameters**
   - Number of clusters (K)
   - Distance metrics
   - Linkage methods (for hierarchical)
   - PCA components
   - Use the elbow method and silhouette analysis"]

;; ## Summary
;;
;; In this tutorial, we covered:
;;
;; 1. **K-means clustering** - Partitioning data into K groups
;; 2. **Hierarchical clustering** - Building cluster trees
;; 3. **DBSCAN** - Density-based clustering with outlier detection
;; 4. **PCA** - Dimensionality reduction for visualization and preprocessing
;; 5. **Feature scaling** - Standard and min-max scaling
;; 6. **Text processing** - TF-IDF for document clustering
;; 7. **Evaluation** - Methods for assessing cluster quality
;; 8. **Complete workflows** - End-to-end unsupervised learning pipelines
;; 9. **Best practices** - Guidelines for successful unsupervised learning
;;
;; ## Next Steps
;;
;; - Explore other Smile clustering algorithms (X-Means, G-Means)
;; - Try t-SNE or UMAP for non-linear dimensionality reduction
;; - Combine unsupervised and supervised learning (semi-supervised)
;; - Use clustering for feature engineering in supervised tasks
;; - Apply to real-world problems: customer segmentation, anomaly detection, etc.
;;
;; For more information:
;; - [metamorph.ml Documentation](https://github.com/scicloj/metamorph.ml)
;; - [Smile ML Library](https://haifengl.github.io/)
;; - [Scicloj Community](https://scicloj.github.io)


^{:kind/md true :kindly/hide-code true}
["---
**Tutorial complete!** You now have a solid foundation for unsupervised machine learning with metamorph.ml."]

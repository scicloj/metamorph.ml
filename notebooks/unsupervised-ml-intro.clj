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
  (:require [scicloj.clay.v2.api :as clay]
            [tablecloth.api :as tc]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.datatype.functional :as dfn]
            [scicloj.metamorph.core :as mm]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.preprocessing :as prep]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            [scicloj.ml.smile.projections]
            [scicloj.ml.smile.clustering]
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

^:kind/md
["**Iris dataset (unlabeled):**"]

(tc/head iris-ds 5)

^:kind/md
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

^:kind/md
["**Scaled data (first 5 rows):**"]

(tc/head scaled-iris 5)

;; Check that scaling worked (mean ≈ 0, std ≈ 1):

^:kind/md
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
              :clustering-method-args [3 100 1e-4]
              ;; :k 3                    ; Number of clusters
              ;; :max-iter 100            ; Maximum iterations
              ;; :tolerance 1e-4
              }
             
             )))      ; Convergence tolerance

;; Fit the clustering model:

(def kmeans-result
  (kmeans-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

;; Extract the trained model:

(def kmeans-model
  (-> kmeans-result :model :model-data))

^:kind/md
["**K-means clustering complete!**"]

;; ### 1.4 Analyzing Cluster Assignments
;;
;; Get the cluster assignments (which cluster each point belongs to):

(def cluster-assignments
  (-> kmeans-model :clustering))


^:kind/md
(str "**Number of unique clusters found:** "
     (count (distinct cluster-assignments)))

;; Add cluster assignments to the original data:

(def iris-with-clusters
  (tc/add-column iris-ds :cluster cluster-assignments))

^:kind/md
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
   {:metamorph/id :model}
   (ml/model {:model-type :smile.projections/pca-cov 
              :algorithm :pca-cov
              :cnames numeric-cols
              :target-dims 2})))          ; Reduce to 2 dimensions

;; Fit PCA:

(def pca-result
  (pca-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

(-> 
 (mm/transform-pipe iris-ds pca-pipeline pca-result)
 :model
 :model-data
 :dataset
 keys
 )

(def pca-transformed
  (:metamorph/data )
  )

(-> pca-result keys)
^:kind/md
["**PCA transformation complete!**"]

^:kind/md
(str "**Original dimensions:** " (second (ds/shape iris-ds)))

^:kind/md
(str "**Reduced dimensions:** " (second (ds/shape pca-transformed)))

;; View the transformed data:

(tc/head pca-transformed 10)

;; ### 2.2 Explained Variance
;;
;; PCA components capture different amounts of variance. The first component
;; captures the most variance, the second captures the second most, etc.

(def pca-model
  (-> pca-result :model :model-data))

^:kind/md
"The PCA model contains information about explained variance and component loadings."

;; ### 2.3 Combining PCA with Clustering
;;
;; A common pattern is to use PCA for dimensionality reduction, then cluster
;; in the reduced space:

(def pca-kmeans-pipeline
  (mm/pipeline
   ;; Standardize
   (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
   ;; Reduce dimensions
   {:metamorph/id :pca}
   (ml/model {:model-type :smile.manifold/pca
              :dimension 2})
   ;; Cluster in reduced space
   {:metamorph/id :kmeans}
   (ml/model {:model-type :smile.clustering/kmeans
              :k 3
              :max-iter 100})))

(def pca-kmeans-result
  (pca-kmeans-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

^:kind/md
"**PCA + K-means pipeline complete!**"

;; Get cluster assignments from the combined pipeline:

(def pca-clusters
  (:cluster-id (:metamorph/data pca-kmeans-result)))

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

(def hclust-pipeline
  (mm/pipeline
   (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
   {:metamorph/id :model}
   (ml/model {:model-type :smile.clustering/hierarchical
              :k 3                      ; Cut tree to get 3 clusters
              :linkage :complete})))    ; Linkage method: :single, :complete, :average, :ward

(def hclust-result
  (hclust-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

(def hclust-assignments
  (:cluster-id (:metamorph/data hclust-result)))

(def iris-hclust
  (tc/add-column iris-ds :hclust-cluster hclust-assignments))

^:kind/md
"**Hierarchical clustering results:**"

(-> iris-hclust
    (tc/group-by [:hclust-cluster])
    (tc/aggregate {:count tc/row-count
                   :mean-petal-length #(dfn/mean (% :petal-length))})
    (tc/order-by [:hclust-cluster]))

;; ## Part 4: Feature Engineering and Preprocessing
;;
;; Proper preprocessing is crucial for unsupervised learning.

;; ### 4.1 Standard Scaling (Z-score Normalization)

(def std-scaled-ds
  (-> (mm/pipeline
       (ds-mm/std-scale cf/numeric {:mean? true :stddev? true}))
      (apply [{:metamorph/data iris-ds
               :metamorph/mode :fit}])
      :metamorph/data))

^:kind/md
"**Standard scaling:** Transforms features to have mean=0 and std=1"

(ds/descriptive-stats std-scaled-ds [:mean :standard-deviation])

;; ### 4.2 Min-Max Scaling

(def minmax-scaled-ds
  (-> (mm/pipeline
       (ds-mm/min-max-scale cf/numeric {:min -1 :max 1}))
      (apply [{:metamorph/data iris-ds
               :metamorph/mode :fit}])
      :metamorph/data))

^:kind/md
"**Min-max scaling:** Transforms features to a specific range (here: -1 to 1)"

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

^:kind/md
"**Text documents:**"

documents-ds

;; ### 5.1 Tokenization and TF-IDF
;;
;; First, we need to tokenize text and compute TF-IDF features:

(require '[scicloj.metamorph.ml.text :as text])
(require '[clojure.string :as str])

;; Tokenize documents:

(defn simple-tokenize [text]
  (-> text
      str/lower-case
      (str/split #"\s+")))

(def tokenized-docs
  (tc/add-column documents-ds
                 :tokens
                 (map simple-tokenize (:text documents-ds))))

^:kind/md
"**Tokenized documents:**"

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

^:kind/md
"**Tidy text format (sample):**"

(tc/head tidy-docs 10)

;; Compute term frequencies:

(def term-counts
  (-> tidy-docs
      (tc/group-by [:doc-id :token])
      (tc/aggregate {:n tc/row-count})))

^:kind/md
"**Term counts (sample):**"

(tc/head term-counts 10)

;; ### 5.2 Document Similarity
;;
;; After TF-IDF vectorization, we can cluster documents based on their
;; semantic similarity.

^:kind/md
"**Note:** Full TF-IDF clustering requires converting the term-document matrix
to a format suitable for clustering. The `scicloj.metamorph.ml.text` namespace
provides functions for this."

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

^:kind/md
"**Common evaluation approaches:**
- **Elbow method:** Plot WCSS vs. number of clusters, look for the 'elbow'
- **Silhouette analysis:** Compute silhouette score for each point
- **Gap statistic:** Compare within-cluster dispersion to null reference
- **Domain validation:** Check if clusters make sense in your domain"

;; ## Part 7: The Elbow Method for Choosing K
;;
;; The elbow method helps determine the optimal number of clusters.

(defn fit-kmeans-for-k [ds k]
  (let [pipeline (mm/pipeline
                  (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
                  {:metamorph/id :model}
                  (ml/model {:model-type :smile.clustering/kmeans
                             :k k
                             :max-iter 100}))
        result (pipeline {:metamorph/data ds
                         :metamorph/mode :fit})]
    {:k k
     :model (-> result :model :model-data)
     :result result}))

;; Try different values of K:

(def k-values [2 3 4 5 6 7 8])

^:kind/md
"**Testing different values of K...**"

(def elbow-results
  (mapv #(fit-kmeans-for-k iris-ds %) k-values))

^:kind/md
(str "**Tried K values:** " k-values)

^:kind/md
"To find the optimal K, plot WCSS vs K and look for an 'elbow' where the
rate of decrease slows down."

;; ## Part 8: Complete Unsupervised Workflow
;;
;; Here's a complete workflow combining preprocessing, dimensionality
;; reduction, and clustering:

(defn unsupervised-workflow [dataset n-components n-clusters]
  "Complete unsupervised learning workflow"
  (let [;; Build the pipeline
        pipeline (mm/pipeline
                  ;; Step 1: Standardize features
                  (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})

                  ;; Step 2: Dimensionality reduction with PCA
                  {:metamorph/id :pca}
                  (ml/model {:model-type :smile.manifold/pca
                             :dimension n-components})

                  ;; Step 3: Cluster in reduced space
                  {:metamorph/id :kmeans}
                  (ml/model {:model-type :smile.clustering/kmeans
                             :k n-clusters
                             :max-iter 100}))

        ;; Fit the pipeline
        result (pipeline {:metamorph/data dataset
                         :metamorph/mode :fit})

        ;; Extract results
        transformed-data (:metamorph/data result)
        pca-model (-> result :pca :model-data)
        kmeans-model (-> result :kmeans :model-data)
        cluster-assignments (:cluster-id transformed-data)]

    {:pipeline pipeline
     :context result
     :transformed-data transformed-data
     :cluster-assignments cluster-assignments
     :pca-model pca-model
     :kmeans-model kmeans-model
     :add-clusters-to-original
     (fn [original-ds]
       (tc/add-column original-ds :cluster cluster-assignments))}))

;; Use the workflow:

(def workflow-result
  (unsupervised-workflow iris-ds 2 3))

^:kind/md
"**Complete workflow executed!**"

;; Add clusters to original data:

(def iris-final
  ((:add-clusters-to-original workflow-result) iris-ds))

^:kind/md
"**Final clustered data:**"

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
  (tc/sample iris-ds 10 {:seed 123}))

^:kind/md
"**New data to transform:**"

(tc/head new-data 5)

;; Apply the trained pipeline:

(def new-data-transformed
  (-> ((:pipeline workflow-result)
       (merge (:context workflow-result)
              {:metamorph/data new-data
               :metamorph/mode :transform}))
      :metamorph/data))

^:kind/md
"**Transformed new data with cluster assignments:**"

(tc/select-columns new-data-transformed [:cluster-id])

;; ## Part 10: Advanced Topics

;; ### 10.1 DBSCAN (Density-Based Clustering)
;;
;; DBSCAN can find clusters of arbitrary shape and identify outliers.

(def dbscan-pipeline
  (mm/pipeline
   (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
   {:metamorph/id :model}
   (ml/model {:model-type :smile.clustering/dbscan
              :min-pts 5                ; Minimum points for a cluster
              :radius 0.5})))           ; Neighborhood radius

(def dbscan-result
  (dbscan-pipeline
   {:metamorph/data iris-ds
    :metamorph/mode :fit}))

^:kind/md
"**DBSCAN clustering:** Can detect outliers (labeled as cluster -1)"

(def dbscan-clusters
  (:cluster-id (:metamorph/data dbscan-result)))

(-> (tc/add-column iris-ds :dbscan-cluster dbscan-clusters)
    (tc/group-by [:dbscan-cluster])
    (tc/aggregate {:count tc/row-count})
    (tc/order-by [:dbscan-cluster]))

;; ### 10.2 Different Linkage Methods in Hierarchical Clustering

^:kind/md
"**Hierarchical clustering linkage methods:**
- `:single` - Minimum distance between clusters (can create long chains)
- `:complete` - Maximum distance between clusters (creates tight clusters)
- `:average` - Average distance between all pairs
- `:ward` - Minimizes within-cluster variance (often best results)"

(defn try-linkage [linkage-method]
  (let [pipeline (mm/pipeline
                  (ds-mm/std-scale cf/numeric {:mean? true :stddev? true})
                  {:metamorph/id :model}
                  (ml/model {:model-type :smile.clustering/hierarchical
                             :k 3
                             :linkage linkage-method}))
        result (pipeline {:metamorph/data iris-ds
                         :metamorph/mode :fit})]
    {:linkage linkage-method
     :clusters (:cluster-id (:metamorph/data result))}))

(def linkage-comparison
  (map try-linkage [:single :complete :average :ward]))

^:kind/md
"**Compared 4 different linkage methods**"

;; ## Part 11: Best Practices for Unsupervised Learning

^:kind/md
"### Best Practices

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
   - Use the elbow method and silhouette analysis"

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

^:kind/md
"---
**Tutorial complete!** You now have a solid foundation for unsupervised machine learning with metamorph.ml."

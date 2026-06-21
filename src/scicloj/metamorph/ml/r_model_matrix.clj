(ns scicloj.metamorph.ml.r-model-matrix
  "R-style formula-based feature engineering and linear regression.

   This namespace provides tools to leverage R's powerful formula syntax for
   feature engineering and linear modeling within Clojure. R formulas enable
   expressive specification of interactions, transformations, and categorical
   expansions without manual column manipulation.

   Key Functions:

   - `r-model-matrix`: Convert dataset + R formula to design matrix
   - `lm`: Simplified linear regression using R formulas

   Implementation Backends:
   The namespace supports multiple R execution backends:

   - `:ocpu`    Remote R via OpenCPU (cloud.opencpu.org) - no local R needed
   - `:renjin` Java-based R implementation (https://renjin.org/)
   - `:clojisr` Local R via clojisr (requires R installation)

   Model Matrix Capabilities:
   R formulas handle:

   - Basic features: `y ~ x1 + x2`
   - Interactions: `y ~ x1 * x2` (expands to x1 + x2 + x1:x2)
   - Polynomial terms: `y ~ x + I(x^2)`
   - Categorical encoding: Automatic dummy variable creation
   - Intercept control: `y ~ x - 1` (remove intercept)
   - Exclusions: `y ~ . - x3` (all columns except x3)

   Linear Regression (lm):
   Combines formula-based feature engineering with OLS regression training.
   Returns a ready-to-use trained model for predictions.

   
   Notes:

   - OpenCPU backend is convenient but requires internet connectivity
   - Renjin is standalone but may have some R incompatibilities
   - clojisr requires a local R installation but offers full R compatibility
   - Returned model matrices exclude row names and intercept columns by default

   See also: [[scicloj.metamorph.ml.design-matrix]] for Clojure-native feature engineering"
  (:require
   [cheshire.core :as json]
   [clojure.string :as str]
   [metadoc.examples :refer [example-session]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.rdatasets :as rdatasets]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.print :as print]
   [tech.v3.datatype :as dt]
   [scicloj.metamorph.ml.impl.r :as impl-r]
   ))


 
 
 
 (defn r-model-matrix
   "Compute a model matrix from a dataset and an R-style formula.

   Parameters:
    
   - `ds`         A tech.ml.dataset dataset representing the input data.
   - `r-formula`  A string containing the R formula to use for model matrix construction. The formua is interpreted by R itself, so should be full compatible
   - `impl`       An implementation keyword, either

       - `:ocpu`    Uses an online service https://www.opencpu.org/api.html (server: cloud.opencpu.org)
       - `:renjine` Uses https://renjin.org/   
       - `:clojisr` Uses https://github.com/scicloj/clojisr, which requires a local R installation 
    
   Each implementation requires dependencies to be added:
    
   - `:ocpu` :  [opencpu-clj/opencpu-clj \"0.3.1\"] 
   - `:renjin` : [org.renjin/renjin-script-engine \"3.5-beta76\"]
   - `:clojisr` : [scicloj/clojisr \"1.1.0\"]



   Returns a dataset containing the constructed design matrix.
   If `ds` contains `target` columns, they are added to the returned dataset.
    
   Dispatches to the appropriate backend implementation.

    
   Returns a map with
   
   - `:model-matrix-dataset` having the TMD containing the design matrix specified by `r-formula`
   - `:attributes` the (R) attributes of the model.matrix object
    
    "
   {:metadoc/examples
    [(example-session "Call with renjin backend"
                      (require '[scicloj.metamorph.ml.rdatasets :as rdatasets])
                      (->
                       (rdatasets/datasets-mtcars)
                       (r-model-matrix "mpg ~ as.factor(cyl) * hp + disp" :renjin)
                       :model-matrix-dataset
                       (print/dataset->str)))
     (example-session "Call with ocpu backend"
                      (require '[scicloj.metamorph.ml.rdatasets :as rdatasets])
                      (->
                       (rdatasets/datasets-iris)
                       (ds/remove-column :rownames)
                       (ds-mod/set-inference-target [:species])
                       (r-model-matrix "species ~ ." :ocpu)
                       :model-matrix-dataset
                       (print/dataset->str)))]}
   
   [dataset r-formula impl]

   (let [target-ds (cf/target dataset)

         result
         (case impl
           :ocpu (impl-r/model-matrix--ocpu dataset r-formula)
           :renjin (impl-r/model-matrix--renjine dataset r-formula)
           :clojisr (impl-r/model-matrix--clojisr dataset r-formula))
         model-matrix-dataset (:model-matrix-dataset result)]

     (assoc result
            :model-matrix-dataset (merge model-matrix-dataset target-ds))))
 
(defn lm
  "Train a linear model using an R-style formula.

   This function combines R formula-based feature engineering with ordinary least
   squares (OLS) regression. It creates a design matrix from the input dataset using
   the specified R formula, then trains a linear model on the resulting features.

   Parameters:

   - `ds`             A tech.ml.dataset dataset containing the input data with all
                      variables referenced in the formula and target variable.
   - `formula`        A string containing the R formula (e.g., \"y ~ x1 + x2 * x3\").
                      The formula is interpreted by the R backend.
   - `target-var`     A keyword or string naming the target variable for regression.
                      This variable must be present in the input dataset.
   - `formula-impl`   An implementation keyword for formula evaluation:

     - `:ocpu`    Uses OpenCPU (cloud.opencpu.org), no local R needed
     - `:renjin` Uses Renjin, a Java implementation of R
     - `:clojisr` Uses clojisr with local R installation

   Requires setup of dependencies of teh engine, see: `r-model-matrix`
    
   Returns:
   A trained linear model (OLS from fastmath) ready for predictions. The model
   excludes the intercept column and row names from the design matrix by default.

   Example:
   ```
   (lm iris-data \"Sepal.Width ~ Sepal.Length + Petal.Length\" :Sepal.Width :renjin)
   ```
   "
  [ds formula target-var formula-impl]
  (-> ds
      (r-model-matrix formula formula-impl)
      :model-matrix-dataset
      (tc/drop-columns [:$row.names "(Intercept)" "X.Intercept."])
      (tc/add-column target-var (get ds target-var))
      (ds-mod/set-inference-target [target-var])
      (ml/train {:model-type :fastmath/ols})))




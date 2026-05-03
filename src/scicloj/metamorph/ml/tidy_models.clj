(ns scicloj.metamorph.ml.tidy-models
  "Model output standardization and validation following tidymodels conventions.

   This namespace implements the tidymodels philosophy (inspired by R's tidymodels/broom
   packages) for standardized, machine-readable model outputs. All model outputs conform
   to consistent schemas defined in canonical column specification files.

   Three Core Output Functions:

   **glance**: One-row model summary
   - High-level goodness-of-fit statistics
   - Examples: R², AIC, BIC, log-likelihood, F-statistic, p-value
   - Use case: Quick model performance overview

   **tidy**: One-row-per-component output
   - Component-level details (e.g., one row per coefficient)
   - Examples: term, estimate, std.error, statistic, p.value
   - Use case: Detailed model inspection and reporting

   **augment**: One-row-per-observation output
   - Adds model predictions/residuals to original data
   - Original columns plus: .fitted, .resid, .hat, .sigma, .cooksd
   - Use case: Diagnostics and visualization of predictions

   Validation and Schema Management:

   - `allowed-tidy-columns`: Canonical list of valid tidy column names
   - `allowed-glance-columns`: Canonical list of valid glance column names
   - `allowed-augment-columns`: Canonical list of valid augment column names
   - `validate-tidy-ds`: Validates dataset conforms to tidy standard
   - `validate-glance-ds`: Validates dataset conforms to glance standard
   - `validate-augment-ds`: Validates dataset conforms to augment standard

   Schemas are maintained in GitHub repository (resources/*.edn):
   - columms-tidy.edn
   - columms-glance.edn
   - columms-augment.edn

   Control Validation:
   The `*validate-tidy-fns*` dynamic variable controls strict validation:
   - `true` (default): Raises exception on invalid columns
   - `false`: Silently allows any columns

   
   Integration:
   Model implementations use these validators in their tidy-fn/glance-fn/augment-fn
   to ensure outputs conform to standardized schemas for consistency across models.

   See also: `scicloj.metamorph.ml` for training and prediction,
   `scicloj.metamorph.ml.regression` and `scicloj.metamorph.ml.classification`
   for specific model implementations"
  (:require
   [clojure.edn :as edn]
   [clojure.set :as set]
   [tech.v3.dataset :as ds]))

(def ^:dynamic
 ^{:doc "Controls if the result columns of the tidy fns of a model
(glance-fn, tidy-fn, augment-fn is validated against these base
https://github.com/scicloj/metamorph.ml/tree/main/resources/*.edn
 and if on violation they fail."
   :added "1.0"}
 *validate-tidy-fns* true)

(defn allowed-glance-columns
  "Returns the allowed column names for glance datasets.

  Fetches the canonical list from the metamorph.ml GitHub repository
  (columms-glance.edn). Glance datasets provide one-row model summaries with
  goodness-of-fit measures.

  See also: `scicloj.metamorph.ml/glance`, `validate-glance-ds`"
  []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-glance.edn"))))

(defn allowed-tidy-columns
  "Returns the allowed column names for tidy datasets.

  Fetches the canonical list from the metamorph.ml GitHub repository
  (columms-tidy.edn). Tidy datasets provide one-row-per-component model
  summaries (e.g., coefficients, statistics).

  See also: `scicloj.metamorph.ml/tidy`, `validate-tidy-ds`"
  []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-tidy.edn"))))

(defn allowed-augment-columns
  "Returns the allowed column names for augment datasets.

  Fetches the canonical list from the metamorph.ml GitHub repository
  (columms-augment.edn). Augment datasets add observation-level model outputs
  (predictions, residuals, influence) to the original data.

  See also: `scicloj.metamorph.ml/augment`, `validate-augment-ds`"
  []
  (keys
   (edn/read-string (slurp "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/resources/columms-augment.edn"))))

(defn _get-allowed-keys
  "Internal function that fetches all allowed columns for tidy model functions.

  Returns a map with keys `:glance`, `:tidy`, and `:augment`, each containing
  a sequence of allowed column names.

  Not memoized. Use `get-allowed-keys` (memoized version) for repeated access."
  []
  {:glance (allowed-glance-columns)
   :tidy (allowed-tidy-columns)
   :augment (allowed-augment-columns)})

(def get-allowed-keys
  "Memoized function that returns allowed columns for all tidy model functions.

  Returns a map with keys:

  * `:glance` - Allowed columns for model summary
  * `:tidy` - Allowed columns for component-wise output
  * `:augment` - Allowed columns for observation-level output

  Results are cached for efficient repeated access. Fetches data from GitHub
  on first call.

  See also: `validate-tidy-ds`, `validate-glance-ds`, `validate-augment-ds`"
  (memoize _get-allowed-keys))



(defn- validate-ds [ds allowed-columns fn-name]
  (if (true? *validate-tidy-fns*)
    (let [
          invalid-keys
          (set/difference
           (into #{} (ds/column-names ds))
           (into #{} allowed-columns))]
      (if (empty? invalid-keys)
        ds
        (throw (Exception. (format "invalid keys from %s: %s" fn-name invalid-keys)))))
    ds))

(defn validate-tidy-ds
  "Validates that a dataset conforms to the tidy model standard.

  `ds` - Dataset to validate

  Returns the dataset if valid. Throws an exception if the dataset contains
  column names not in the allowed tidy columns list (when `*validate-tidy-fns*`
  is true).

  Used internally by model tidy-fn implementations.

  See also: `scicloj.metamorph.ml/tidy`, `allowed-tidy-columns`"
  [ds]
  (validate-ds ds (:tidy (get-allowed-keys))  "tidy-fn"))

(defn validate-glance-ds
  "Validates that a dataset conforms to the glance model standard.

  `ds` - Dataset to validate

  Returns the dataset if valid. Throws an exception if the dataset contains
  column names not in the allowed glance columns list (when `*validate-tidy-fns*`
  is true).

  Used internally by model glance-fn implementations.

  See also: `scicloj.metamorph.ml/glance`, `allowed-glance-columns`"
  [ds]
  (validate-ds ds (:glance (get-allowed-keys)) "glance-fn"))

(defn validate-augment-ds
  "Validates that a dataset conforms to the augment model standard.

  `ds` - Dataset to validate
  `data` - Original dataset (for allowed column name extraction)

  Returns the dataset if valid. Throws an exception if the dataset contains
  column names not in the allowed augment columns list or the original data
  columns (when `*validate-tidy-fns*` is true).

  Used internally by model augment-fn implementations.

  See also: `scicloj.metamorph.ml/augment`, `allowed-augment-columns`"
  [ds data]
  (validate-ds
   ds
   (concat (:augment (get-allowed-keys)) (ds/column-names data))
   "augment-fn"))

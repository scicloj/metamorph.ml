(ns scicloj.metamorph.ml.gridsearch
  "Gridsearching as defined by create a map with gridsearch definitions
  for its values and then gridsearching which produces a sequence of full
  defined maps.


  The initial default implementation uses the sobol sequence."
  (:require [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.casting :as casting])
  (:import [org.apache.commons.math3.random SobolSequenceGenerator]))


(defn- sobol-seq
  "Given a set dimension count and optional start index, return an infinite sequence of
  points in the unit hypercube with coordinates in range [0-1].
  :start-index - starting index.  Constant time to start at any index.
  Returns sequence of double arrays of length n-dim."
  [long-space-projections n-total-elements gridsearch-start-index]
  (let [gen (SobolSequenceGenerator. (count long-space-projections))]
    (when gridsearch-start-index
      (.skipTo gen gridsearch-start-index))
    (->> (repeatedly #(.nextVector gen))
         (map (fn [dvec]
                (mapv (fn [dval proj]
                        (proj dval))
                      dvec
                      long-space-projections)))
         ;;remove ones we have already tried before.
         (distinct)
         ;;Meaning we have only these many steps left
         (take n-total-elements))))


(defmulti ^:private project
  "Project a long into the gridsearched element space."
  (fn [gridsearch-element value]
    (::type gridsearch-element)))


(defmethod project :default
  [gridsearch-element value]
  (errors/throwf "Failed to find projection for gridsearch element %s" gridsearch-element))


(defn linear
  "Creates a linear grid search definition for hyperparameter tuning.

  `start` - Starting value of the range
  `end` - Ending value of the range
  `n-steps` - Number of evenly-spaced steps (default: 100)
  `res-dtype-or-space` - Either a datatype keyword (`:float64`, `:int32`, etc.)
                         or a vector of categorical values (default: `:float64`)

  Returns a grid search definition map that can be used with `sobol-gridsearch`.

  When `res-dtype-or-space` is a vector, values are mapped to categorical options
  by index. For example, `[:small :medium :large]` with 3 steps generates those
  three categorical values.

  See also: `categorical`, `sobol-gridsearch`"
  ([start end n-steps res-dtype-or-space]
   (let [start (double start)
         end (double end)
         n-steps (long n-steps)]
     {::type :linear
      :start start
      :end end
      :n-steps n-steps
      :result-space res-dtype-or-space}))
  ([start end n-steps]
   (linear start end n-steps :float64))
  ([start end]
   (linear start end 100 :float64)))


(defn categorical
  "Creates a categorical grid search definition from a vector of values.

  `value-vec` - Vector of categorical values to search over

  Returns a grid search definition map that will enumerate all values in the
  vector during grid search. This is a convenience wrapper around `linear` for
  categorical parameters.

  Example: `(categorical [:adam :sgd :rmsprop])` creates a definition that will
  try all three optimizers.

  See also: `linear`, `sobol-gridsearch`"
  [value-vec]
  (let [n-elems (count value-vec)]
    (linear 0 (dec n-elems) n-elems value-vec)))


(defmethod project :linear
  [{:keys [start end n-steps result-space]} value]
  (let [value (long value)
        start (double start)
        end (double end)
        rel-value (/ (double value) (double (dec n-steps)))
        range (- end start)
        rel-value (* rel-value range)
        final-value (+ rel-value start)]
    (cond
      (sequential? result-space)
      (result-space (Math/round final-value))
      (keyword? result-space)
      (casting/cast final-value result-space))))


(defn- map->axis
  ([data-map path axis]
   (reduce (fn [axis [k v]]
             (cond
               (::type v)
               (let [{:keys [n-steps]} v
                     n-steps (dec (double n-steps))]
                 (conj axis {:axis (conj path k)
                             :element v
                             :lspace-proj (fn ^long [^double sobol-val]
                                            (Math/round (* sobol-val n-steps)))}))
               (map? v)
               (map->axis v (conj path k) axis)
               :else
               axis))
           axis
           data-map))
  ([data-map]
   (map->axis data-map [] [])))


(defn sobol-gridsearch
  "Given an map of key->values where some of the values are gridsearch definitions
  produce a sequence of fully defined maps.


```clojure
user> (require '[tech.v3.ml.gridsearch :as ml-gs])
nil
user> (def opt-map  {:a (ml-gs/categorical [:a :b :c])
                     :b (ml-gs/linear 0.01 1 10)
                     :c :not-searched})
user> opt-map
{:a
 {:tech.v3.ml.gridsearch/type :linear,
  :start 0.0,
  :end 2.0,
  :n-steps 3,
  :result-space [:a :b :c]}
  ...

user> (ml-gs/sobol-gridsearch opt-map)
({:a :b, :b 0.56, :c :not-searched}
 {:a :c, :b 0.22999999999999998, :c :not-searched}
 {:a :b, :b 0.78, :c :not-searched}
...
```
  "
  ([opt-map start-idx]
   (let [axis (map->axis opt-map)]
     (if (seq axis)
       (let [lspace-projections (mapv :lspace-proj axis)
             total-steps (long (apply * 1.0 (map #(get-in % [:element :n-steps])
                                                 axis)))]
         (->> (sobol-seq lspace-projections total-steps start-idx)
              (map (fn [seq-data]
                     (reduce (fn [opt-map [seq-elem {:keys [axis element]}]]
                               (assoc-in opt-map axis (project element seq-elem)))
                             opt-map
                             (map vector seq-data axis))))))
       [opt-map])))
  ([opt-map]
   (sobol-gridsearch opt-map 0)))

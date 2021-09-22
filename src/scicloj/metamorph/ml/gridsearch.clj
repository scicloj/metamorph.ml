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
  "Create a gridsearch definition which does a linear search.

  * res-dtype-or-space map be either a datatype keyword or a vector
    of categorical values."
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
  "Given a vector a categorical values create a gridsearch definition."
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
  ([data path axis index]
   (println "data:" data)
   (println "path:" path)
   (println "axis:" axis)
   (def data data)
   (def path path)
   (def axis axis)

   (reduce (fn [something-1 something-2]
             (println "something-1:" something-1)
             (println "something-2:" something-2)
             (let [k nil v nil]
               ;; (def k k)
               ;; (def v v)
               (def axis axis)
               (def index index)
               (println "axis: " axis)
               (println :k k :v v)
               (cond
                 (::type v)
                 (let [{:keys [start end n-steps]} v
                       start (double start)
                       end (double end)
                       n-steps (dec (double n-steps))]
                   (conj axis {:axis (conj path k)
                               :element v
                               :lspace-proj (fn ^long [^double sobol-val]
                                              (Math/round (* sobol-val n-steps)))}))
                 (map? v)
                 (map->axis v (conj path k) axis index)

                 (sequential? v)
                 (map
                  #(let [{:keys [start end n-steps]} %
                         start (double start)
                         end (double end)
                         n-steps (dec (double n-steps))]
                     (conj axis {:axis (conj  (conj path k) index)
                                 :element %
                                 :lspace-proj (fn ^long [^double sobol-val]
                                                (Math/round (* sobol-val n-steps)))}))

                  v)
                 ;; (map->axis v k axis (inc index))

                 :else
                 axis)))
           axis
           data-map))
  ([data-map]
   (map->axis data-map [] [] 0)))


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
   (def opt-map opt-map)
   (def index index)
   (let [axiss (map->axis opt-map)]
     (def axiss axiss)
     (if (seq axiss)
       (mapv
        (fn [axis]
          (let [lspace-projections (mapv :lspace-proj axis)
                total-steps (long (apply * 1.0 (map #(get-in % [:element :n-steps])
                                                    axis)))]
            (->> (sobol-seq lspace-projections total-steps start-idx)
                 (map (fn [seq-data]
                        (reduce (fn [opt-map [seq-elem {:keys [axis element]}]]
                                  (assoc-in opt-map axis (project element seq-elem)))
                                opt-map
                                (map vector seq-data axis)))))))
        axiss)
       [opt-map])))
  ([opt-map]
   (sobol-gridsearch opt-map 0)))


(comment
  (def opt-map)

  (map->axis {:a [(categorical [:a :b])]}))

(map->axis {:a (categorical [:a :b])})
(sobol-gridsearch {:a (categorical [:a :b])})



(sobol-gridsearch {:a [(categorical [:a :b])]})

(map->axis {});; => []


(map->axis {:a 1});; => []


(map->axis {:a {:a 1}})

(map->axis {:a (categorical [:a])})
;; => [{:axis [:a],
;;      :element
;;      {:scicloj.metamorph.ml.gridsearch/type :linear,
;;       :start 0.0,
;;       :end 0.0,
;;       :n-steps 1,
;;       :result-space [:a]},
;;      :lspace-proj
;;      #function[scicloj.metamorph.ml.gridsearch/map->axis/fn--12908/fn--12913]}]
;;
;;
(map->axis {:a (categorical [:a :b])})
;; => [{:axis [:a],
;;      :element
;;      {:scicloj.metamorph.ml.gridsearch/type :linear,
;;       :start 0.0,
;;       :end 1.0,
;;       :n-steps 2,
;;       :result-space [:a :b]},
;;      :lspace-proj
;;      #function[scicloj.metamorph.ml.gridsearch/map->axis/fn--12908/fn--12913]}]



(map->axis {:a [(categorical [:a :b])]})
;; => ([{:axis [:a 0],
;;       :element
;;       {:scicloj.metamorph.ml.gridsearch/type :linear,
;;        :start 0.0,
;;        :end 1.0,
;;        :n-steps 2,
;;        :result-space [:a :b]},
;;       :lspace-proj
;;       #function[scicloj.metamorph.ml.gridsearch/map->axis/fn--13231/fn--13238/fn--13240]}])
;;
(map->axis {:a [(categorical [:a :b])]})

(sobol-gridsearch {:a [(categorical [:a :b :c])]})
(map->axis [:a :b :c])

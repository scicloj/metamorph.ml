(ns scicloj.metamorph.ml.dataset-metamorph
  (:refer-clojure :exclude [boolean])
  (:require [scicloj.metamorph.ml.dataset]
            [tech.v3.protocols.dataset :as prot])

  )

(defn dataset?
  "Is `ds` a `dataset` type?"
  [ds]
  (satisfies? prot/PColumnarDataset ds))

(defmacro build-pipelined-function
  [f m]
  (let [args (map (comp vec rest) (:arglists m))
        doc-string (:doc m)]
    `(defn ~(symbol (name f)) {:doc ~doc-string}
       ~@(for [arg args
               :let [narg (mapv #(if (map? %) 'options %) arg)
                     [a & r] (split-with (partial not= '&) narg)]]
           (list narg `(fn [ds#]
                         (let [ctx# (if (dataset? ds#)
                                      {:metamorph/data ds#} ds#)]
                           (assoc ctx# :metamorph/data (apply ~f (ctx# :metamorph/data) ~@a ~(rest r))))))))))

(def ^:private excludes '#{})

(defmacro process-all-api-symbols
  []
  (let [ps (ns-publics 'scicloj.metamorph.ml.dataset)]
    `(do ~@(for [[f v] ps
                 :when (not (excludes f))
                 :let [m (meta v)
                       f (symbol "scicloj.metamorph.ml.dataset" (name f))]]
             `(build-pipelined-function ~f ~m)))))

(process-all-api-symbols)


(ns scicloj.metamorph.ml.preprocessing
  (:require
   [tech.v3.dataset.math :as std-math]
   [tech.v3.dataset :as ds]

   ))

(defn std-scale [col-seq]
  (fn [{:metamorph/keys [data id mode] :as ctx}]
    (case mode
      :fit
      (let [ds (ds/select-columns data col-seq)
            fit-std-xform (std-math/fit-std-scale ds)]
        (assoc ctx
               id {:fit-std-xform fit-std-xform}
               :metamorph/data (merge data (std-math/transform-std-scale ds fit-std-xform))))
      :transform
      (assoc ctx :metamorph/data
             (merge data
                    (-> (ds/select-columns data col-seq)
                        (std-math/transform-std-scale
                         (get-in ctx [id :fit-std-xform]))))))))

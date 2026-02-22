(ns scicloj.metamorph.ml.cache
  (:require [scicloj.metamorph.ml :as ml]
            [clojure.java.io :as io]
            [taoensso.nippy :as nippy]))


(defn enable-atom-cache!
  "Enables the caching of train/predict calls in an atom.
   
   `cache-atom`: Clojure atom used for caching.
   
   "
  [cache-atom]

  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key] (get @cache-atom key))
                                  :set-fn (fn [key value] (swap! cache-atom assoc key value))}))

(defn enable-disk-cache!
  "Enables the caching of train/predict calls in an directory on disk.
     
     `cacche-dir`: Directory used for caching.
     
     "
  [cache-dir]

  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key]
                                            (let [f (format "%s/%s.nippy" cache-dir key)]
                                              (when (.exists  (io/file f))
                                                (nippy/thaw-from-file f))))
                                  :set-fn (fn [key value]
                                            (nippy/freeze-to-file
                                             (format "%s/%s.nippy" cache-dir key)
                                             value))}))

(defn disable-cache! []
  (reset! ml/train-predict-cache {:use-cache false
                                  :get-fn nil
                                  :set-fn nil}))

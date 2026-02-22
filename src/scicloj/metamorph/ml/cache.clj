(ns scicloj.metamorph.ml.cache
  (:require [scicloj.metamorph.ml :as ml]
            [clojure.java.io :as io]
            [taoensso.nippy :as nippy]))

(defn- macro-available? [sym]
  (try (-> sym requiring-resolve meta :macro) (catch Exception _ false)))

(defmacro when-requiring-resolve-macro
  [sym & body]
  (when (macro-available? sym) `(do ~@body)))


(defn enable-redis-cache!
  "Enables the caching of train/predict calls in an redis server
   using [carmine](https://github.com/taoensso/carmine) library
   
   `wcar-opts`: Clojure atom used for caching.

   'com.taoensso/carmine' needed to be added as depenency
   
   "
  [wcar-opts]
  (assert (macro-available? 'taoensso.carmine/wcar) "please add 'com.taoensso/carmine' to classpath")
  (reset! ml/train-predict-cache
          {:use-cache true
           :get-fn (fn [key]
                     (when-requiring-resolve-macro
                      taoensso.carmine/wcar
                      (taoensso.carmine/wcar wcar-opts
                                             (taoensso.carmine/get key))))
           :set-fn (fn [key value]
                     (when-requiring-resolve-macro
                      taoensso.carmine/wcar
                      (taoensso.carmine/wcar wcar-opts
                                             (taoensso.carmine/set key value))))}))


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
                                            (let [file-name (format "%s/%s.nippy" cache-dir key)]
                                              (when (.exists  (io/file file-name))
                                                (nippy/thaw-from-file file-name))))
                                  :set-fn (fn [key value]
                                            (nippy/freeze-to-file
                                             (format "%s/%s.nippy" cache-dir key)
                                             value))}))

(defn disable-cache! []
  (reset! ml/train-predict-cache {:use-cache false
                                  :get-fn nil
                                  :set-fn nil}))

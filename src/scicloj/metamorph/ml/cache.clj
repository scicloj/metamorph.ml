(ns scicloj.metamorph.ml.cache
  (:require [scicloj.metamorph.ml :as ml]
            [clojure.java.io :as io]
            [taoensso.carmine :as car]
            [taoensso.nippy :as nippy]
            )
  )

(defn enable-redis-cache! [wcar-opts]
  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key] (car/wcar wcar-opts (car/get key)))
                                  :set-fn (fn [key value] (car/wcar wcar-opts (car/set key value)))}))

(defn enable-atom-cache! [cache-map]
  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key] (get @cache-map key))
                                  :set-fn (fn [key value] (swap! cache-map assoc key value))}))

(defn enable-disk-cache! [cache-dir]
  (reset! ml/train-predict-cache {:use-cache true
                                  :get-fn (fn [key]
                                            (let [f (format "/tmp/cache/%s.nippy" key)]
                                              (when (.exists  (io/file f))
                                                (nippy/thaw-from-file f))))
                                  :set-fn (fn [key value]
                                            (nippy/freeze-to-file
                                             (format "/tmp/cache/%s.nippy" key)
                                             value))}))

(defn disable-cache! []
  (reset! ml/train-predict-cache {:use-cache false}))

(ns scicloj.metamorph.ml.tracking
  (:require [clojure.java.io :as io]
            [clojure.string  :as str])
  (:import [org.mlflow.tracking MlflowClient MlflowContext]))

(def mlflow (MlflowContext. "http://localhost:5000"))

(def run  (.startRun mlflow "test-run"))
(.logParam run "alpha"  "0.5")
(.logMetric run "MSE" 0.0)

;; (def uri
;;   (->
;;    (io/file (.getArtifactUri run))
;;    str
;;    (str/split #":")
;;    second))

;; (def artifact-root (str "mlruns/" uri))

;; (io/copy (io/file "deps.edn") (io/file (str artifact-root "/deps.edn")))

(.logArtifact run (.toPath (io/file "features.txt")))
(.logArtifact run (.toPath (io/file "deps.edn")))
(.endRun run)

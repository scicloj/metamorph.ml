(ns scicloj.metamorph.cache-test
  (:require  [clojure.test :as t]
             [scicloj.metamorph.ml.cache :as mlcache]
             [scicloj.metamorph.ml.toydata :as data]
             [tech.v3.dataset :as ds]
             [taoensso.nippy :as nippy])
  (:import [java.nio.file Files]
           [java.nio.file.attribute FileAttribute]))


(t/deftest roundtrip
  (let [
        temp-dir (str (Files/createTempDirectory "test" (make-array FileAttribute 0)))
        fs-map (mlcache/fs-persisted-map-factory temp-dir)
        _ (assoc fs-map "xxxx" "hello")]
    (t/is (= ["xxxx"] (keys fs-map)))
    (t/is (= "hello" (get fs-map "xxxx")))
    (t/is (= "hello" (nippy/thaw-from-file (str temp-dir "/xxxx.nippy"))))))

(def iris (data/iris-ds))

(t/deftest roundtrip-iris
  (let [
        temp-dir (str (Files/createTempDirectory "test" (make-array FileAttribute 0)))
        fs-map (mlcache/fs-persisted-map-factory temp-dir)
        _ (assoc fs-map "iris" iris)]

    (t/is (= (ds/rows iris)
             (ds/rows (get fs-map "iris"))))
    (t/is (= (map meta (vals iris))
             (map meta (vals (get fs-map "iris")))))))

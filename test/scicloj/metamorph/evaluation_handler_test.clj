(ns scicloj.metamorph.evaluation-handler-test
  (:require  [clojure.test :refer [is deftest]]
             [scicloj.metamorph.ml-test]
             [scicloj.metamorph.ml.evaluation-handler :refer [get-source-information qualify-keywords]]))

(deftest qualify-keywords-test
  (is (= [[:scicloj.metamorph.evaluation-handler-test/do-xxx]
          [:tech.v3.dataset.metamorph/set-inference-target [:species]]]
         (qualify-keywords [[::do-xxx]
                            [:ds-mm/set-inference-target [:species]]] (find-ns 'scicloj.metamorph.ml-test)))))



(deftest source-info-test []
  (let [source-info
        (get-source-information [[:scicloj.metamorph.ml-test/do-xxx]
                                 [:tech.v3.dataset.metamorph/set-inference-target [:species]]]
                                (find-ns 'scicloj.metamorph.ml-test)
                                "/home/carsten/Dropbox/sources/metamorph.ml/test/scicloj/metamorph/ml_test.clj")]

    (def source-info source-info)
    (is (=  "(defn do-xxx [col] col)\n"
            (-> source-info :fn-sources (get 'scicloj.metamorph.ml-test/do-xxx) :code-local-source)))))


(get-source-information [[:tech.v3.dataset.metamorph/set-inference-target]]
                        (find-ns 'scicloj.metamorph.ml-test)
                        "/home/carsten/Dropbox/sources/metamorph.ml/test/scicloj/metamorph/ml_test.clj")

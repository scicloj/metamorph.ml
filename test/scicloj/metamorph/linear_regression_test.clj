(ns scicloj.metamorph.linear-regression-test
  (:require
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.toydata :as data]
   [tech.v3.dataset :as ds]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.ml.regression]
   [tech.v3.dataset.modelling :as ds-mod]))


(deftest linear-regression-mtcars
  (let [ds
        (->
         (data/mtcars-ds)
         (ds/drop-columns [:model])
         (ds-mod/set-inference-target :mpg))

        model (ml/train ds {:model-type :fastmath/ols})

        glance (-> (ml/glance model) (ds/rows :as-map) first)
        tidy (ml/tidy model)
        augment (ml/augment model ds)]


    (is (=   0.8066423189909859 (-> glance :adj.r.squared)))
    (is (=   0.8690157644777647   (:r.squared glance)))
    (is (=   147.49443001665065 (-> glance :rss)))
    (is (=   1126.0471874999998 (-> glance :totss)))
    (is (=   161.70981043447966 (-> glance :aic)))
    (is (=   177.83290536527664 (-> glance :bic)))
    (is (=   -69.85490521723983 (-> glance :log-lik)))
    (is (=    97.85527574833492 (:mse glance)))
    (is (=     3.7931521057466E-7 (:p.value glance)))
    (is (=    10 (:df glance)))
    (is (=    32 (:n glance)))
    (is (=    21 (:df.residual glance)))
    (is (=    13.932463690208833 (:statistic glance)))


         
    (is (= [:mpg :cyl :disp :hp :drat :wt :qsec :vs :am :gear :carb] (-> tidy :term)))
    (is (= [12.303374155996154
            -0.11144047788686227
            0.013335239913341098
            -0.021482118989136358
            0.7871109722361158
            -3.715303928327478
            0.8210407496746307
            0.3177628141854153
            2.520226887208426
            0.6554130170817919
            -0.1994192548562663]
           (-> tidy :estimate)))
    (is (= [18.71788442872143
            1.0450233625100218
            0.017857500314128354
            0.02176857925270321
            1.6353730686050887
            1.8944142995327073
            0.7308447960234485
            2.104508605707742
            2.0566505534822648
            1.4932599610727966
            0.8287524976817674]
           (-> tidy :std.error)))
    (is (= [0.5181243968984746
            0.9160873755159584
            0.46348865035386666
            0.33495531411697543
            0.6352778979694993
            0.06325215114455572
            0.2739412699723611
            0.8814234719769896
            0.23398971070679475
            0.6652064342930224
            0.8121787129526958]
           (-> tidy :p.value)))
    (is (= [0.657305808402011
            -0.1066392215569186
            0.7467584868409959
            -0.986840654126232
            0.48130361649375314
            -1.9611887057883417
            1.1234132802777572
            0.1509914539306682
            1.2254035489603454
            0.43891421063143365
            -0.24062582666609506]
           (-> tidy :statistic)))
    (is (= [:mpg :cyl :disp :hp :drat :wt :qsec :vs :am :gear :carb :.resid :.fitted]
           (ds/column-names augment)))
    (is (=
         [-1.599505761262371
          -1.1118860793566583
          -3.4506440847987783
          0.16259545332427194
          1.0065659713025106
          -2.283039035671571
          -0.08625625278094162
          1.9039881151113356
          -1.61908989756229
          0.5009700576605098
          -1.391654392144261
          2.227837889556623
          1.7004264039903525
          -0.5422246994631266
          -1.634013415128031
          -0.5364377108561804
          4.206370638165357
          4.627094192221644
          0.503261088688653
          4.387630904261229
          -2.143103441605021
          -1.4430532211760934
          -2.532181497829331
          -0.00602197620179723
          2.508321011307789
          -0.9934686934445978
          -0.15295396086860436
          2.7637274172027553
          -3.0700408028647495
          0.0061718455252339766
          1.058881617940136
          -2.9682676832437735]
         (:.resid augment)))
    (is (= [22.599505761262364
            22.11188607935665
            26.25064408479878
            21.237404546675727
            17.69343402869749
            20.383039035671572
            14.386256252780942
            22.496011884888667
            24.419089897562287
            18.699029942339486
            19.19165439214426
            14.172162110443374
            15.599573596009648
            15.742224699463126
            12.03401341512803
            10.936437710856179
            10.493629361834639
            27.77290580777835
            29.896738911311346
            29.512369095738777
            23.643103441605028
            16.943053221176093
            17.73218149782933
            13.306021976201801
            16.69167898869221
            28.293468693444595
            26.152953960868608
            27.636272582797247
            18.87004080286475
            19.693828154474765
            13.941118382059862
            24.368267683243772]
           (augment :.fitted)))))
;; => #'scicloj.metamorph.linear-regression-test/linear-regression-mtcars







;; (->
;;  (ml/glance model)
;;  (ds/rows)
;;  first)
;; => {:p.value 3.7931521057466E-7,
;;     :statistic 13.932463690208833,
;;     :adj.r.squared 0.8066423189909859,
;;     :n 32,
;;     :mse 97.85527574833492,
;;     :rss 147.49443001665065,
;;     :df 10,
;;     :df.residual 21,
;;     :aic 161.70981043447966,
;;     :bic 177.83290536527664,
;;     :totss 1126.0471874999998,
;;     :r.squared 0.8690157644777647,
;;     :log-lik -69.85490521723983}

;; R
;; [,1]
;; r.squared       0.8690157644778
;; adj.r.squared   0.8066423189910
;; sigma           2.6501970278655
;; statistic      13.9324636902088
;; p.value         0.0000003793152
;; df             10.0000000000000
;; logLik        -69.8549052172399
;; AIC           163.7098104344797
;; BIC           181.2986412680764
;; deviance      147.4944300166508
;; df.residual    21.0000000000000
;; nobs           32.0000000000000




;; (ml/tidy model)
;; => _unnamed [11 5]:
;;    | :term |  :statistic |   :estimate |   :p.value |  :std.error |
;;    |-------|------------:|------------:|-----------:|------------:|
;;    |  :mpg |  0.65730581 | 12.30337416 | 0.51812440 | 18.71788443 |
;;    |  :cyl | -0.10663922 | -0.11144048 | 0.91608738 |  1.04502336 |
;;    | :disp |  0.74675849 |  0.01333524 | 0.46348865 |  0.01785750 |
;;    |   :hp | -0.98684065 | -0.02148212 | 0.33495531 |  0.02176858 |
;;    | :drat |  0.48130362 |  0.78711097 | 0.63527790 |  1.63537307 |
;;    |   :wt | -1.96118871 | -3.71530393 | 0.06325215 |  1.89441430 |
;;    | :qsec |  1.12341328 |  0.82104075 | 0.27394127 |  0.73084480 |
;;    |   :vs |  0.15099145 |  0.31776281 | 0.88142347 |  2.10450861 |
;;    |   :am |  1.22540355 |  2.52022689 | 0.23398971 |  2.05665055 |
;;    | :gear |  0.43891421 |  0.65541302 | 0.66520643 |  1.49325996 |
;;    | :carb | -0.24062583 | -0.19941925 | 0.81217871 |  0.82875250 |


;; R
;; > tidy(m)
;; # A tibble: 11 × 5
;;    term        estimate std.error statistic p.value
;;    <chr>          <dbl>     <dbl>     <dbl>   <dbl>
;;  1 (Intercept)  12.3      18.7        0.657  0.518
;;  2 cyl          -0.111     1.05      -0.107  0.916
;;  3 disp          0.0133    0.0179     0.747  0.463
;;  4 hp           -0.0215    0.0218    -0.987  0.335
;;  5 drat          0.787     1.64       0.481  0.635
;;  6 wt           -3.72      1.89      -1.96   0.0633
;;  7 qsec          0.821     0.731      1.12   0.274
;;  8 vs            0.318     2.10       0.151  0.881
;;  9 am            2.52      2.06       1.23   0.234
;; 10 gear          0.655     1.49       0.439  0.665
;; 11 carb         -0.199     0.829     -0.241  0.812




;; (ml/augment model ds)
;; => _unnamed [32 13]:
;;    | :mpg | :cyl | :disp | :hp | :drat |   :wt | :qsec | :vs | :am | :gear | :carb |     :.resid |    :.fitted |
;;    |-----:|-----:|------:|----:|------:|------:|------:|----:|----:|------:|------:|------------:|------------:|
;;    | 21.0 |    6 | 160.0 | 110 |  3.90 | 2.620 | 16.46 |   0 |   1 |     4 |     4 | -1.59950576 | 22.59950576 |
;;    | 21.0 |    6 | 160.0 | 110 |  3.90 | 2.875 | 17.02 |   0 |   1 |     4 |     4 | -1.11188608 | 22.11188608 |
;;    | 22.8 |    4 | 108.0 |  93 |  3.85 | 2.320 | 18.61 |   1 |   1 |     4 |     1 | -3.45064408 | 26.25064408 |
;;    | 21.4 |    6 | 258.0 | 110 |  3.08 | 3.215 | 19.44 |   1 |   0 |     3 |     1 |  0.16259545 | 21.23740455 |
;;    | 18.7 |    8 | 360.0 | 175 |  3.15 | 3.440 | 17.02 |   0 |   0 |     3 |     2 |  1.00656597 | 17.69343403 |
;;    | 18.1 |    6 | 225.0 | 105 |  2.76 | 3.460 | 20.22 |   1 |   0 |     3 |     1 | -2.28303904 | 20.38303904 |
;;    | 14.3 |    8 | 360.0 | 245 |  3.21 | 3.570 | 15.84 |   0 |   0 |     3 |     4 | -0.08625625 | 14.38625625 |
;;    | 24.4 |    4 | 146.7 |  62 |  3.69 | 3.190 | 20.00 |   1 |   0 |     4 |     2 |  1.90398812 | 22.49601188 |
;;    | 22.8 |    4 | 140.8 |  95 |  3.92 | 3.150 | 22.90 |   1 |   0 |     4 |     2 | -1.61908990 | 24.41908990 |
;;    | 19.2 |    6 | 167.6 | 123 |  3.92 | 3.440 | 18.30 |   1 |   0 |     4 |     4 |  0.50097006 | 18.69902994 |
;;    |  ... |  ... |   ... | ... |   ... |   ... |   ... | ... | ... |   ... |   ... |         ... |         ... |
;;    | 15.5 |    8 | 318.0 | 150 |  2.76 | 3.520 | 16.87 |   0 |   0 |     3 |     2 | -1.44305322 | 16.94305322 |
;;    | 15.2 |    8 | 304.0 | 150 |  3.15 | 3.435 | 17.30 |   0 |   0 |     3 |     2 | -2.53218150 | 17.73218150 |
;;    | 13.3 |    8 | 350.0 | 245 |  3.73 | 3.840 | 15.41 |   0 |   0 |     3 |     4 | -0.00602198 | 13.30602198 |
;;    | 19.2 |    8 | 400.0 | 175 |  3.08 | 3.845 | 17.05 |   0 |   0 |     3 |     2 |  2.50832101 | 16.69167899 |
;;    | 27.3 |    4 |  79.0 |  66 |  4.08 | 1.935 | 18.90 |   1 |   1 |     4 |     1 | -0.99346869 | 28.29346869 |
;;    | 26.0 |    4 | 120.3 |  91 |  4.43 | 2.140 | 16.70 |   0 |   1 |     5 |     2 | -0.15295396 | 26.15295396 |
;;    | 30.4 |    4 |  95.1 | 113 |  3.77 | 1.513 | 16.90 |   1 |   1 |     5 |     2 |  2.76372742 | 27.63627258 |
;;    | 15.8 |    8 | 351.0 | 264 |  4.22 | 3.170 | 14.50 |   0 |   1 |     5 |     4 | -3.07004080 | 18.87004080 |
;;    | 19.7 |    6 | 145.0 | 175 |  3.62 | 2.770 | 15.50 |   0 |   1 |     5 |     6 |  0.00617185 | 19.69382815 |
;;    | 15.0 |    8 | 301.0 | 335 |  3.54 | 3.570 | 14.60 |   0 |   1 |     5 |     8 |  1.05888162 | 13.94111838 |
;;    | 21.4 |    4 | 121.0 | 109 |  4.11 | 2.780 | 18.60 |   1 |   1 |     4 |     2 | -2.96826768 | 24.36826768 |



;; ----------------------------------------------------------------
;;  R
;;  m=lm(mpg ~ .,mtcars)


;; print(augment(m),width=Inf)
;; # A tibble: 32 × 18
;;    .rownames           mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb .fitted  .resid  .hat .sigma   .cooksd .std.resid
;;    <chr>             <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>   <dbl>   <dbl> <dbl>  <dbl>     <dbl>      <dbl>
;;  1 Mazda RX4          21       6  160    110  3.9   2.62  16.5     0     1     4     4    22.6 -1.60   0.303   2.68 0.0206       -0.723
;;  2 Mazda RX4 Wag      21       6  160    110  3.9   2.88  17.0     0     1     4     4    22.1 -1.11   0.290   2.70 0.00922      -0.498
;;  3 Datsun 710         22.8     4  108     93  3.85  2.32  18.6     1     1     4     1    26.3 -3.45   0.239   2.57 0.0635       -1.49
;;  4 Hornet 4 Drive     21.4     6  258    110  3.08  3.22  19.4     1     0     3     1    21.2  0.163  0.228   2.72 0.000131      0.0698
;;  5 Hornet Sportabout  18.7     8  360    175  3.15  3.44  17.0     0     0     3     2    17.7  1.01   0.200   2.70 0.00408       0.425
;;  6 Valiant            18.1     6  225    105  2.76  3.46  20.2     1     0     3     1    20.4 -2.28   0.282   2.65 0.0370       -1.02
;;  7 Duster 360         14.3     8  360    245  3.21  3.57  15.8     0     0     3     4    14.4 -0.0863 0.326   2.72 0.0000691    -0.0396
;;  8 Merc 240D          24.4     4  147.    62  3.69  3.19  20       1     0     4     2    22.5  1.90   0.330   2.67 0.0345        0.878
;;  9 Merc 230           22.8     4  141.    95  3.92  3.15  22.9     1     0     4     2    24.4 -1.62   0.742   2.62 0.379        -1.20
;; 10 Merc 280           19.2     6  168.   123  3.92  3.44  18.3     1     0     4     4    18.7  0.501  0.429   2.71 0.00428       0.250

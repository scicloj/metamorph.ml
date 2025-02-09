(ns scicloj.metamorph.ml.rdatasets
  (:require [tablecloth.api :as tc] [clojure.string :as str] [camel-snake-kebab.core :as csk])
  (:import [com.vladsch.flexmark.html2md.converter FlexmarkHtmlConverter]))


;;    Based on
;;    @Manual{,
;;    title = {Rdatasets: A collection of datasets originally distributed in various R packages},
;;    author = {Vincent Arel-Bundock},
;;    year = {2024},
;;    note = {R package version 1.0.0},
;;    url = {https://vincentarelbundock.github.io/Rdatasets},
;;   }
    
(defn clean-R-relevant [s]
  (-> (str/replace s #"\n\n### Examples(?s).*```" "")
   (str/replace ":::: container\n::: container\n" "")
   (str/replace #"### Usage(?s).*```" "")
   (str/replace "R Documentation" "Documentation")
   (str/replace #"\|(\-*)\|(\-*)\|" "")))
(defn doc-url->md [doc]
  (clean-R-relevant (.. (FlexmarkHtmlConverter/builder) build (convert (slurp doc)))))
(defn _fetch-dataset [csv] (-> csv (tc/dataset {:key-fn csk/->kebab-case-keyword})))

(def fetch-dataset (memoize _fetch-dataset))

(defn AER-Affairs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Affairs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Affairs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Affairs.csv"))

(defn AER-ArgentinaCPI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ArgentinaCPI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ArgentinaCPI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ArgentinaCPI.csv"))

(defn AER-BankWages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/BankWages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/BankWages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/BankWages.csv"))

(defn AER-BenderlyZwick
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/BenderlyZwick.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/BenderlyZwick.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/BenderlyZwick.csv"))

(defn AER-BondYield
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/BondYield.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/BondYield.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/BondYield.csv"))

(defn AER-CartelStability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CartelStability.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CartelStability.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CartelStability.csv"))

(defn AER-CASchools
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CASchools.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CASchools.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CASchools.csv"))

(defn AER-ChinaIncome
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ChinaIncome.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ChinaIncome.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ChinaIncome.csv"))

(defn AER-CigarettesB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CigarettesB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CigarettesB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CigarettesB.csv"))

(defn AER-CigarettesSW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CigarettesSW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CigarettesSW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CigarettesSW.csv"))

(defn AER-CollegeDistance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CollegeDistance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CollegeDistance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv"))

(defn AER-ConsumerGood
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ConsumerGood.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ConsumerGood.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ConsumerGood.csv"))

(defn AER-CPS1985
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPS1985.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPS1985.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPS1985.csv"))

(defn AER-CPS1988
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPS1988.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPS1988.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPS1988.csv"))

(defn AER-CPSSW04
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW04.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW04.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSW04.csv"))

(defn AER-CPSSW3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSW3.csv"))

(defn AER-CPSSW8
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW8.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW8.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSW8.csv"))

(defn AER-CPSSW9204
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW9204.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW9204.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSW9204.csv"))

(defn AER-CPSSW9298
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW9298.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSW9298.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSW9298.csv"))

(defn AER-CPSSWEducation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSWEducation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CPSSWEducation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CPSSWEducation.csv"))

(defn AER-CreditCard
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/CreditCard.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/CreditCard.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/CreditCard.csv"))

(defn AER-DJFranses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/DJFranses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/DJFranses.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/DJFranses.csv"))

(defn AER-DJIA8012
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/DJIA8012.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/DJIA8012.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/DJIA8012.csv"))

(defn AER-DoctorVisits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/DoctorVisits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/DoctorVisits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/DoctorVisits.csv"))

(defn AER-DutchAdvert
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/DutchAdvert.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/DutchAdvert.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/DutchAdvert.csv"))

(defn AER-DutchSales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/DutchSales.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/DutchSales.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/DutchSales.csv"))

(defn AER-Electricity1955
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Electricity1955.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Electricity1955.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Electricity1955.csv"))

(defn AER-Electricity1970
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Electricity1970.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Electricity1970.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Electricity1970.csv"))

(defn AER-EquationCitations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/EquationCitations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/EquationCitations.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/EquationCitations.csv"))

(defn AER-Equipment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Equipment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Equipment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Equipment.csv"))

(defn AER-EuroEnergy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/EuroEnergy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/EuroEnergy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/EuroEnergy.csv"))

(defn AER-Fatalities
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fatalities.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fatalities.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Fatalities.csv"))

(defn AER-Fertility
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fertility.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fertility.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Fertility.csv"))

(defn AER-Fertility2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fertility2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Fertility2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Fertility2.csv"))

(defn AER-FrozenJuice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/FrozenJuice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/FrozenJuice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/FrozenJuice.csv"))

(defn AER-GermanUnemployment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GermanUnemployment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GermanUnemployment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GermanUnemployment.csv"))

(defn AER-GoldSilver
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GoldSilver.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GoldSilver.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GoldSilver.csv"))

(defn AER-GrowthDJ
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GrowthDJ.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GrowthDJ.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GrowthDJ.csv"))

(defn AER-GrowthSW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GrowthSW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GrowthSW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GrowthSW.csv"))

(defn AER-Grunfeld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Grunfeld.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Grunfeld.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Grunfeld.csv"))

(defn AER-GSOEP9402
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GSOEP9402.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GSOEP9402.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GSOEP9402.csv"))

(defn AER-GSS7402
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/GSS7402.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/GSS7402.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/GSS7402.csv"))

(defn AER-Guns
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Guns.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Guns.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Guns.csv"))

(defn AER-HealthInsurance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/HealthInsurance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/HealthInsurance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/HealthInsurance.csv"))

(defn AER-HMDA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/HMDA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/HMDA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/HMDA.csv"))

(defn AER-HousePrices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/HousePrices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv"))

(defn AER-Journals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Journals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Journals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Journals.csv"))

(defn AER-KleinI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/KleinI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/KleinI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/KleinI.csv"))

(defn AER-Longley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Longley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Longley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Longley.csv"))

(defn AER-ManufactCosts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ManufactCosts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ManufactCosts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ManufactCosts.csv"))

(defn AER-MarkDollar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MarkDollar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MarkDollar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MarkDollar.csv"))

(defn AER-MarkPound
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MarkPound.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MarkPound.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MarkPound.csv"))

(defn AER-MASchools
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MASchools.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MASchools.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MASchools.csv"))

(defn AER-Medicaid1986
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Medicaid1986.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Medicaid1986.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Medicaid1986.csv"))

(defn AER-Mortgage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Mortgage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Mortgage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Mortgage.csv"))

(defn AER-MotorCycles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MotorCycles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MotorCycles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MotorCycles.csv"))

(defn AER-MotorCycles2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MotorCycles2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MotorCycles2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MotorCycles2.csv"))

(defn AER-MSCISwitzerland
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MSCISwitzerland.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MSCISwitzerland.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MSCISwitzerland.csv"))

(defn AER-Municipalities
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Municipalities.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Municipalities.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Municipalities.csv"))

(defn AER-MurderRates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/MurderRates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/MurderRates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/MurderRates.csv"))

(defn AER-NaturalGas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/NaturalGas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/NaturalGas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/NaturalGas.csv"))

(defn AER-NMES1988
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/NMES1988.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/NMES1988.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/NMES1988.csv"))

(defn AER-NYSESW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/NYSESW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/NYSESW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/NYSESW.csv"))

(defn AER-OECDGas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/OECDGas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/OECDGas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/OECDGas.csv"))

(defn AER-OECDGrowth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/OECDGrowth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/OECDGrowth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/OECDGrowth.csv"))

(defn AER-OlympicTV
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/OlympicTV.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/OlympicTV.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/OlympicTV.csv"))

(defn AER-OrangeCounty
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/OrangeCounty.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/OrangeCounty.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/OrangeCounty.csv"))

(defn AER-Parade2005
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/Parade2005.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/Parade2005.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/Parade2005.csv"))

(defn AER-PepperPrice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/PepperPrice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/PepperPrice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PepperPrice.csv"))

(defn AER-PhDPublications
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/PhDPublications.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/PhDPublications.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PhDPublications.csv"))

(defn AER-ProgramEffectiveness
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ProgramEffectiveness.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ProgramEffectiveness.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ProgramEffectiveness.csv"))

(defn AER-PSID1976
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID1976.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID1976.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PSID1976.csv"))

(defn AER-PSID1982
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID1982.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID1982.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PSID1982.csv"))

(defn AER-PSID7682
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID7682.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/PSID7682.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/PSID7682.csv"))

(defn AER-RecreationDemand
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/RecreationDemand.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/RecreationDemand.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/RecreationDemand.csv"))

(defn AER-ResumeNames
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ResumeNames.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ResumeNames.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ResumeNames.csv"))

(defn AER-ShipAccidents
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/ShipAccidents.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/ShipAccidents.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/ShipAccidents.csv"))

(defn AER-SIC33
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/SIC33.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/SIC33.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/SIC33.csv"))

(defn AER-SmokeBan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/SmokeBan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/SmokeBan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/SmokeBan.csv"))

(defn AER-SportsCards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/SportsCards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/SportsCards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/SportsCards.csv"))

(defn AER-STAR
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/STAR.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/STAR.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/STAR.csv"))

(defn AER-StrikeDuration
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/StrikeDuration.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/StrikeDuration.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/StrikeDuration.csv"))

(defn AER-SwissLabor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/SwissLabor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/SwissLabor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/SwissLabor.csv"))

(defn AER-TeachingRatings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/TeachingRatings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/TeachingRatings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/TeachingRatings.csv"))

(defn AER-TechChange
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/TechChange.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/TechChange.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/TechChange.csv"))

(defn AER-TradeCredit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/TradeCredit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/TradeCredit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/TradeCredit.csv"))

(defn AER-TravelMode
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/TravelMode.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/TravelMode.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/TravelMode.csv"))

(defn AER-UKInflation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/UKInflation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/UKInflation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/UKInflation.csv"))

(defn AER-UKNonDurables
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/UKNonDurables.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/UKNonDurables.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/UKNonDurables.csv"))

(defn AER-USAirlines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USAirlines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USAirlines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USAirlines.csv"))

(defn AER-USConsump1950
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1950.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1950.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USConsump1950.csv"))

(defn AER-USConsump1979
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1979.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1979.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USConsump1979.csv"))

(defn AER-USConsump1993
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1993.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USConsump1993.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USConsump1993.csv"))

(defn AER-USCrudes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USCrudes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USCrudes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USCrudes.csv"))

(defn AER-USGasB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USGasB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USGasB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USGasB.csv"))

(defn AER-USGasG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USGasG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USGasG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USGasG.csv"))

(defn AER-USInvest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USInvest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USInvest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USInvest.csv"))

(defn AER-USMacroB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMacroB.csv"))

(defn AER-USMacroG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMacroG.csv"))

(defn AER-USMacroSW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMacroSW.csv"))

(defn AER-USMacroSWM
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSWM.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSWM.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMacroSWM.csv"))

(defn AER-USMacroSWQ
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSWQ.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMacroSWQ.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMacroSWQ.csv"))

(defn AER-USMoney
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMoney.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USMoney.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USMoney.csv"))

(defn AER-USProdIndex
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USProdIndex.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USProdIndex.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USProdIndex.csv"))

(defn AER-USSeatBelts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USSeatBelts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USSeatBelts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USSeatBelts.csv"))

(defn AER-USStocksSW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/USStocksSW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/USStocksSW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/USStocksSW.csv"))

(defn AER-WeakInstrument
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/AER/WeakInstrument.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/AER/WeakInstrument.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/AER/WeakInstrument.csv"))

(defn aod-antibio
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/antibio.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/antibio.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/antibio.csv"))

(defn aod-cohorts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/cohorts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/cohorts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/cohorts.csv"))

(defn aod-dja
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/dja.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/dja.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/dja.csv"))

(defn aod-lizards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/lizards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/lizards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/lizards.csv"))

(defn aod-mice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/mice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/mice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/mice.csv"))

(defn aod-orob1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/orob1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/orob1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/orob1.csv"))

(defn aod-orob2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/orob2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/orob2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/orob2.csv"))

(defn aod-rabbits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/rabbits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/rabbits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/rabbits.csv"))

(defn aod-rats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/rats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/rats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/rats.csv"))

(defn aod-salmonella
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/aod/salmonella.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/aod/salmonella.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/aod/salmonella.csv"))

(defn asaur-ashkenazi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/ashkenazi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/ashkenazi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/ashkenazi.csv"))

(defn asaur-ChanningHouse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/ChanningHouse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/ChanningHouse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/ChanningHouse.csv"))

(defn asaur-gastricXelox
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/gastricXelox.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/gastricXelox.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/gastricXelox.csv"))

(defn asaur-hepatoCellular
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/hepatoCellular.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/hepatoCellular.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/hepatoCellular.csv"))

(defn asaur-pancreatic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pancreatic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pancreatic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/pancreatic.csv"))

(defn asaur-pancreatic2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pancreatic2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pancreatic2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/pancreatic2.csv"))

(defn asaur-pharmacoSmoking
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pharmacoSmoking.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/pharmacoSmoking.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/pharmacoSmoking.csv"))

(defn asaur-prostateSurvival
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/asaur/prostateSurvival.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/asaur/prostateSurvival.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/asaur/prostateSurvival.csv"))

(defn bakeoff-bakersdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/bakers.csv"))

(defn bakeoff-bakers_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakers_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakers_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/bakers_raw.csv"))

(defn bakeoff-bakes_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakes_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/bakes_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/bakes_raw.csv"))

(defn bakeoff-challengesdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/challenges.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/challenges.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/challenges.csv"))

(defn bakeoff-episodesdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/episodes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/episodes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/episodes.csv"))

(defn bakeoff-episodes_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/episodes_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/episodes_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/episodes_raw.csv"))

(defn bakeoff-ratingsdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/ratings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/ratings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/ratings.csv"))

(defn bakeoff-ratings_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/ratings_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/ratings_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/ratings_raw.csv"))

(defn bakeoff-results_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/results_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/results_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/results_raw.csv"))

(defn bakeoff-seasons_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/seasons_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/seasons_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/seasons_raw.csv"))

(defn bakeoff-series_rawdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/series_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/series_raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/series_raw.csv"))

(defn bakeoff-spice_test_widedata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/spice_test_wide.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/bakeoff/spice_test_wide.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/bakeoff/spice_test_wide.csv"))

(defn betareg-CarTask
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/CarTask.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/CarTask.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/CarTask.csv"))

(defn betareg-FoodExpenditure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/FoodExpenditure.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/FoodExpenditure.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/FoodExpenditure.csv"))

(defn betareg-GasolineYield
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/GasolineYield.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/GasolineYield.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/GasolineYield.csv"))

(defn betareg-ImpreciseTask
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/ImpreciseTask.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/ImpreciseTask.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/ImpreciseTask.csv"))

(defn betareg-LossAversion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/LossAversion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/LossAversion.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/LossAversion.csv"))

(defn betareg-MockJurors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/MockJurors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/MockJurors.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/MockJurors.csv"))

(defn betareg-ReadingSkills
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/ReadingSkills.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/ReadingSkills.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/ReadingSkills.csv"))

(defn betareg-StressAnxiety
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/StressAnxiety.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/StressAnxiety.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/StressAnxiety.csv"))

(defn betareg-WeatherTask
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/betareg/WeatherTask.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/betareg/WeatherTask.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/betareg/WeatherTask.csv"))

(defn boot-acme
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/acme.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/acme.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/acme.csv"))

(defn boot-aids
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/aids.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/aids.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/aids.csv"))

(defn boot-aircondit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/aircondit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/aircondit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/aircondit.csv"))

(defn boot-aircondit7
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/aircondit7.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/aircondit7.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/aircondit7.csv"))

(defn boot-amis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/amis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/amis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/amis.csv"))

(defn boot-aml
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/aml.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/aml.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/aml.csv"))

(defn boot-beaver
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/beaver.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/beaver.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/beaver.csv"))

(defn boot-bigcity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/bigcity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/bigcity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/bigcity.csv"))

(defn boot-brambles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/brambles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/brambles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/brambles.csv"))

(defn boot-breslow
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/breslow.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/breslow.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/breslow.csv"))

(defn boot-calcium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/calcium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/calcium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/calcium.csv"))

(defn boot-cane
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/cane.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/cane.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/cane.csv"))

(defn boot-capability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/capability.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/capability.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/capability.csv"))

(defn boot-catsM
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/catsM.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/catsM.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/catsM.csv"))

(defn boot-cav
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/cav.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/cav.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/cav.csv"))

(defn boot-cd4
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/cd4.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/cd4.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/cd4.csv"))

(defn boot-channing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/channing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/channing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/channing.csv"))

(defn boot-city
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/city.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/city.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/city.csv"))

(defn boot-claridge
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/claridge.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/claridge.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/claridge.csv"))

(defn boot-cloth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/cloth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/cloth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/cloth.csv"))

(defn boot-co.transfer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/co.transfer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/co.transfer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/co.transfer.csv"))

(defn boot-coal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/coal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/coal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/coal.csv"))

(defn boot-darwin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/darwin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/darwin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/darwin.csv"))

(defn boot-dogs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/dogs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/dogs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/dogs.csv"))

(defn boot-downs.bc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/downs.bc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/downs.bc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/downs.bc.csv"))

(defn boot-ducks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/ducks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/ducks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/ducks.csv"))

(defn boot-fir
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/fir.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/fir.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/fir.csv"))

(defn boot-frets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/frets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/frets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/frets.csv"))

(defn boot-grav
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/grav.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/grav.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/grav.csv"))

(defn boot-gravity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/gravity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/gravity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/gravity.csv"))

(defn boot-hirose
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/hirose.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/hirose.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/hirose.csv"))

(defn boot-islay
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/islay.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/islay.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/islay.csv"))

(defn boot-manaus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/manaus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/manaus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/manaus.csv"))

(defn boot-melanoma
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/melanoma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/melanoma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/melanoma.csv"))

(defn boot-motor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/motor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/motor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/motor.csv"))

(defn boot-neuro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/neuro.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/neuro.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/neuro.csv"))

(defn boot-nitrofen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/nitrofen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/nitrofen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/nitrofen.csv"))

(defn boot-nodal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/nodal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/nodal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/nodal.csv"))

(defn boot-nuclear
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/nuclear.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/nuclear.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/nuclear.csv"))

(defn boot-paulsen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/paulsen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/paulsen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/paulsen.csv"))

(defn boot-poisons
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/poisons.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/poisons.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/poisons.csv"))

(defn boot-polar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/polar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/polar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/polar.csv"))

(defn boot-remission
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/remission.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/remission.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/remission.csv"))

(defn boot-salinity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/salinity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/salinity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/salinity.csv"))

(defn boot-survival
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/survival.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/survival.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/survival.csv"))

(defn boot-tau
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/tau.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/tau.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/tau.csv"))

(defn boot-tuna
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/tuna.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/tuna.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/tuna.csv"))

(defn boot-urine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/urine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/urine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/urine.csv"))

(defn boot-wool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/boot/wool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/boot/wool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/boot/wool.csv"))

(defn carData-Adler
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Adler.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Adler.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Adler.csv"))

(defn carData-AMSsurvey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/AMSsurvey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/AMSsurvey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/AMSsurvey.csv"))

(defn carData-Angell
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Angell.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Angell.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Angell.csv"))

(defn carData-Anscombe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Anscombe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Anscombe.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Anscombe.csv"))

(defn carData-Arrests
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Arrests.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Arrests.csv"))

(defn carData-Baumann
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Baumann.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Baumann.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Baumann.csv"))

(defn carData-BEPS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/BEPS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/BEPS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/BEPS.csv"))

(defn carData-Bfox
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Bfox.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Bfox.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Bfox.csv"))

(defn carData-Blackmore
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Blackmore.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Blackmore.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Blackmore.csv"))

(defn carData-Burt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Burt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Burt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Burt.csv"))

(defn carData-CanPop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/CanPop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/CanPop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/CanPop.csv"))

(defn carData-CES11
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/CES11.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/CES11.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/CES11.csv"))

(defn carData-Chile
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Chile.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Chile.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Chile.csv"))

(defn carData-Chirot
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Chirot.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Chirot.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Chirot.csv"))

(defn carData-Cowles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Cowles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Cowles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Cowles.csv"))

(defn carData-Davis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Davis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Davis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Davis.csv"))

(defn carData-DavisThin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/DavisThin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/DavisThin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/DavisThin.csv"))

(defn carData-Depredations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Depredations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Depredations.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Depredations.csv"))

(defn carData-Duncan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Duncan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Duncan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Duncan.csv"))

(defn carData-Ericksen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ericksen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ericksen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Ericksen.csv"))

(defn carData-Florida
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Florida.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Florida.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Florida.csv"))

(defn carData-Freedman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Freedman.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Freedman.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Freedman.csv"))

(defn carData-Friendly
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Friendly.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Friendly.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Friendly.csv"))

(defn carData-Ginzberg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ginzberg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ginzberg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Ginzberg.csv"))

(defn carData-Greene
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Greene.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Greene.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Greene.csv"))

(defn carData-GSSvocab
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/GSSvocab.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/GSSvocab.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/GSSvocab.csv"))

(defn carData-Guyer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Guyer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Guyer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Guyer.csv"))

(defn carData-Hartnagel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Hartnagel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Hartnagel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Hartnagel.csv"))

(defn carData-Highway1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Highway1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Highway1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Highway1.csv"))

(defn carData-KosteckiDillon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/KosteckiDillon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/KosteckiDillon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/KosteckiDillon.csv"))

(defn carData-Leinhardt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Leinhardt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Leinhardt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Leinhardt.csv"))

(defn carData-LoBD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/LoBD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/LoBD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/LoBD.csv"))

(defn carData-Mandel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Mandel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Mandel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Mandel.csv"))

(defn carData-Migration
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Migration.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Migration.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Migration.csv"))

(defn carData-Moore
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Moore.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Moore.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Moore.csv"))

(defn carData-MplsDemo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/MplsDemo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/MplsDemo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/MplsDemo.csv"))

(defn carData-MplsStops
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/MplsStops.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/MplsStops.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/MplsStops.csv"))

(defn carData-Mroz
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Mroz.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Mroz.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Mroz.csv"))

(defn carData-OBrienKaiser
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/OBrienKaiser.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/OBrienKaiser.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/OBrienKaiser.csv"))

(defn carData-OBrienKaiserLong
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/OBrienKaiserLong.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/OBrienKaiserLong.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/OBrienKaiserLong.csv"))

(defn carData-Ornstein
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ornstein.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Ornstein.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Ornstein.csv"))

(defn carData-Pottery
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Pottery.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Pottery.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Pottery.csv"))

(defn carData-Prestige
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Prestige.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Prestige.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Prestige.csv"))

(defn carData-Quartet
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Quartet.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Quartet.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Quartet.csv"))

(defn carData-Robey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Robey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Robey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Robey.csv"))

(defn carData-Rossi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Rossi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Rossi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Rossi.csv"))

(defn carData-Sahlins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Sahlins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Sahlins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Sahlins.csv"))

(defn carData-Salaries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Salaries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Salaries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Salaries.csv"))

(defn carData-SLID
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/SLID.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/SLID.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/SLID.csv"))

(defn carData-Soils
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Soils.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Soils.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Soils.csv"))

(defn carData-States
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/States.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/States.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/States.csv"))

(defn carData-TitanicSurvival
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/TitanicSurvival.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/TitanicSurvival.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/TitanicSurvival.csv"))

(defn carData-Transact
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Transact.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Transact.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Transact.csv"))

(defn carData-UN
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/UN.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/UN.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/UN.csv"))

(defn carData-UN98
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/UN98.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/UN98.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/UN98.csv"))

(defn carData-USPop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/USPop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/USPop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/USPop.csv"))

(defn carData-Vocab
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Vocab.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Vocab.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Vocab.csv"))

(defn carData-WeightLoss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/WeightLoss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/WeightLoss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/WeightLoss.csv"))

(defn carData-Wells
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wells.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wells.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wells.csv"))

(defn carData-Womenlf
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Womenlf.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Womenlf.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Womenlf.csv"))

(defn carData-Wong
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wong.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wong.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wong.csv"))

(defn carData-Wool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/Wool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Wool.csv"))

(defn carData-WVS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/carData/WVS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/carData/WVS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/carData/WVS.csv"))

(defn causaldata-abortion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/abortion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/abortion.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/abortion.csv"))

(defn causaldata-adult_services
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/adult_services.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/adult_services.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/adult_services.csv"))

(defn causaldata-auto
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/auto.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/auto.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/auto.csv"))

(defn causaldata-avocado
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/avocado.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/avocado.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/avocado.csv"))

(defn causaldata-black_politicians
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/black_politicians.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/black_politicians.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/black_politicians.csv"))

(defn causaldata-castle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/castle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/castle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/castle.csv"))

(defn causaldata-close_college
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/close_college.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/close_college.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/close_college.csv"))

(defn causaldata-close_elections_lmb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/close_elections_lmb.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/close_elections_lmb.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/close_elections_lmb.csv"))

(defn causaldata-cps_mixtape
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/cps_mixtape.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/cps_mixtape.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/cps_mixtape.csv"))

(defn causaldata-credit_cards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/credit_cards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/credit_cards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/credit_cards.csv"))

(defn causaldata-gapminder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gapminder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gapminder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/gapminder.csv"))

(defn causaldata-google_stock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/google_stock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/google_stock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/google_stock.csv"))

(defn causaldata-gov_transfers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gov_transfers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gov_transfers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/gov_transfers.csv"))

(defn causaldata-gov_transfers_density
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gov_transfers_density.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/gov_transfers_density.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/gov_transfers_density.csv"))

(defn causaldata-greek_data
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/greek_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/greek_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/greek_data.csv"))

(defn causaldata-mortgages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/mortgages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/mortgages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/mortgages.csv"))

(defn causaldata-Mroz
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/Mroz.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/Mroz.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/Mroz.csv"))

(defn causaldata-nhefs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs.csv"))

(defn causaldata-nhefs_codebook
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs_codebook.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs_codebook.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs_codebook.csv"))

(defn causaldata-nhefs_complete
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs_complete.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nhefs_complete.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nhefs_complete.csv"))

(defn causaldata-nsw_mixtape
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nsw_mixtape.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/nsw_mixtape.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/nsw_mixtape.csv"))

(defn causaldata-organ_donations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/organ_donations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/organ_donations.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/organ_donations.csv"))

(defn causaldata-restaurant_inspections
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/restaurant_inspections.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/restaurant_inspections.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/restaurant_inspections.csv"))

(defn causaldata-ri
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/ri.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/ri.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/ri.csv"))

(defn causaldata-scorecard
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/scorecard.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/scorecard.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/scorecard.csv"))

(defn causaldata-snow
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/snow.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/snow.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/snow.csv"))

(defn causaldata-social_insure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/social_insure.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/social_insure.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/social_insure.csv"))

(defn causaldata-texas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/texas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/texas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/texas.csv"))

(defn causaldata-thornton_hiv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/thornton_hiv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/thornton_hiv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/thornton_hiv.csv"))

(defn causaldata-titanic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/titanic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/titanic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/titanic.csv"))

(defn causaldata-training_bias_reduction
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/training_bias_reduction.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/training_bias_reduction.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/training_bias_reduction.csv"))

(defn causaldata-training_example
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/training_example.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/training_example.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/training_example.csv"))

(defn causaldata-yule
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/yule.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/causaldata/yule.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/causaldata/yule.csv"))

(defn cluster-agriculture
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/agriculture.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/agriculture.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/agriculture.csv"))

(defn cluster-animals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/animals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/animals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/animals.csv"))

(defn cluster-chorSub
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/chorSub.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/chorSub.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/chorSub.csv"))

(defn cluster-flower
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/flower.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/flower.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/flower.csv"))

(defn cluster-plantTraits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/plantTraits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/plantTraits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/plantTraits.csv"))

(defn cluster-pluton
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/pluton.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/pluton.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/pluton.csv"))

(defn cluster-ruspini
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/ruspini.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/ruspini.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/ruspini.csv"))

(defn cluster-votes.repub
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/votes.repub.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/votes.repub.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/votes.repub.csv"))

(defn cluster-xclara
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/cluster/xclara.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/cluster/xclara.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/cluster/xclara.csv"))

(defn collegeScorecard-school
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/collegeScorecard/school.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/collegeScorecard/school.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/collegeScorecard/school.csv"))

(defn collegeScorecard-scorecard
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/collegeScorecard/scorecard.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/collegeScorecard/scorecard.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/collegeScorecard/scorecard.csv"))

(defn COUNT-affairs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/affairs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/affairs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/affairs.csv"))

(defn COUNT-azcabgptca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azcabgptca.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azcabgptca.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/azcabgptca.csv"))

(defn COUNT-azdrg112
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azdrg112.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azdrg112.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/azdrg112.csv"))

(defn COUNT-azpro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azpro.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azpro.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/azpro.csv"))

(defn COUNT-azprocedure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azprocedure.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/azprocedure.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/azprocedure.csv"))

(defn COUNT-badhealth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/badhealth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/badhealth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/badhealth.csv"))

(defn COUNT-fasttrakg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/fasttrakg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/fasttrakg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/fasttrakg.csv"))

(defn COUNT-fishing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/fishing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/fishing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/fishing.csv"))

(defn COUNT-lbw
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/lbw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/lbw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/lbw.csv"))

(defn COUNT-lbwgrp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/lbwgrp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/lbwgrp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/lbwgrp.csv"))

(defn COUNT-loomis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/loomis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/loomis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/loomis.csv"))

(defn COUNT-mdvis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/mdvis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/mdvis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/mdvis.csv"))

(defn COUNT-medpar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/medpar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/medpar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/medpar.csv"))

(defn COUNT-nuts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/nuts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/nuts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/nuts.csv"))

(defn COUNT-rwm
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/rwm.csv"))

(defn COUNT-rwm1984
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm1984.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm1984.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/rwm1984.csv"))

(defn COUNT-rwm5yr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm5yr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/rwm5yr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/rwm5yr.csv"))

(defn COUNT-ships
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/ships.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/ships.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/ships.csv"))

(defn COUNT-smoking
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/smoking.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/smoking.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/smoking.csv"))

(defn COUNT-titanic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/titanic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/titanic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/titanic.csv"))

(defn COUNT-titanicgrp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/titanicgrp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/COUNT/titanicgrp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/COUNT/titanicgrp.csv"))

(defn crch-RainIbk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/crch/RainIbk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/crch/RainIbk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/crch/RainIbk.csv"))

(defn DAAG-ACF1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ACF1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ACF1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ACF1.csv"))

(defn DAAG-ais
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ais.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ais.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ais.csv"))

(defn DAAG-alc2018
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/alc2018.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/alc2018.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/alc2018.csv"))

(defn DAAG-allbacks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/allbacks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/allbacks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/allbacks.csv"))

(defn DAAG-anesthetic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/anesthetic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/anesthetic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/anesthetic.csv"))

(defn DAAG-ant111b
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ant111b.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ant111b.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ant111b.csv"))

(defn DAAG-antigua
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/antigua.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/antigua.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/antigua.csv"))

(defn DAAG-appletaste
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/appletaste.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/appletaste.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/appletaste.csv"))

(defn DAAG-audists
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/audists.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/audists.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/audists.csv"))

(defn DAAG-aulatlong
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/aulatlong.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/aulatlong.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/aulatlong.csv"))

(defn DAAG-austpop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/austpop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/austpop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/austpop.csv"))

(defn DAAG-biomass
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/biomass.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/biomass.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/biomass.csv"))

(defn DAAG-bomregions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/bomregions.csv"))

(defn DAAG-bomregions2018
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions2018.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions2018.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/bomregions2018.csv"))

(defn DAAG-bomregions2021
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions2021.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomregions2021.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/bomregions2021.csv"))

(defn DAAG-bomsoi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomsoi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bomsoi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/bomsoi.csv"))

(defn DAAG-bostonc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bostonc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/bostonc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/bostonc.csv"))

(defn DAAG-carprice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/carprice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/carprice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/carprice.csv"))

(defn DAAG-Cars93.summary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Cars93.summary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Cars93.summary.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/Cars93.summary.csv"))

(defn DAAG-cerealsugar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cerealsugar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cerealsugar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cerealsugar.csv"))

(defn DAAG-cfseal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cfseal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cfseal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cfseal.csv"))

(defn DAAG-cities
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cities.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cities.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cities.csv"))

(defn DAAG-codling
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/codling.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/codling.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/codling.csv"))

(defn DAAG-coralPval
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/coralPval.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/coralPval.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/coralPval.csv"))

(defn DAAG-cottonworkers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cottonworkers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cottonworkers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cottonworkers.csv"))

(defn DAAG-cps1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cps1.csv"))

(defn DAAG-cps2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cps2.csv"))

(defn DAAG-cps3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cps3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cps3.csv"))

(defn DAAG-cricketer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cricketer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cricketer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cricketer.csv"))

(defn DAAG-cuckoohosts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cuckoohosts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cuckoohosts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cuckoohosts.csv"))

(defn DAAG-cuckoos
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cuckoos.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/cuckoos.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/cuckoos.csv"))

(defn DAAG-dengue
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/dengue.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/dengue.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/dengue.csv"))

(defn DAAG-dewpoint
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/dewpoint.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/dewpoint.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/dewpoint.csv"))

(defn DAAG-droughts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/droughts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/droughts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/droughts.csv"))

(defn DAAG-edcCO2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/edcCO2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/edcCO2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/edcCO2.csv"))

(defn DAAG-edcT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/edcT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/edcT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/edcT.csv"))

(defn DAAG-elastic1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elastic1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elastic1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/elastic1.csv"))

(defn DAAG-elastic2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elastic2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elastic2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/elastic2.csv"))

(defn DAAG-elasticband
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elasticband.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/elasticband.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/elasticband.csv"))

(defn DAAG-fossilfuel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fossilfuel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fossilfuel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/fossilfuel.csv"))

(defn DAAG-fossum
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fossum.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fossum.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/fossum.csv"))

(defn DAAG-frogs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/frogs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/frogs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/frogs.csv"))

(defn DAAG-frostedflakes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/frostedflakes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/frostedflakes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/frostedflakes.csv"))

(defn DAAG-fruitohms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fruitohms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/fruitohms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/fruitohms.csv"))

(defn DAAG-gaba
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/gaba.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/gaba.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/gaba.csv"))

(defn DAAG-geophones
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/geophones.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/geophones.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/geophones.csv"))

(defn DAAG-greatLakes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/greatLakes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/greatLakes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/greatLakes.csv"))

(defn DAAG-grog
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/grog.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/grog.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/grog.csv"))

(defn DAAG-headInjury
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/headInjury.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/headInjury.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/headInjury.csv"))

(defn DAAG-hills
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hills.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hills.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/hills.csv"))

(defn DAAG-hills2000
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hills2000.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hills2000.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/hills2000.csv"))

(defn DAAG-hotspots
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hotspots.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hotspots.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/hotspots.csv"))

(defn DAAG-hotspots2006
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hotspots2006.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hotspots2006.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/hotspots2006.csv"))

(defn DAAG-houseprices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/houseprices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/houseprices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/houseprices.csv"))

(defn DAAG-humanpower1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/humanpower1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/humanpower1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/humanpower1.csv"))

(defn DAAG-humanpower2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/humanpower2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/humanpower2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/humanpower2.csv"))

(defn DAAG-hurricNamed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hurricNamed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/hurricNamed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/hurricNamed.csv"))

(defn DAAG-intersalt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/intersalt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/intersalt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/intersalt.csv"))

(defn DAAG-ironslag
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ironslag.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ironslag.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ironslag.csv"))

(defn DAAG-jobs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/jobs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/jobs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/jobs.csv"))

(defn DAAG-kiwishade
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/kiwishade.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/kiwishade.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/kiwishade.csv"))

(defn DAAG-leafshape
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leafshape.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leafshape.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/leafshape.csv"))

(defn DAAG-leafshape17
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leafshape17.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leafshape17.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/leafshape17.csv"))

(defn DAAG-leaftemp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leaftemp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leaftemp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/leaftemp.csv"))

(defn DAAG-leaftemp.all
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leaftemp.all.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/leaftemp.all.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/leaftemp.all.csv"))

(defn DAAG-litters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/litters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/litters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/litters.csv"))

(defn DAAG-lognihills
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/lognihills.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/lognihills.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/lognihills.csv"))

(defn DAAG-Lottario
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Lottario.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Lottario.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/Lottario.csv"))

(defn DAAG-lung
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/lung.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/lung.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/lung.csv"))

(defn DAAG-Manitoba.lakes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Manitoba.lakes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/Manitoba.lakes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/Manitoba.lakes.csv"))

(defn DAAG-mdbAVtJtoD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mdbAVtJtoD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mdbAVtJtoD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/mdbAVtJtoD.csv"))

(defn DAAG-measles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/measles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/measles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/measles.csv"))

(defn DAAG-medExpenses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/medExpenses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/medExpenses.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/medExpenses.csv"))

(defn DAAG-mifem
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mifem.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mifem.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/mifem.csv"))

(defn DAAG-mignonette
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mignonette.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/mignonette.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/mignonette.csv"))

(defn DAAG-milk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/milk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/milk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/milk.csv"))

(defn DAAG-modelcars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/modelcars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/modelcars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/modelcars.csv"))

(defn DAAG-monica
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/monica.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/monica.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/monica.csv"))

(defn DAAG-moths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/moths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/moths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/moths.csv"))

(defn DAAG-nassCDS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nassCDS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nassCDS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nassCDS.csv"))

(defn DAAG-nasshead
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nasshead.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nasshead.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nasshead.csv"))

(defn DAAG-nihills
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nihills.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nihills.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nihills.csv"))

(defn DAAG-nsw74demo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74demo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74demo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nsw74demo.csv"))

(defn DAAG-nsw74psid1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psid1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psid1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nsw74psid1.csv"))

(defn DAAG-nsw74psid3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psid3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psid3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nsw74psid3.csv"))

(defn DAAG-nsw74psidA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psidA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nsw74psidA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nsw74psidA.csv"))

(defn DAAG-nswdemo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nswdemo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nswdemo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nswdemo.csv"))

(defn DAAG-nswpsid1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nswpsid1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/nswpsid1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/nswpsid1.csv"))

(defn DAAG-oddbooks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/oddbooks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/oddbooks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/oddbooks.csv"))

(defn DAAG-orings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/orings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/orings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/orings.csv"))

(defn DAAG-ozone
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ozone.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/ozone.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/ozone.csv"))

(defn DAAG-pair65
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/pair65.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/pair65.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/pair65.csv"))

(defn DAAG-possum
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/possum.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/possum.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/possum.csv"))

(defn DAAG-possumsites
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/possumsites.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/possumsites.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/possumsites.csv"))

(defn DAAG-poxetc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/poxetc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/poxetc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/poxetc.csv"))

(defn DAAG-primates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/primates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/primates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/primates.csv"))

(defn DAAG-progression
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/progression.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/progression.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/progression.csv"))

(defn DAAG-psid1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/psid1.csv"))

(defn DAAG-psid2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/psid2.csv"))

(defn DAAG-psid3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/psid3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/psid3.csv"))

(defn DAAG-races2000
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/races2000.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/races2000.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/races2000.csv"))

(defn DAAG-rainforest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rainforest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rainforest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/rainforest.csv"))

(defn DAAG-rareplants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rareplants.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rareplants.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/rareplants.csv"))

(defn DAAG-repPsych
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/repPsych.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/repPsych.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/repPsych.csv"))

(defn DAAG-rice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/rice.csv"))

(defn DAAG-rockArt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rockArt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/rockArt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/rockArt.csv"))

(defn DAAG-roller
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/roller.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/roller.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/roller.csv"))

(defn DAAG-science
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/science.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/science.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/science.csv"))

(defn DAAG-seedrates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/seedrates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/seedrates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/seedrates.csv"))

(defn DAAG-socsupport
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/socsupport.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/socsupport.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/socsupport.csv"))

(defn DAAG-softbacks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/softbacks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/softbacks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/softbacks.csv"))

(defn DAAG-sorption
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/sorption.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/sorption.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/sorption.csv"))

(defn DAAG-SP500close
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/SP500close.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/SP500close.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/SP500close.csv"))

(defn DAAG-SP500W90
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/SP500W90.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/SP500W90.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/SP500W90.csv"))

(defn DAAG-spam7
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/spam7.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/spam7.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/spam7.csv"))

(defn DAAG-stVincent
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/stVincent.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/stVincent.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/stVincent.csv"))

(defn DAAG-sugar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/sugar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/sugar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/sugar.csv"))

(defn DAAG-tinting
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/tinting.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/tinting.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/tinting.csv"))

(defn DAAG-tomato
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/tomato.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/tomato.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/tomato.csv"))

(defn DAAG-toycars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/toycars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/toycars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/toycars.csv"))

(defn DAAG-vince111b
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/vince111b.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/vince111b.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/vince111b.csv"))

(defn DAAG-vlt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/vlt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/vlt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/vlt.csv"))

(defn DAAG-wages1833
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/wages1833.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/wages1833.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/wages1833.csv"))

(defn DAAG-whoops
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/whoops.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/whoops.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/whoops.csv"))

(defn DAAG-worldRecords
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/worldRecords.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/DAAG/worldRecords.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/DAAG/worldRecords.csv"))

(defn datasets-ability.cov
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ability.cov.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ability.cov.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ability.cov.csv"))

(defn datasets-airmiles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/airmiles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/airmiles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/airmiles.csv"))

(defn datasets-AirPassengers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/AirPassengers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/AirPassengers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"))

(defn datasets-airquality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/airquality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/airquality.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/airquality.csv"))

(defn datasets-anscombe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/anscombe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/anscombe.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/anscombe.csv"))

(defn datasets-attenu
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/attenu.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/attenu.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/attenu.csv"))

(defn datasets-attitude
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/attitude.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/attitude.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/attitude.csv"))

(defn datasets-austres
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/austres.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/austres.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/austres.csv"))

(defn datasets-beaver1beavers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/beaver1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/beaver1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/beaver1.csv"))

(defn datasets-beaver2beavers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/beaver2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/beaver2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/beaver2.csv"))

(defn datasets-BJsales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BJsales.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BJsales.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/BJsales.csv"))

(defn datasets-BJsales.leadBJsales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BJsales.lead.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BJsales.lead.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/BJsales.lead.csv"))

(defn datasets-BOD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BOD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/BOD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/BOD.csv"))

(defn datasets-cars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/cars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/cars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/cars.csv"))

(defn datasets-ChickWeight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ChickWeight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ChickWeight.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ChickWeight.csv"))

(defn datasets-chickwts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/chickwts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/chickwts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/chickwts.csv"))

(defn datasets-CO2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/CO2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/CO2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/CO2.csv"))

(defn datasets-co2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/co2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/co2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/co2.csv"))

(defn datasets-crimtab
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/crimtab.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/crimtab.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/crimtab.csv"))

(defn datasets-discoveries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/discoveries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/discoveries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/discoveries.csv"))

(defn datasets-DNase
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/DNase.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/DNase.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/DNase.csv"))

(defn datasets-esoph
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/esoph.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/esoph.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/esoph.csv"))

(defn datasets-euro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/euro.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/euro.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/euro.csv"))

(defn datasets-euro.crosseuro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/euro.cross.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/euro.cross.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/euro.cross.csv"))

(defn datasets-eurodist
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/eurodist.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/eurodist.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/eurodist.csv"))

(defn datasets-EuStockMarkets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/EuStockMarkets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/EuStockMarkets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/EuStockMarkets.csv"))

(defn datasets-faithful
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/faithful.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/faithful.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/faithful.csv"))

(defn datasets-fdeathsUKLungDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/fdeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/fdeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/fdeaths.csv"))

(defn datasets-Formaldehyde
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Formaldehyde.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Formaldehyde.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Formaldehyde.csv"))

(defn datasets-freeny
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/freeny.csv"))

(defn datasets-freeny.xfreeny
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.x.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.x.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/freeny.x.csv"))

(defn datasets-freeny.yfreeny
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.y.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/freeny.y.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/freeny.y.csv"))

(defn datasets-HairEyeColor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/HairEyeColor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/HairEyeColor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/HairEyeColor.csv"))

(defn datasets-Harman23.cor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Harman23.cor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Harman23.cor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Harman23.cor.csv"))

(defn datasets-Harman74.cor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Harman74.cor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Harman74.cor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Harman74.cor.csv"))

(defn datasets-Indometh
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Indometh.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Indometh.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Indometh.csv"))

(defn datasets-infert
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/infert.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/infert.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/infert.csv"))

(defn datasets-InsectSprays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/InsectSprays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/InsectSprays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/InsectSprays.csv"))

(defn datasets-iris
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris.csv"))

(defn datasets-iris3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/iris3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/iris3.csv"))

(defn datasets-islands
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/islands.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/islands.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/islands.csv"))

(defn datasets-JohnsonJohnson
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/JohnsonJohnson.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/JohnsonJohnson.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/JohnsonJohnson.csv"))

(defn datasets-LakeHuron
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/LakeHuron.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/LakeHuron.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/LakeHuron.csv"))

(defn datasets-ldeathsUKLungDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ldeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ldeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ldeaths.csv"))

(defn datasets-lh
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/lh.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/lh.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lh.csv"))

(defn datasets-LifeCycleSavings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/LifeCycleSavings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/LifeCycleSavings.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/LifeCycleSavings.csv"))

(defn datasets-Loblolly
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Loblolly.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Loblolly.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Loblolly.csv"))

(defn datasets-longley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/longley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/longley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/longley.csv"))

(defn datasets-lynx
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/lynx.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/lynx.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/lynx.csv"))

(defn datasets-mdeathsUKLungDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/mdeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/mdeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mdeaths.csv"))

(defn datasets-morley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/morley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/morley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/morley.csv"))

(defn datasets-mtcars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/mtcars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/mtcars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/mtcars.csv"))

(defn datasets-nhtemp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/nhtemp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/nhtemp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/nhtemp.csv"))

(defn datasets-Nile
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Nile.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Nile.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Nile.csv"))

(defn datasets-nottem
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/nottem.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/nottem.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/nottem.csv"))

(defn datasets-npk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/npk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/npk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/npk.csv"))

(defn datasets-occupationalStatus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/occupationalStatus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/occupationalStatus.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/occupationalStatus.csv"))

(defn datasets-Orange
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Orange.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Orange.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Orange.csv"))

(defn datasets-OrchardSprays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/OrchardSprays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/OrchardSprays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/OrchardSprays.csv"))

(defn datasets-PlantGrowth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/PlantGrowth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/PlantGrowth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/PlantGrowth.csv"))

(defn datasets-precip
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/precip.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/precip.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/precip.csv"))

(defn datasets-presidents
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/presidents.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/presidents.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/presidents.csv"))

(defn datasets-pressure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/pressure.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/pressure.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/pressure.csv"))

(defn datasets-Puromycin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Puromycin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Puromycin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Puromycin.csv"))

(defn datasets-quakes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/quakes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/quakes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/quakes.csv"))

(defn datasets-randu
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/randu.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/randu.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/randu.csv"))

(defn datasets-rivers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/rivers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/rivers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/rivers.csv"))

(defn datasets-rock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/rock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/rock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/rock.csv"))

(defn datasets-Seatbelts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Seatbelts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Seatbelts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Seatbelts.csv"))

(defn datasets-sleep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sleep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sleep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sleep.csv"))

(defn datasets-stack.lossstackloss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stack.loss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stack.loss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/stack.loss.csv"))

(defn datasets-stack.xstackloss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stack.x.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stack.x.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/stack.x.csv"))

(defn datasets-stackloss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stackloss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/stackloss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/stackloss.csv"))

(defn datasets-state.abbstate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.abb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.abb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.abb.csv"))

(defn datasets-state.areastate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.area.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.area.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.area.csv"))

(defn datasets-state.centerstate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.center.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.center.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.center.csv"))

(defn datasets-state.divisionstate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.division.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.division.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.division.csv"))

(defn datasets-state.namestate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.name.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.name.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.name.csv"))

(defn datasets-state.regionstate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.region.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.region.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.region.csv"))

(defn datasets-state.x77state
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.x77.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/state.x77.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/state.x77.csv"))

(defn datasets-sunspot.month
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspot.month.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspot.month.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.month.csv"))

(defn datasets-sunspot.year
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspot.year.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspot.year.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv"))

(defn datasets-sunspots
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspots.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/sunspots.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspots.csv"))

(defn datasets-swiss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/swiss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/swiss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/swiss.csv"))

(defn datasets-Theoph
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Theoph.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Theoph.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Theoph.csv"))

(defn datasets-Titanic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Titanic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/Titanic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/Titanic.csv"))

(defn datasets-ToothGrowth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ToothGrowth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/ToothGrowth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/ToothGrowth.csv"))

(defn datasets-treering
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/treering.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/treering.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/treering.csv"))

(defn datasets-trees
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/trees.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/trees.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/trees.csv"))

(defn datasets-UCBAdmissions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UCBAdmissions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UCBAdmissions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/UCBAdmissions.csv"))

(defn datasets-UKDriverDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UKDriverDeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UKDriverDeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/UKDriverDeaths.csv"))

(defn datasets-UKgas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UKgas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UKgas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/UKgas.csv"))

(defn datasets-USAccDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USAccDeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USAccDeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USAccDeaths.csv"))

(defn datasets-USArrests
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USArrests.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USArrests.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USArrests.csv"))

(defn datasets-UScitiesD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UScitiesD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/UScitiesD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/UScitiesD.csv"))

(defn datasets-USJudgeRatings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USJudgeRatings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USJudgeRatings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USJudgeRatings.csv"))

(defn datasets-USPersonalExpenditure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USPersonalExpenditure.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/USPersonalExpenditure.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/USPersonalExpenditure.csv"))

(defn datasets-uspop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/uspop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/uspop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/uspop.csv"))

(defn datasets-VADeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/VADeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/VADeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/VADeaths.csv"))

(defn datasets-volcano
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/volcano.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/volcano.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/volcano.csv"))

(defn datasets-warpbreaks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/warpbreaks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/warpbreaks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/warpbreaks.csv"))

(defn datasets-women
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/women.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/women.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/women.csv"))

(defn datasets-WorldPhones
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WorldPhones.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WorldPhones.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/WorldPhones.csv"))

(defn datasets-WWWusage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WWWusage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WWWusage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/WWWusage.csv"))

(defn dplyr-band_instruments
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_instruments.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_instruments.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dplyr/band_instruments.csv"))

(defn dplyr-band_instruments2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_instruments2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_instruments2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dplyr/band_instruments2.csv"))

(defn dplyr-band_members
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_members.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/band_members.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dplyr/band_members.csv"))

(defn dplyr-starwars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/starwars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/starwars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dplyr/starwars.csv"))

(defn dplyr-storms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/storms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dplyr/storms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dplyr/storms.csv"))

(defn dragracer-rpdr_contep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_contep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_contep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dragracer/rpdr_contep.csv"))

(defn dragracer-rpdr_contestants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_contestants.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_contestants.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dragracer/rpdr_contestants.csv"))

(defn dragracer-rpdr_ep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_ep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dragracer/rpdr_ep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dragracer/rpdr_ep.csv"))

(defn drc-acidiq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/acidiq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/acidiq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/acidiq.csv"))

(defn drc-algae
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/algae.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/algae.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/algae.csv"))

(defn drc-auxins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/auxins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/auxins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/auxins.csv"))

(defn drc-chickweed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/chickweed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/chickweed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/chickweed.csv"))

(defn drc-chickweed0
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/chickweed0.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/chickweed0.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/chickweed0.csv"))

(defn drc-daphnids
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/daphnids.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/daphnids.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/daphnids.csv"))

(defn drc-decontaminants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/decontaminants.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/decontaminants.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/decontaminants.csv"))

(defn drc-deguelin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/deguelin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/deguelin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/deguelin.csv"))

(defn drc-earthworms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/earthworms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/earthworms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/earthworms.csv"))

(defn drc-etmotc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/etmotc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/etmotc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/etmotc.csv"))

(defn drc-finney71
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/finney71.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/finney71.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/finney71.csv"))

(defn drc-G.aparine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/G.aparine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/G.aparine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/G.aparine.csv"))

(defn drc-germination
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/germination.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/germination.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/germination.csv"))

(defn drc-glymet
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/glymet.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/glymet.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/glymet.csv"))

(defn drc-H.virescens
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/H.virescens.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/H.virescens.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/H.virescens.csv"))

(defn drc-heartrate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/heartrate.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/heartrate.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/heartrate.csv"))

(defn drc-leaflength
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/leaflength.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/leaflength.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/leaflength.csv"))

(defn drc-lepidium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/lepidium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/lepidium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/lepidium.csv"))

(defn drc-lettuce
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/lettuce.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/lettuce.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/lettuce.csv"))

(defn drc-M.bahia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/M.bahia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/M.bahia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/M.bahia.csv"))

(defn drc-mecter
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/mecter.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/mecter.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/mecter.csv"))

(defn drc-metals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/metals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/metals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/metals.csv"))

(defn drc-methionine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/methionine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/methionine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/methionine.csv"))

(defn drc-nasturtium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/nasturtium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/nasturtium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/nasturtium.csv"))

(defn drc-O.mykiss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/O.mykiss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/O.mykiss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/O.mykiss.csv"))

(defn drc-P.promelas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/P.promelas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/P.promelas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/P.promelas.csv"))

(defn drc-RScompetition
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/RScompetition.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/RScompetition.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/RScompetition.csv"))

(defn drc-ryegrass
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/ryegrass.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/ryegrass.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/ryegrass.csv"))

(defn drc-S.alba
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/S.alba.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/S.alba.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/S.alba.csv"))

(defn drc-S.capricornutum
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/S.capricornutum.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/S.capricornutum.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/S.capricornutum.csv"))

(defn drc-secalonic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/secalonic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/secalonic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/secalonic.csv"))

(defn drc-selenium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/selenium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/selenium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/selenium.csv"))

(defn drc-spinach
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/spinach.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/spinach.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/spinach.csv"))

(defn drc-terbuthylazin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/terbuthylazin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/terbuthylazin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/terbuthylazin.csv"))

(defn drc-vinclozolin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/drc/vinclozolin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/drc/vinclozolin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/drc/vinclozolin.csv"))

(defn dslabs-admissions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/admissions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/admissions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/admissions.csv"))

(defn dslabs-brca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/brca.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/brca.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/brca.csv"))

(defn dslabs-brexit_polls
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/brexit_polls.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/brexit_polls.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/brexit_polls.csv"))

(defn dslabs-death_prob
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/death_prob.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/death_prob.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/death_prob.csv"))

(defn dslabs-divorce_margarine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/divorce_margarine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/divorce_margarine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/divorce_margarine.csv"))

(defn dslabs-gapminder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/gapminder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/gapminder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/gapminder.csv"))

(defn dslabs-greenhouse_gases
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/greenhouse_gases.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/greenhouse_gases.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/greenhouse_gases.csv"))

(defn dslabs-heights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/heights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/heights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/heights.csv"))

(defn dslabs-historic_co2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/historic_co2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/historic_co2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/historic_co2.csv"))

(defn dslabs-mice_weightsmice_weigths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/mice_weights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/mice_weights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/mice_weights.csv"))

(defn dslabs-movielens
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/movielens.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/movielens.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/movielens.csv"))

(defn dslabs-murders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/murders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/murders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/murders.csv"))

(defn dslabs-na_example
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/na_example.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/na_example.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/na_example.csv"))

(defn dslabs-nyc_regents_scores
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/nyc_regents_scores.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/nyc_regents_scores.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/nyc_regents_scores.csv"))

(defn dslabs-oecdgapminder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/oecd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/oecd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/oecd.csv"))

(defn dslabs-olive
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/olive.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/olive.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/olive.csv"))

(defn dslabs-opecgapminder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/opec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/opec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/opec.csv"))

(defn dslabs-outlier_example
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/outlier_example.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/outlier_example.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/outlier_example.csv"))

(defn dslabs-polls_2008
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/polls_2008.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/polls_2008.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/polls_2008.csv"))

(defn dslabs-polls_us_election_2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/polls_us_election_2016.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/polls_us_election_2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/polls_us_election_2016.csv"))

(defn dslabs-pr_death_countspr-death-counts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/pr_death_counts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/pr_death_counts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/pr_death_counts.csv"))

(defn dslabs-raw_data_research_funding_rates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/raw_data_research_funding_rates.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/raw_data_research_funding_rates.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/raw_data_research_funding_rates.csv"))

(defn dslabs-reported_heights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/reported_heights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/reported_heights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/reported_heights.csv"))

(defn dslabs-research_funding_rates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/research_funding_rates.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/research_funding_rates.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/research_funding_rates.csv"))

(defn dslabs-results_us_election_2016polls_us_election_2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/results_us_election_2016.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/results_us_election_2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/results_us_election_2016.csv"))

(defn dslabs-sentiment_countstrump_tweets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/sentiment_counts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/sentiment_counts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/sentiment_counts.csv"))

(defn dslabs-stars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/stars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/stars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/stars.csv"))

(defn dslabs-temp_carbon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/temp_carbon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/temp_carbon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/temp_carbon.csv"))

(defn dslabs-tissue_gene_expression
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/tissue_gene_expression.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/tissue_gene_expression.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/tissue_gene_expression.csv"))

(defn dslabs-trump_tweets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/trump_tweets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/trump_tweets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/trump_tweets.csv"))

(defn dslabs-us_contagious_diseases
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/us_contagious_diseases.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/dslabs/us_contagious_diseases.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/dslabs/us_contagious_diseases.csv"))

(defn Ecdat-Accident
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Accident.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Accident.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Accident.csv"))

(defn Ecdat-AccountantsAuditorsPct
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/AccountantsAuditorsPct.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/AccountantsAuditorsPct.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/AccountantsAuditorsPct.csv"))

(defn Ecdat-Airline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Airline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Airline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Airline.csv"))

(defn Ecdat-Airq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Airq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Airq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Airq.csv"))

(defn Ecdat-bankingCrises
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/bankingCrises.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/bankingCrises.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/bankingCrises.csv"))

(defn Ecdat-Benefits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Benefits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Benefits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Benefits.csv"))

(defn Ecdat-Bids
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Bids.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Bids.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Bids.csv"))

(defn Ecdat-breaches
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/breaches.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/breaches.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/breaches.csv"))

(defn Ecdat-BudgetFood
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetFood.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetFood.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/BudgetFood.csv"))

(defn Ecdat-BudgetItaly
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetItaly.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetItaly.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/BudgetItaly.csv"))

(defn Ecdat-BudgetUK
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetUK.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/BudgetUK.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/BudgetUK.csv"))

(defn Ecdat-Bwages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Bwages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Bwages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Bwages.csv"))

(defn Ecdat-Capm
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Capm.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Capm.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Capm.csv"))

(defn Ecdat-Car
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Car.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Car.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Car.csv"))

(defn Ecdat-Caschool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Caschool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Caschool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Caschool.csv"))

(defn Ecdat-Catsup
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Catsup.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Catsup.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Catsup.csv"))

(defn Ecdat-Cigar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cigar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cigar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Cigar.csv"))

(defn Ecdat-Cigarette
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cigarette.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cigarette.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Cigarette.csv"))

(defn Ecdat-Clothing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Clothing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Clothing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Clothing.csv"))

(defn Ecdat-Computers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Computers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Computers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Computers.csv"))

(defn Ecdat-Consumption
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Consumption.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Consumption.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Consumption.csv"))

(defn Ecdat-coolingFromNuclearWar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/coolingFromNuclearWar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/coolingFromNuclearWar.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/coolingFromNuclearWar.csv"))

(defn Ecdat-CPSch3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CPSch3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CPSch3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/CPSch3.csv"))

(defn Ecdat-Cracker
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cracker.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Cracker.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Cracker.csv"))

(defn Ecdat-CRANpackages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRANpackages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRANpackages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/CRANpackages.csv"))

(defn Ecdat-Crime
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Crime.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Crime.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Crime.csv"))

(defn Ecdat-CRSPday
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRSPday.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRSPday.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/CRSPday.csv"))

(defn Ecdat-CRSPmon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRSPmon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/CRSPmon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/CRSPmon.csv"))

(defn Ecdat-Diamond
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Diamond.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Diamond.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Diamond.csv"))

(defn Ecdat-DM
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DM.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DM.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/DM.csv"))

(defn Ecdat-Doctor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Doctor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Doctor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Doctor.csv"))

(defn Ecdat-DoctorAUS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DoctorAUS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DoctorAUS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/DoctorAUS.csv"))

(defn Ecdat-DoctorContacts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DoctorContacts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/DoctorContacts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/DoctorContacts.csv"))

(defn Ecdat-Earnings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Earnings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Earnings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Earnings.csv"))

(defn Ecdat-Electricity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Electricity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Electricity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Electricity.csv"))

(defn Ecdat-Fair
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fair.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fair.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Fair.csv"))

(defn Ecdat-Fatality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fatality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fatality.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Fatality.csv"))

(defn Ecdat-Fishing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fishing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Fishing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Fishing.csv"))

(defn Ecdat-Forward
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Forward.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Forward.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Forward.csv"))

(defn Ecdat-FriendFoe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/FriendFoe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/FriendFoe.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/FriendFoe.csv"))

(defn Ecdat-Garch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Garch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Garch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Garch.csv"))

(defn Ecdat-Gasoline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Gasoline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Gasoline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Gasoline.csv"))

(defn Ecdat-Griliches
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Griliches.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Griliches.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Griliches.csv"))

(defn Ecdat-Grunfeld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Grunfeld.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Grunfeld.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Grunfeld.csv"))

(defn Ecdat-HC
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HC.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HC.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/HC.csv"))

(defn Ecdat-Hdma
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hdma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hdma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Hdma.csv"))

(defn Ecdat-Heating
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Heating.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Heating.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Heating.csv"))

(defn Ecdat-Hedonic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hedonic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hedonic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Hedonic.csv"))

(defn Ecdat-HHSCyberSecurityBreaches
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HHSCyberSecurityBreaches.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HHSCyberSecurityBreaches.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/HHSCyberSecurityBreaches.csv"))

(defn Ecdat-HI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/HI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/HI.csv"))

(defn Ecdat-Hmda
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hmda.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hmda.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Hmda.csv"))

(defn Ecdat-Housing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Housing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Housing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv"))

(defn Ecdat-Hstarts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hstarts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Hstarts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Hstarts.csv"))

(defn Ecdat-Icecream
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Icecream.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Icecream.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Icecream.csv"))

(defn Ecdat-incidents.byCountryYr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/incidents.byCountryYr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/incidents.byCountryYr.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/incidents.byCountryYr.csv"))

(defn Ecdat-incomeInequality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/incomeInequality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/incomeInequality.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/incomeInequality.csv"))

(defn Ecdat-IncomeUK
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/IncomeUK.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/IncomeUK.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/IncomeUK.csv"))

(defn Ecdat-Irates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Irates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Irates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Irates.csv"))

(defn Ecdat-Journals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Journals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Journals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Journals.csv"))

(defn Ecdat-Kakadu
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Kakadu.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Kakadu.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Kakadu.csv"))

(defn Ecdat-Ketchup
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Ketchup.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Ketchup.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Ketchup.csv"))

(defn Ecdat-Klein
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Klein.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Klein.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Klein.csv"))

(defn Ecdat-LaborSupply
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/LaborSupply.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/LaborSupply.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/LaborSupply.csv"))

(defn Ecdat-Labour
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Labour.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Labour.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Labour.csv"))

(defn Ecdat-Longley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Longley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Longley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Longley.csv"))

(defn Ecdat-LT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/LT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/LT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/LT.csv"))

(defn Ecdat-Macrodat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Macrodat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Macrodat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Macrodat.csv"))

(defn Ecdat-Males
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Males.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Males.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Males.csv"))

(defn Ecdat-ManufCost
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/ManufCost.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/ManufCost.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ManufCost.csv"))

(defn Ecdat-Mathlevel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mathlevel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mathlevel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mathlevel.csv"))

(defn Ecdat-MCAS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MCAS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MCAS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/MCAS.csv"))

(defn Ecdat-MedExp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MedExp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MedExp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/MedExp.csv"))

(defn Ecdat-Metal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Metal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Metal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Metal.csv"))

(defn Ecdat-Mishkin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mishkin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mishkin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mishkin.csv"))

(defn Ecdat-Mode
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mode.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mode.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mode.csv"))

(defn Ecdat-ModeChoice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/ModeChoice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/ModeChoice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/ModeChoice.csv"))

(defn Ecdat-Mofa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mofa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mofa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mofa.csv"))

(defn Ecdat-Money
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Money.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Money.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Money.csv"))

(defn Ecdat-MoneyUS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MoneyUS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MoneyUS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/MoneyUS.csv"))

(defn Ecdat-Mpyr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mpyr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mpyr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mpyr.csv"))

(defn Ecdat-Mroz
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mroz.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Mroz.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Mroz.csv"))

(defn Ecdat-MunExp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MunExp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MunExp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/MunExp.csv"))

(defn Ecdat-MW
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MW.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/MW.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/MW.csv"))

(defn Ecdat-NaturalPark
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/NaturalPark.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/NaturalPark.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/NaturalPark.csv"))

(defn Ecdat-Nerlove
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Nerlove.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Nerlove.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Nerlove.csv"))

(defn Ecdat-nkill.byCountryYr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nkill.byCountryYr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nkill.byCountryYr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/nkill.byCountryYr.csv"))

(defn Ecdat-nonEnglishNames
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nonEnglishNames.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nonEnglishNames.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/nonEnglishNames.csv"))

(defn Ecdat-nuclearWeaponStates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nuclearWeaponStates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/nuclearWeaponStates.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/nuclearWeaponStates.csv"))

(defn Ecdat-OCC1950
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/OCC1950.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/OCC1950.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/OCC1950.csv"))

(defn Ecdat-OFP
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/OFP.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/OFP.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/OFP.csv"))

(defn Ecdat-Oil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Oil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Oil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Oil.csv"))

(defn Ecdat-Orange
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Orange.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Orange.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Orange.csv"))

(defn Ecdat-Participation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Participation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Participation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Participation.csv"))

(defn Ecdat-PatentsHGH
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PatentsHGH.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PatentsHGH.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/PatentsHGH.csv"))

(defn Ecdat-PatentsRD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PatentsRD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PatentsRD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/PatentsRD.csv"))

(defn Ecdat-PE
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PE.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PE.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/PE.csv"))

(defn Ecdat-politicalKnowledge
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/politicalKnowledge.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/politicalKnowledge.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/politicalKnowledge.csv"))

(defn Ecdat-Pound
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Pound.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Pound.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Pound.csv"))

(defn Ecdat-PPP
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PPP.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PPP.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/PPP.csv"))

(defn Ecdat-Pricing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Pricing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Pricing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Pricing.csv"))

(defn Ecdat-Produc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Produc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Produc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Produc.csv"))

(defn Ecdat-PSID
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PSID.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/PSID.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/PSID.csv"))

(defn Ecdat-RetSchool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/RetSchool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/RetSchool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/RetSchool.csv"))

(defn Ecdat-Schooling
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Schooling.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Schooling.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Schooling.csv"))

(defn Ecdat-Solow
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Solow.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Solow.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Solow.csv"))

(defn Ecdat-Somerville
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Somerville.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Somerville.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Somerville.csv"))

(defn Ecdat-SP500
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SP500.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SP500.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/SP500.csv"))

(defn Ecdat-Star
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Star.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Star.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Star.csv"))

(defn Ecdat-Strike
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Strike.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Strike.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Strike.csv"))

(defn Ecdat-StrikeDur
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/StrikeDur.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/StrikeDur.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/StrikeDur.csv"))

(defn Ecdat-StrikeNb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/StrikeNb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/StrikeNb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/StrikeNb.csv"))

(defn Ecdat-SumHes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SumHes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/SumHes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/SumHes.csv"))

(defn Ecdat-Tbrate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tbrate.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tbrate.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Tbrate.csv"))

(defn Ecdat-terrorism
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/terrorism.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/terrorism.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/terrorism.csv"))

(defn Ecdat-Tobacco
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tobacco.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tobacco.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Tobacco.csv"))

(defn Ecdat-Train
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Train.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Train.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Train.csv"))

(defn Ecdat-TranspEq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/TranspEq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/TranspEq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/TranspEq.csv"))

(defn Ecdat-Treatment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Treatment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Treatment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Treatment.csv"))

(defn Ecdat-Tuna
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tuna.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Tuna.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Tuna.csv"))

(defn Ecdat-UnempDur
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/UnempDur.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/UnempDur.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/UnempDur.csv"))

(defn Ecdat-Unemployment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Unemployment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Unemployment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Unemployment.csv"))

(defn Ecdat-University
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/University.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/University.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/University.csv"))

(defn Ecdat-USclassifiedDocuments
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USclassifiedDocuments.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USclassifiedDocuments.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USclassifiedDocuments.csv"))

(defn Ecdat-USFinanceIndustry
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USFinanceIndustry.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USFinanceIndustry.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USFinanceIndustry.csv"))

(defn Ecdat-USGDPpresidents
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USGDPpresidents.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USGDPpresidents.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USGDPpresidents.csv"))

(defn Ecdat-USincarcerations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USincarcerations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USincarcerations.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USincarcerations.csv"))

(defn Ecdat-USnewspapers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USnewspapers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USnewspapers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USnewspapers.csv"))

(defn Ecdat-USPS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USPS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USPS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USPS.csv"))

(defn Ecdat-USstateAbbreviations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USstateAbbreviations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/USstateAbbreviations.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/USstateAbbreviations.csv"))

(defn Ecdat-UStaxWords
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/UStaxWords.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/UStaxWords.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/UStaxWords.csv"))

(defn Ecdat-VietNamH
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/VietNamH.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/VietNamH.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/VietNamH.csv"))

(defn Ecdat-VietNamI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/VietNamI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/VietNamI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/VietNamI.csv"))

(defn Ecdat-Wages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Wages.csv"))

(defn Ecdat-Wages1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Wages1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Wages1.csv"))

(defn Ecdat-Workinghours
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Workinghours.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Workinghours.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Workinghours.csv"))

(defn Ecdat-Yen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Yen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Yen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Yen.csv"))

(defn Ecdat-Yogurt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Yogurt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Ecdat/Yogurt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Yogurt.csv"))

(defn evir-bmw
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/bmw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/bmw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/bmw.csv"))

(defn evir-danish
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/danish.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/danish.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/danish.csv"))

(defn evir-nidd.annual
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/nidd.annual.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/nidd.annual.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/nidd.annual.csv"))

(defn evir-nidd.thresh
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/nidd.thresh.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/nidd.thresh.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/nidd.thresh.csv"))

(defn evir-siemens
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/siemens.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/siemens.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/siemens.csv"))

(defn evir-sp.raw
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/sp.raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/sp.raw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/sp.raw.csv"))

(defn evir-spto87
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/evir/spto87.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/evir/spto87.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/evir/spto87.csv"))

(defn forecast-gas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/forecast/gas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/forecast/gas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/forecast/gas.csv"))

(defn forecast-gold
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/forecast/gold.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/forecast/gold.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/forecast/gold.csv"))

(defn forecast-taylor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/forecast/taylor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/forecast/taylor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/forecast/taylor.csv"))

(defn forecast-wineind
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/forecast/wineind.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/forecast/wineind.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/forecast/wineind.csv"))

(defn forecast-woolyrnq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/forecast/woolyrnq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/forecast/woolyrnq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/forecast/woolyrnq.csv"))

(defn fpp2-a10
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/a10.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/a10.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/a10.csv"))

(defn fpp2-arrivals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/arrivals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/arrivals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/arrivals.csv"))

(defn fpp2-ausair
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/ausair.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/ausair.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/ausair.csv"))

(defn fpp2-ausbeer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/ausbeer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/ausbeer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/ausbeer.csv"))

(defn fpp2-auscafe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/auscafe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/auscafe.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/auscafe.csv"))

(defn fpp2-austa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/austa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/austa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/austa.csv"))

(defn fpp2-austourists
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/austourists.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/austourists.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/austourists.csv"))

(defn fpp2-calls
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/calls.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/calls.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/calls.csv"))

(defn fpp2-debitcards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/debitcards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/debitcards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/debitcards.csv"))

(defn fpp2-departures
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/departures.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/departures.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/departures.csv"))

(defn fpp2-elecdaily
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdaily.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdaily.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecdaily.csv"))

(defn fpp2-elecdemand
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdemand.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecdemand.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecdemand.csv"))

(defn fpp2-elecequip
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecequip.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecequip.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecequip.csv"))

(defn fpp2-elecsales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecsales.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/elecsales.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecsales.csv"))

(defn fpp2-euretail
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/euretail.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/euretail.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/euretail.csv"))

(defn fpp2-gasoline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/gasoline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/gasoline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/gasoline.csv"))

(defn fpp2-goog
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/goog.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/goog.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/goog.csv"))

(defn fpp2-goog200
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/goog200.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/goog200.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/goog200.csv"))

(defn fpp2-guinearice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/guinearice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/guinearice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/guinearice.csv"))

(defn fpp2-h02
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/h02.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/h02.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/h02.csv"))

(defn fpp2-hyndsight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/hyndsight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/hyndsight.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/hyndsight.csv"))

(defn fpp2-insurance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/insurance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/insurance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/insurance.csv"))

(defn fpp2-livestock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/livestock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/livestock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/livestock.csv"))

(defn fpp2-marathon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/marathon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/marathon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/marathon.csv"))

(defn fpp2-maxtemp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/maxtemp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/maxtemp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/maxtemp.csv"))

(defn fpp2-melsyd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/melsyd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/melsyd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/melsyd.csv"))

(defn fpp2-mens400
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/mens400.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/mens400.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/mens400.csv"))

(defn fpp2-oil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/oil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/oil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/oil.csv"))

(defn fpp2-prison
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/prison.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/prison.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/prison.csv"))

(defn fpp2-prisonLF
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/prisonLF.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/prisonLF.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/prisonLF.csv"))

(defn fpp2-qauselec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qauselec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qauselec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/qauselec.csv"))

(defn fpp2-qcement
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qcement.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qcement.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/qcement.csv"))

(defn fpp2-qgas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qgas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/qgas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/qgas.csv"))

(defn fpp2-sunspotarea
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/sunspotarea.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/sunspotarea.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/sunspotarea.csv"))

(defn fpp2-uschange
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/uschange.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/uschange.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/uschange.csv"))

(defn fpp2-usmelec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/usmelec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/usmelec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/usmelec.csv"))

(defn fpp2-visnights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/visnights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/visnights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/visnights.csv"))

(defn fpp2-wmurders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/wmurders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp2/wmurders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/wmurders.csv"))

(defn fpp3-aus_accommodation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_accommodation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_accommodation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_accommodation.csv"))

(defn fpp3-aus_airpassengers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_airpassengers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_airpassengers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_airpassengers.csv"))

(defn fpp3-aus_arrivals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_arrivals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_arrivals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_arrivals.csv"))

(defn fpp3-aus_births
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_births.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_births.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_births.csv"))

(defn fpp3-aus_fertility
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_fertility.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_fertility.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_fertility.csv"))

(defn fpp3-aus_inbound
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_inbound.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_inbound.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_inbound.csv"))

(defn fpp3-aus_migration
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_migration.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_migration.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_migration.csv"))

(defn fpp3-aus_mortality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_mortality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_mortality.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_mortality.csv"))

(defn fpp3-aus_outbound
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_outbound.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_outbound.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_outbound.csv"))

(defn fpp3-aus_tobacco
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_tobacco.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_tobacco.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_tobacco.csv"))

(defn fpp3-aus_vehicle_sales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_vehicle_sales.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/aus_vehicle_sales.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/aus_vehicle_sales.csv"))

(defn fpp3-bank_calls
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/bank_calls.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/bank_calls.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/bank_calls.csv"))

(defn fpp3-boston_marathon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/boston_marathon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/boston_marathon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/boston_marathon.csv"))

(defn fpp3-canadian_gas
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/canadian_gas.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/canadian_gas.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/canadian_gas.csv"))

(defn fpp3-guinea_rice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/guinea_rice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/guinea_rice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/guinea_rice.csv"))

(defn fpp3-insurance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/insurance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/insurance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/insurance.csv"))

(defn fpp3-melb_walkers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/melb_walkers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/melb_walkers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/melb_walkers.csv"))

(defn fpp3-nsw_offences
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/nsw_offences.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/nsw_offences.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/nsw_offences.csv"))

(defn fpp3-ny_childcare
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/ny_childcare.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/ny_childcare.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/ny_childcare.csv"))

(defn fpp3-otexts_views
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/otexts_views.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/otexts_views.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/otexts_views.csv"))

(defn fpp3-prices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/prices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/prices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/prices.csv"))

(defn fpp3-souvenirs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/souvenirs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/souvenirs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/souvenirs.csv"))

(defn fpp3-us_change
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_change.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_change.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/us_change.csv"))

(defn fpp3-us_employment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_employment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_employment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/us_employment.csv"))

(defn fpp3-us_gasoline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_gasoline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/fpp3/us_gasoline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/fpp3/us_gasoline.csv"))

(defn gap-hg18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg18.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg18.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gap/hg18.csv"))

(defn gap-hg19
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg19.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg19.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gap/hg19.csv"))

(defn gap-hg38
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg38.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gap/hg38.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gap/hg38.csv"))

(defn gapminder-continent_colors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/continent_colors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/continent_colors.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/gapminder/continent_colors.csv"))

(defn gapminder-country_codes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/country_codes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/country_codes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gapminder/country_codes.csv"))

(defn gapminder-country_colors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/country_colors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/country_colors.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gapminder/country_colors.csv"))

(defn gapminder-gapminder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/gapminder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/gapminder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gapminder/gapminder.csv"))

(defn gapminder-gapminder_unfiltered
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/gapminder_unfiltered.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/gapminder/gapminder_unfiltered.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/gapminder/gapminder_unfiltered.csv"))

(defn geepack-dietox
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/dietox.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/dietox.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/dietox.csv"))

(defn geepack-koch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/koch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/koch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/koch.csv"))

(defn geepack-muscatine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/muscatine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/muscatine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/muscatine.csv"))

(defn geepack-ohio
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/ohio.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/ohio.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/ohio.csv"))

(defn geepack-respdis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/respdis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/respdis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/respdis.csv"))

(defn geepack-respiratory
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/respiratory.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/respiratory.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/respiratory.csv"))

(defn geepack-seizure
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/seizure.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/seizure.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/seizure.csv"))

(defn geepack-sitka89
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/sitka89.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/sitka89.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/sitka89.csv"))

(defn geepack-spruce
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/geepack/spruce.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/geepack/spruce.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/geepack/spruce.csv"))

(defn ggplot2-diamonds
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/diamonds.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/diamonds.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv"))

(defn ggplot2-economics
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/economics.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/economics.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/economics.csv"))

(defn ggplot2-economics_long
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/economics_long.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/economics_long.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/economics_long.csv"))

(defn ggplot2-faithfuld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/faithfuld.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/faithfuld.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/faithfuld.csv"))

(defn ggplot2-luv_colours
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/luv_colours.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/luv_colours.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/luv_colours.csv"))

(defn ggplot2-midwest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/midwest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/midwest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/midwest.csv"))

(defn ggplot2-mpg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/mpg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/mpg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/mpg.csv"))

(defn ggplot2-msleep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/msleep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/msleep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/msleep.csv"))

(defn ggplot2-presidential
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/presidential.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/presidential.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/presidential.csv"))

(defn ggplot2-seals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/seals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/seals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/seals.csv"))

(defn ggplot2-txhousing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/txhousing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/txhousing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/txhousing.csv"))

(defn ggplot2movies-movies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2movies/movies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2movies/movies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2movies/movies.csv"))

(defn gt-constants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/constants.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/constants.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/constants.csv"))

(defn gt-countrypops
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/countrypops.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/countrypops.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/countrypops.csv"))

(defn gt-exibble
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/exibble.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/exibble.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/exibble.csv"))

(defn gt-films
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/films.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/films.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/films.csv"))

(defn gt-gibraltar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/gibraltar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/gibraltar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/gibraltar.csv"))

(defn gt-gtcars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/gtcars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/gtcars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/gtcars.csv"))

(defn gt-illness
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/illness.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/illness.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/illness.csv"))

(defn gt-metro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/metro.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/metro.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/metro.csv"))

(defn gt-nuclides
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/nuclides.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/nuclides.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/nuclides.csv"))

(defn gt-peeps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/peeps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/peeps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/peeps.csv"))

(defn gt-photolysis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/photolysis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/photolysis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/photolysis.csv"))

(defn gt-pizzaplace
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/pizzaplace.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/pizzaplace.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/pizzaplace.csv"))

(defn gt-reactions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/reactions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/reactions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/reactions.csv"))

(defn gt-rx_addv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/rx_addv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/rx_addv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/rx_addv.csv"))

(defn gt-rx_adsl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/rx_adsl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/rx_adsl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/rx_adsl.csv"))

(defn gt-sp500
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/sp500.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/sp500.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/sp500.csv"))

(defn gt-sza
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/sza.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/sza.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/sza.csv"))

(defn gt-towny
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/gt/towny.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/gt/towny.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/gt/towny.csv"))

(defn heplots-AddHealth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/AddHealth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/AddHealth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/AddHealth.csv"))

(defn heplots-Adopted
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Adopted.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Adopted.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Adopted.csv"))

(defn heplots-Bees
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Bees.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Bees.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Bees.csv"))

(defn heplots-Diabetes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Diabetes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Diabetes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Diabetes.csv"))

(defn heplots-FootHead
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/FootHead.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/FootHead.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/FootHead.csv"))

(defn heplots-Headache
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Headache.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Headache.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Headache.csv"))

(defn heplots-Hernior
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Hernior.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Hernior.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Hernior.csv"))

(defn heplots-Iwasaki_Big_Five
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Iwasaki_Big_Five.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Iwasaki_Big_Five.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Iwasaki_Big_Five.csv"))

(defn heplots-mathscore
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/mathscore.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/mathscore.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/mathscore.csv"))

(defn heplots-MockJury
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/MockJury.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/MockJury.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/MockJury.csv"))

(defn heplots-NeuroCog
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/NeuroCog.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/NeuroCog.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/NeuroCog.csv"))

(defn heplots-NLSY
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/NLSY.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/NLSY.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/NLSY.csv"))

(defn heplots-Oslo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Oslo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Oslo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Oslo.csv"))

(defn heplots-Overdose
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Overdose.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Overdose.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Overdose.csv"))

(defn heplots-Parenting
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Parenting.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Parenting.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Parenting.csv"))

(defn heplots-peng
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/peng.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/peng.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/peng.csv"))

(defn heplots-Plastic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Plastic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Plastic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Plastic.csv"))

(defn heplots-Pottery2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Pottery2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Pottery2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Pottery2.csv"))

(defn heplots-Probe1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Probe1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Probe1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Probe1.csv"))

(defn heplots-Probe2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Probe2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Probe2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Probe2.csv"))

(defn heplots-RatWeight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/RatWeight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/RatWeight.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/RatWeight.csv"))

(defn heplots-ReactTime
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/ReactTime.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/ReactTime.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/ReactTime.csv"))

(defn heplots-Rohwer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Rohwer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Rohwer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Rohwer.csv"))

(defn heplots-RootStock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/RootStock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/RootStock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/RootStock.csv"))

(defn heplots-Sake
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Sake.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Sake.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Sake.csv"))

(defn heplots-schooldata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/schooldata.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/schooldata.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/schooldata.csv"))

(defn heplots-Skulls
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Skulls.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/Skulls.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/Skulls.csv"))

(defn heplots-SocGrades
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/SocGrades.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/SocGrades.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/SocGrades.csv"))

(defn heplots-SocialCog
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/SocialCog.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/SocialCog.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/SocialCog.csv"))

(defn heplots-TIPI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/TIPI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/TIPI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/TIPI.csv"))

(defn heplots-VocabGrowth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/VocabGrowth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/VocabGrowth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/VocabGrowth.csv"))

(defn heplots-WeightLoss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/heplots/WeightLoss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/heplots/WeightLoss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/heplots/WeightLoss.csv"))

(defn HistData-Arbuthnot
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Arbuthnot.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Arbuthnot.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Arbuthnot.csv"))

(defn HistData-Armada
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Armada.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Armada.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Armada.csv"))

(defn HistData-Bowley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Bowley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Bowley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Bowley.csv"))

(defn HistData-Breslau
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Breslau.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Breslau.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Breslau.csv"))

(defn HistData-Cavendish
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Cavendish.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Cavendish.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Cavendish.csv"))

(defn HistData-ChestSizes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ChestSizes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ChestSizes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/ChestSizes.csv"))

(defn HistData-ChestStigler
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ChestStigler.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ChestStigler.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/ChestStigler.csv"))

(defn HistData-Cholera
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Cholera.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Cholera.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Cholera.csv"))

(defn HistData-CholeraDeaths1849
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CholeraDeaths1849.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CholeraDeaths1849.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/CholeraDeaths1849.csv"))

(defn HistData-CushnyPeebles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CushnyPeebles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CushnyPeebles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/CushnyPeebles.csv"))

(defn HistData-CushnyPeeblesN
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CushnyPeeblesN.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/CushnyPeeblesN.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/CushnyPeeblesN.csv"))

(defn HistData-Dactyl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Dactyl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Dactyl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Dactyl.csv"))

(defn HistData-DrinksWages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/DrinksWages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/DrinksWages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/DrinksWages.csv"))

(defn HistData-EdgeworthDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/EdgeworthDeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/EdgeworthDeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/EdgeworthDeaths.csv"))

(defn HistData-Fingerprints
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Fingerprints.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Fingerprints.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Fingerprints.csv"))

(defn HistData-Galton
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Galton.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Galton.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Galton.csv"))

(defn HistData-GaltonFamilies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/GaltonFamilies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/GaltonFamilies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/GaltonFamilies.csv"))

(defn HistData-Guerry
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Guerry.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv"))

(defn HistData-HalleyLifeTable
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/HalleyLifeTable.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/HalleyLifeTable.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/HalleyLifeTable.csv"))

(defn HistData-Jevons
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Jevons.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Jevons.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Jevons.csv"))

(defn HistData-Langren.all
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Langren.all.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Langren.all.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Langren.all.csv"))

(defn HistData-Langren1644
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Langren1644.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Langren1644.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Langren1644.csv"))

(defn HistData-Macdonell
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Macdonell.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Macdonell.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Macdonell.csv"))

(defn HistData-MacdonellDF
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/MacdonellDF.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/MacdonellDF.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/MacdonellDF.csv"))

(defn HistData-Mayer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Mayer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Mayer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Mayer.csv"))

(defn HistData-Michelson
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Michelson.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Michelson.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Michelson.csv"))

(defn HistData-MichelsonSets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/MichelsonSets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/MichelsonSets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/MichelsonSets.csv"))

(defn HistData-Minard.cities
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.cities.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.cities.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Minard.cities.csv"))

(defn HistData-Minard.temp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.temp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.temp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Minard.temp.csv"))

(defn HistData-Minard.troops
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.troops.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Minard.troops.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Minard.troops.csv"))

(defn HistData-Nightingale
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Nightingale.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Nightingale.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Nightingale.csv"))

(defn HistData-OldMaps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/OldMaps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/OldMaps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/OldMaps.csv"))

(defn HistData-PearsonLee
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/PearsonLee.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/PearsonLee.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/PearsonLee.csv"))

(defn HistData-PolioTrials
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/PolioTrials.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/PolioTrials.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/PolioTrials.csv"))

(defn HistData-Pollen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Pollen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Pollen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Pollen.csv"))

(defn HistData-Prostitutes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Prostitutes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Prostitutes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Prostitutes.csv"))

(defn HistData-Pyx
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Pyx.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Pyx.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Pyx.csv"))

(defn HistData-Quarrels
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Quarrels.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Quarrels.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Quarrels.csv"))

(defn HistData-Saturn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Saturn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Saturn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Saturn.csv"))

(defn HistData-Snow.dates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.dates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.dates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Snow.dates.csv"))

(defn HistData-Snow.deaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.deaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.deaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Snow.deaths.csv"))

(defn HistData-Snow.deaths2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.deaths2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.deaths2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Snow.deaths2.csv"))

(defn HistData-Snow.pumps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.pumps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.pumps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Snow.pumps.csv"))

(defn HistData-Snow.streets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.streets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Snow.streets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Snow.streets.csv"))

(defn HistData-Virginis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Virginis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Virginis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Virginis.csv"))

(defn HistData-Virginis.interp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Virginis.interp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Virginis.interp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Virginis.interp.csv"))

(defn HistData-Wheat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Wheat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Wheat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Wheat.csv"))

(defn HistData-Wheat.monarchs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Wheat.monarchs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Wheat.monarchs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Wheat.monarchs.csv"))

(defn HistData-Yeast
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Yeast.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/Yeast.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Yeast.csv"))

(defn HistData-YeastD.mat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/YeastD.mat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/YeastD.mat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/YeastD.mat.csv"))

(defn HistData-ZeaMays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ZeaMays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HistData/ZeaMays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/ZeaMays.csv"))

(defn HLMdiag-ahd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/ahd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/ahd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HLMdiag/ahd.csv"))

(defn HLMdiag-autism
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/autism.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/autism.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HLMdiag/autism.csv"))

(defn HLMdiag-radon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/radon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/radon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HLMdiag/radon.csv"))

(defn HLMdiag-wages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/wages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HLMdiag/wages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HLMdiag/wages.csv"))

(defn HSAUR-agefat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/agefat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/agefat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/agefat.csv"))

(defn HSAUR-aspirin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/aspirin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/aspirin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/aspirin.csv"))

(defn HSAUR-BCG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/BCG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/BCG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/BCG.csv"))

(defn HSAUR-birthdeathrates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/birthdeathrates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/birthdeathrates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/birthdeathrates.csv"))

(defn HSAUR-bladdercancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/bladdercancer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/bladdercancer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/bladdercancer.csv"))

(defn HSAUR-BtheB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/BtheB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/BtheB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/BtheB.csv"))

(defn HSAUR-clouds
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/clouds.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/clouds.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/clouds.csv"))

(defn HSAUR-CYGOB1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/CYGOB1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/CYGOB1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/CYGOB1.csv"))

(defn HSAUR-epilepsy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/epilepsy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/epilepsy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/epilepsy.csv"))

(defn HSAUR-Forbes2000
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/Forbes2000.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/Forbes2000.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/Forbes2000.csv"))

(defn HSAUR-foster
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/foster.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/foster.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/foster.csv"))

(defn HSAUR-gardenflowers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/gardenflowers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/gardenflowers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/gardenflowers.csv"))

(defn HSAUR-GHQ
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/GHQ.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/GHQ.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/GHQ.csv"))

(defn HSAUR-heptathlon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/heptathlon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/heptathlon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/heptathlon.csv"))

(defn HSAUR-Lanza
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/Lanza.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/Lanza.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/Lanza.csv"))

(defn HSAUR-mastectomy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/mastectomy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/mastectomy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/mastectomy.csv"))

(defn HSAUR-meteo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/meteo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/meteo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/meteo.csv"))

(defn HSAUR-orallesions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/orallesions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/orallesions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/orallesions.csv"))

(defn HSAUR-phosphate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/phosphate.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/phosphate.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/phosphate.csv"))

(defn HSAUR-pistonrings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/pistonrings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/pistonrings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/pistonrings.csv"))

(defn HSAUR-planets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/planets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/planets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/planets.csv"))

(defn HSAUR-plasma
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/plasma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/plasma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/plasma.csv"))

(defn HSAUR-polyps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/polyps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/polyps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/polyps.csv"))

(defn HSAUR-polyps3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/polyps3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/polyps3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/polyps3.csv"))

(defn HSAUR-pottery
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/pottery.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/pottery.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/pottery.csv"))

(defn HSAUR-rearrests
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/rearrests.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/rearrests.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/rearrests.csv"))

(defn HSAUR-respiratory
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/respiratory.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/respiratory.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/respiratory.csv"))

(defn HSAUR-roomwidth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/roomwidth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/roomwidth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/roomwidth.csv"))

(defn HSAUR-schizophrenia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schizophrenia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schizophrenia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/schizophrenia.csv"))

(defn HSAUR-schizophrenia2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schizophrenia2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schizophrenia2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/schizophrenia2.csv"))

(defn HSAUR-schooldays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schooldays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/schooldays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/schooldays.csv"))

(defn HSAUR-skulls
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/skulls.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/skulls.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/skulls.csv"))

(defn HSAUR-smoking
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/smoking.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/smoking.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/smoking.csv"))

(defn HSAUR-students
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/students.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/students.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/students.csv"))

(defn HSAUR-suicides
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/suicides.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/suicides.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/suicides.csv"))

(defn HSAUR-toothpaste
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/toothpaste.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/toothpaste.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/toothpaste.csv"))

(defn HSAUR-voting
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/voting.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/voting.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/voting.csv"))

(defn HSAUR-water
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/water.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/water.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/water.csv"))

(defn HSAUR-watervoles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/watervoles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/watervoles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/watervoles.csv"))

(defn HSAUR-waves
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/waves.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/waves.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/waves.csv"))

(defn HSAUR-weightgain
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/weightgain.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/weightgain.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/weightgain.csv"))

(defn HSAUR-womensrole
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/womensrole.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/HSAUR/womensrole.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/HSAUR/womensrole.csv"))

(defn hwde-IndianIrish
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/hwde/IndianIrish.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/hwde/IndianIrish.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/hwde/IndianIrish.csv"))

(defn hwde-mendelABC
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/hwde/mendelABC.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/hwde/mendelABC.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/hwde/mendelABC.csv"))

(defn ISLR-Auto
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Auto.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Auto.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Auto.csv"))

(defn ISLR-Caravan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Caravan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Caravan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Caravan.csv"))

(defn ISLR-Carseats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Carseats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Carseats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Carseats.csv"))

(defn ISLR-College
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/College.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/College.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/College.csv"))

(defn ISLR-Credit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Credit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Credit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Credit.csv"))

(defn ISLR-Default
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Default.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Default.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Default.csv"))

(defn ISLR-Hitters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Hitters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Hitters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Hitters.csv"))

(defn ISLR-NCI60
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/NCI60.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/NCI60.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/NCI60.csv"))

(defn ISLR-OJ
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/OJ.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/OJ.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/OJ.csv"))

(defn ISLR-Portfolio
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Portfolio.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Portfolio.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Portfolio.csv"))

(defn ISLR-Smarket
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Smarket.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Smarket.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Smarket.csv"))

(defn ISLR-Wage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Wage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Wage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Wage.csv"))

(defn ISLR-Weekly
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Weekly.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ISLR/Weekly.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Weekly.csv"))

(defn itsadug-eeg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/itsadug/eeg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/itsadug/eeg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/itsadug/eeg.csv"))

(defn itsadug-simdat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/itsadug/simdat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/itsadug/simdat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/itsadug/simdat.csv"))

(defn KMsurv-aids
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/aids.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/aids.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/aids.csv"))

(defn KMsurv-alloauto
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/alloauto.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/alloauto.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/alloauto.csv"))

(defn KMsurv-allograft
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/allograft.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/allograft.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/allograft.csv"))

(defn KMsurv-azt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/azt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/azt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/azt.csv"))

(defn KMsurv-baboon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/baboon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/baboon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/baboon.csv"))

(defn KMsurv-bcdeter
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bcdeter.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bcdeter.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/bcdeter.csv"))

(defn KMsurv-bfeed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bfeed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bfeed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/bfeed.csv"))

(defn KMsurv-bmt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bmt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bmt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/bmt.csv"))

(defn KMsurv-bnct
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bnct.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/bnct.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/bnct.csv"))

(defn KMsurv-btrial
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/btrial.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/btrial.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/btrial.csv"))

(defn KMsurv-burn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/burn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/burn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/burn.csv"))

(defn KMsurv-channing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/channing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/channing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/channing.csv"))

(defn KMsurv-drug6mp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/drug6mp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/drug6mp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/drug6mp.csv"))

(defn KMsurv-drughiv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/drughiv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/drughiv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/drughiv.csv"))

(defn KMsurv-hodg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/hodg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/hodg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/hodg.csv"))

(defn KMsurv-kidney
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidney.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidney.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/kidney.csv"))

(defn KMsurv-kidrecurr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidrecurr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidrecurr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/kidrecurr.csv"))

(defn KMsurv-kidtran
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidtran.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/kidtran.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/kidtran.csv"))

(defn KMsurv-larynx
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/larynx.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/larynx.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/larynx.csv"))

(defn KMsurv-lung
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/lung.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/lung.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/lung.csv"))

(defn KMsurv-pneumon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/pneumon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/pneumon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/pneumon.csv"))

(defn KMsurv-psych
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/psych.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/psych.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/psych.csv"))

(defn KMsurv-rats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/rats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/rats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/rats.csv"))

(defn KMsurv-std
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/std.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/std.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/std.csv"))

(defn KMsurv-stddiag
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/stddiag.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/stddiag.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/stddiag.csv"))

(defn KMsurv-tongue
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/tongue.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/tongue.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/tongue.csv"))

(defn KMsurv-twins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/twins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/KMsurv/twins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/KMsurv/twins.csv"))

(defn lattice-barley
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/barley.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/barley.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/barley.csv"))

(defn lattice-environmental
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/environmental.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/environmental.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/environmental.csv"))

(defn lattice-ethanol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/ethanol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/ethanol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/ethanol.csv"))

(defn lattice-melanoma
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/melanoma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/melanoma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/melanoma.csv"))

(defn lattice-singer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/singer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/singer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/singer.csv"))

(defn lattice-USMortality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/USMortality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/USMortality.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/USMortality.csv"))

(defn lattice-USRegionalMortality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lattice/USRegionalMortality.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lattice/USRegionalMortality.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/lattice/USRegionalMortality.csv"))

(defn lme4-Arabidopsis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Arabidopsis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Arabidopsis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Arabidopsis.csv"))

(defn lme4-cake
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/cake.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/cake.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cake.csv"))

(defn lme4-cbpp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/cbpp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/cbpp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/cbpp.csv"))

(defn lme4-Dyestuff
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Dyestuff.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Dyestuff.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff.csv"))

(defn lme4-Dyestuff2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Dyestuff2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Dyestuff2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Dyestuff2.csv"))

(defn lme4-grouseticks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/grouseticks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/grouseticks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/grouseticks.csv"))

(defn lme4-grouseticks_agggrouseticks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/grouseticks_agg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/grouseticks_agg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/grouseticks_agg.csv"))

(defn lme4-InstEval
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/InstEval.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/InstEval.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/InstEval.csv"))

(defn lme4-Pastes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Pastes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Pastes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Pastes.csv"))

(defn lme4-Penicillin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Penicillin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/Penicillin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/Penicillin.csv"))

(defn lme4-sleepstudy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/sleepstudy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/sleepstudy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/sleepstudy.csv"))

(defn lme4-VerbAgg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/lme4/VerbAgg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/lme4/VerbAgg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/lme4/VerbAgg.csv"))

(defn MASS-abbey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/abbey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/abbey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/abbey.csv"))

(defn MASS-accdeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/accdeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/accdeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/accdeaths.csv"))

(defn MASS-Aids2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Aids2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Aids2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Aids2.csv"))

(defn MASS-Animals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Animals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Animals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Animals.csv"))

(defn MASS-anorexia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/anorexia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/anorexia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/anorexia.csv"))

(defn MASS-bacteria
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/bacteria.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/bacteria.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/bacteria.csv"))

(defn MASS-beav1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/beav1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/beav1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/beav1.csv"))

(defn MASS-beav2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/beav2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/beav2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/beav2.csv"))

(defn MASS-biopsy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/biopsy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/biopsy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/biopsy.csv"))

(defn MASS-birthwt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/birthwt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/birthwt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/birthwt.csv"))

(defn MASS-Boston
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Boston.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Boston.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Boston.csv"))

(defn MASS-cabbages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cabbages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cabbages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cabbages.csv"))

(defn MASS-caith
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/caith.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/caith.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/caith.csv"))

(defn MASS-Cars93
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Cars93.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Cars93.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Cars93.csv"))

(defn MASS-cats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cats.csv"))

(defn MASS-cement
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cement.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cement.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cement.csv"))

(defn MASS-chem
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/chem.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/chem.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/chem.csv"))

(defn MASS-coop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/coop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/coop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/coop.csv"))

(defn MASS-cpus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cpus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/cpus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/cpus.csv"))

(defn MASS-crabs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/crabs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/crabs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/crabs.csv"))

(defn MASS-Cushings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Cushings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Cushings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Cushings.csv"))

(defn MASS-DDT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/DDT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/DDT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/DDT.csv"))

(defn MASS-deaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/deaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/deaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/deaths.csv"))

(defn MASS-drivers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/drivers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/drivers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/drivers.csv"))

(defn MASS-eagles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/eagles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/eagles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/eagles.csv"))

(defn MASS-epil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/epil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/epil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/epil.csv"))

(defn MASS-farms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/farms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/farms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/farms.csv"))

(defn MASS-fgl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/fgl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/fgl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/fgl.csv"))

(defn MASS-forbes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/forbes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/forbes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/forbes.csv"))

(defn MASS-GAGurine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/GAGurine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/GAGurine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/GAGurine.csv"))

(defn MASS-galaxies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/galaxies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/galaxies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/galaxies.csv"))

(defn MASS-gehan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/gehan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/gehan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/gehan.csv"))

(defn MASS-genotype
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/genotype.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/genotype.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/genotype.csv"))

(defn MASS-geyser
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/geyser.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/geyser.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/geyser.csv"))

(defn MASS-gilgais
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/gilgais.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/gilgais.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/gilgais.csv"))

(defn MASS-hills
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/hills.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/hills.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/hills.csv"))

(defn MASS-housing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/housing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/housing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/housing.csv"))

(defn MASS-immer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/immer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/immer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/immer.csv"))

(defn MASS-Insurance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Insurance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Insurance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Insurance.csv"))

(defn MASS-leuk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/leuk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/leuk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/leuk.csv"))

(defn MASS-mammals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mammals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mammals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mammals.csv"))

(defn MASS-mcycle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mcycle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/mcycle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv"))

(defn MASS-Melanoma
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Melanoma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Melanoma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Melanoma.csv"))

(defn MASS-menarche
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/menarche.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/menarche.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/menarche.csv"))

(defn MASS-michelson
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/michelson.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/michelson.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/michelson.csv"))

(defn MASS-minn38
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/minn38.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/minn38.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/minn38.csv"))

(defn MASS-motors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/motors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/motors.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/motors.csv"))

(defn MASS-muscle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/muscle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/muscle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/muscle.csv"))

(defn MASS-newcomb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/newcomb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/newcomb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/newcomb.csv"))

(defn MASS-nlschools
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/nlschools.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/nlschools.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/nlschools.csv"))

(defn MASS-npk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/npk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/npk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/npk.csv"))

(defn MASS-npr1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/npr1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/npr1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/npr1.csv"))

(defn MASS-oats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/oats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/oats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/oats.csv"))

(defn MASS-OME
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/OME.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/OME.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/OME.csv"))

(defn MASS-painters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/painters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/painters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/painters.csv"))

(defn MASS-petrol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/petrol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/petrol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/petrol.csv"))

(defn MASS-phones
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/phones.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/phones.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/phones.csv"))

(defn MASS-Pima.te
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.te.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.te.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Pima.te.csv"))

(defn MASS-Pima.tr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.tr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.tr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Pima.tr.csv"))

(defn MASS-Pima.tr2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.tr2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Pima.tr2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Pima.tr2.csv"))

(defn MASS-quine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/quine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/quine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/quine.csv"))

(defn MASS-Rabbit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Rabbit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Rabbit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Rabbit.csv"))

(defn MASS-road
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/road.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/road.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/road.csv"))

(defn MASS-rotifer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/rotifer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/rotifer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/rotifer.csv"))

(defn MASS-Rubber
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Rubber.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Rubber.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Rubber.csv"))

(defn MASS-ships
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/ships.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/ships.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/ships.csv"))

(defn MASS-shoes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shoes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shoes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/shoes.csv"))

(defn MASS-shrimp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shrimp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shrimp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/shrimp.csv"))

(defn MASS-shuttle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shuttle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/shuttle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/shuttle.csv"))

(defn MASS-Sitka
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Sitka.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Sitka.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Sitka.csv"))

(defn MASS-Sitka89
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Sitka89.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Sitka89.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Sitka89.csv"))

(defn MASS-Skye
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Skye.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Skye.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Skye.csv"))

(defn MASS-snails
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/snails.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/snails.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/snails.csv"))

(defn MASS-SP500
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/SP500.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/SP500.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/SP500.csv"))

(defn MASS-steam
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/steam.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/steam.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/steam.csv"))

(defn MASS-stormer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/stormer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/stormer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/stormer.csv"))

(defn MASS-survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/survey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/survey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/survey.csv"))

(defn MASS-synth.te
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/synth.te.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/synth.te.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/synth.te.csv"))

(defn MASS-synth.tr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/synth.tr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/synth.tr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/synth.tr.csv"))

(defn MASS-topo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/topo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/topo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/topo.csv"))

(defn MASS-Traffic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Traffic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/Traffic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/Traffic.csv"))

(defn MASS-UScereal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/UScereal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/UScereal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/UScereal.csv"))

(defn MASS-UScrime
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/UScrime.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/UScrime.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/UScrime.csv"))

(defn MASS-VA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/VA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/VA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/VA.csv"))

(defn MASS-waders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/waders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/waders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/waders.csv"))

(defn MASS-whiteside
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/whiteside.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/whiteside.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/whiteside.csv"))

(defn MASS-wtloss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MASS/wtloss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MASS/wtloss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MASS/wtloss.csv"))

(defn MatchIt-lalonde
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/MatchIt/lalonde.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/MatchIt/lalonde.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv"))

(defn mediation-boundsdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/boundsdata.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/boundsdata.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/boundsdata.csv"))

(defn mediation-CEDdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/CEDdata.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/CEDdata.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/CEDdata.csv"))

(defn mediation-framing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/framing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/framing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/framing.csv"))

(defn mediation-jobs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/jobs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/jobs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/jobs.csv"))

(defn mediation-school
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/school.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/school.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/school.csv"))

(defn mediation-student
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mediation/student.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mediation/student.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mediation/student.csv"))

(defn medicaldata-blood_storage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/blood_storage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/blood_storage.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/blood_storage.csv"))

(defn medicaldata-covid_testing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/covid_testing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/covid_testing.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/covid_testing.csv"))

(defn medicaldata-cytomegalovirus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/cytomegalovirus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/cytomegalovirus.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/cytomegalovirus.csv"))

(defn medicaldata-esoph_ca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/esoph_ca.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/esoph_ca.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/esoph_ca.csv"))

(defn medicaldata-indo_rct
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/indo_rct.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/indo_rct.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/indo_rct.csv"))

(defn medicaldata-indometh
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/indometh.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/indometh.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/indometh.csv"))

(defn medicaldata-laryngoscope
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/laryngoscope.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/laryngoscope.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/laryngoscope.csv"))

(defn medicaldata-licorice_gargle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/licorice_gargle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/licorice_gargle.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/licorice_gargle.csv"))

(defn medicaldata-opt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/opt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/opt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/opt.csv"))

(defn medicaldata-polyps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/polyps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/polyps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/polyps.csv"))

(defn medicaldata-scurvy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/scurvy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/scurvy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/scurvy.csv"))

(defn medicaldata-smartpill
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/smartpill.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/smartpill.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/smartpill.csv"))

(defn medicaldata-strep_tb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/strep_tb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/strep_tb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/strep_tb.csv"))

(defn medicaldata-supraclavicular
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/supraclavicular.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/supraclavicular.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/supraclavicular.csv"))

(defn medicaldata-theoph
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/theoph.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/medicaldata/theoph.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/medicaldata/theoph.csv"))

(defn mi-CHAIN
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mi/CHAIN.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mi/CHAIN.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mi/CHAIN.csv"))

(defn mi-nlsyV
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mi/nlsyV.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mi/nlsyV.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mi/nlsyV.csv"))

(defn mlmRev-bdf
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/bdf.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/bdf.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/bdf.csv"))

(defn mlmRev-Chem97
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Chem97.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Chem97.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Chem97.csv"))

(defn mlmRev-Contraception
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Contraception.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Contraception.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Contraception.csv"))

(defn mlmRev-Early
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Early.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Early.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Early.csv"))

(defn mlmRev-egsingle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/egsingle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/egsingle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/egsingle.csv"))

(defn mlmRev-Exam
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Exam.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Exam.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Exam.csv"))

(defn mlmRev-Gcsemv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Gcsemv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Gcsemv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Gcsemv.csv"))

(defn mlmRev-guImmun
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/guImmun.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/guImmun.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/guImmun.csv"))

(defn mlmRev-guPrenat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/guPrenat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/guPrenat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/guPrenat.csv"))

(defn mlmRev-Hsb82
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Hsb82.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Hsb82.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Hsb82.csv"))

(defn mlmRev-Mmmec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Mmmec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Mmmec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Mmmec.csv"))

(defn mlmRev-Oxboys
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Oxboys.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Oxboys.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Oxboys.csv"))

(defn mlmRev-s3bbx
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/s3bbx.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/s3bbx.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/s3bbx.csv"))

(defn mlmRev-s3bby
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/s3bby.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/s3bby.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/s3bby.csv"))

(defn mlmRev-ScotsSec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/ScotsSec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/ScotsSec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/ScotsSec.csv"))

(defn mlmRev-Socatt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Socatt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/Socatt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/Socatt.csv"))

(defn mlmRev-star
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/star.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mlmRev/star.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mlmRev/star.csv"))

(defn modeldata-ad_data
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ad_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ad_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/ad_data.csv"))

(defn modeldata-ames
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ames.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ames.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/ames.csv"))

(defn modeldata-attrition
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/attrition.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/attrition.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/attrition.csv"))

(defn modeldata-biomass
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/biomass.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/biomass.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/biomass.csv"))

(defn modeldata-bivariate_testbivariate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_test.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_test.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/bivariate_test.csv"))

(defn modeldata-bivariate_trainbivariate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_train.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_train.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/bivariate_train.csv"))

(defn modeldata-bivariate_valbivariate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_val.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/bivariate_val.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/bivariate_val.csv"))

(defn modeldata-car_prices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/car_prices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/car_prices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/car_prices.csv"))

(defn modeldata-cat_adoption
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/cat_adoption.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/cat_adoption.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/cat_adoption.csv"))

(defn modeldata-cells
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/cells.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/cells.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/cells.csv"))

(defn modeldata-check_times
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/check_times.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/check_times.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/check_times.csv"))

(defn modeldata-chem_proc_yield
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/chem_proc_yield.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/chem_proc_yield.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/chem_proc_yield.csv"))

(defn modeldata-Chicago
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Chicago.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Chicago.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/Chicago.csv"))

(defn modeldata-concrete
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/concrete.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/concrete.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/concrete.csv"))

(defn modeldata-covers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/covers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/covers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/covers.csv"))

(defn modeldata-credit_data
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/credit_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/credit_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/credit_data.csv"))

(defn modeldata-crickets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/crickets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/crickets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/crickets.csv"))

(defn modeldata-deliveries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/deliveries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/deliveries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/deliveries.csv"))

(defn modeldata-drinks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/drinks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/drinks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/drinks.csv"))

(defn modeldata-grants_2008grants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_2008.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_2008.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/grants_2008.csv"))

(defn modeldata-grants_othergrants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_other.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_other.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/grants_other.csv"))

(defn modeldata-grants_testgrants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_test.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/grants_test.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/grants_test.csv"))

(defn modeldata-hepatic_injury_qsar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hepatic_injury_qsar.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hepatic_injury_qsar.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hepatic_injury_qsar.csv"))

(defn modeldata-hotel_rates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hotel_rates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hotel_rates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hotel_rates.csv"))

(defn modeldata-hpc_cv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hpc_cv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hpc_cv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hpc_cv.csv"))

(defn modeldata-hpc_data
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hpc_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/hpc_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/hpc_data.csv"))

(defn modeldata-ischemic_stroke
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ischemic_stroke.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/ischemic_stroke.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/ischemic_stroke.csv"))

(defn modeldata-leaf_id_flavia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/leaf_id_flavia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/leaf_id_flavia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/leaf_id_flavia.csv"))

(defn modeldata-lending_club
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/lending_club.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/lending_club.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/lending_club.csv"))

(defn modeldata-meats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/meats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/meats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/meats.csv"))

(defn modeldata-mlc_churn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/mlc_churn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/mlc_churn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/mlc_churn.csv"))

(defn modeldata-oils
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/oils.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/oils.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/oils.csv"))

(defn modeldata-parabolic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/parabolic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/parabolic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/parabolic.csv"))

(defn modeldata-pathology
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/pathology.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/pathology.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/pathology.csv"))

(defn modeldata-pd_speech
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/pd_speech.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/pd_speech.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/pd_speech.csv"))

(defn modeldata-penguins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/penguins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/penguins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/penguins.csv"))

(defn modeldata-permeability_qsar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/permeability_qsar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/permeability_qsar.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/permeability_qsar.csv"))

(defn modeldata-Sacramento
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Sacramento.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Sacramento.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/Sacramento.csv"))

(defn modeldata-scat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/scat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/scat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/scat.csv"))

(defn modeldata-Smithsonian
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Smithsonian.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/Smithsonian.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/Smithsonian.csv"))

(defn modeldata-solubility_test
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/solubility_test.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/solubility_test.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/solubility_test.csv"))

(defn modeldata-stackoverflow
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/stackoverflow.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/stackoverflow.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/stackoverflow.csv"))

(defn modeldata-stationsChicago
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/stations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/stations.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/stations.csv"))

(defn modeldata-steroidogenic_toxicity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/steroidogenic_toxicity.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/steroidogenic_toxicity.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/steroidogenic_toxicity.csv"))

(defn modeldata-tate_text
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/tate_text.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/tate_text.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/tate_text.csv"))

(defn modeldata-taxi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/taxi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/taxi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/taxi.csv"))

(defn modeldata-testing_datasmall_fine_foods
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/testing_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/testing_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/testing_data.csv"))

(defn modeldata-training_datasmall_fine_foods
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/training_data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/training_data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/training_data.csv"))

(defn modeldata-two_class_dat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/two_class_dat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/two_class_dat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/two_class_dat.csv"))

(defn modeldata-two_class_example
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/two_class_example.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/two_class_example.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/two_class_example.csv"))

(defn modeldata-wa_churn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/wa_churn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/modeldata/wa_churn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/modeldata/wa_churn.csv"))

(defn mosaicData-Alcohol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Alcohol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Alcohol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Alcohol.csv"))

(defn mosaicData-Birthdays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Birthdays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Birthdays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Birthdays.csv"))

(defn mosaicData-Births
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Births.csv"))

(defn mosaicData-Births2015
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births2015.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births2015.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Births2015.csv"))

(defn mosaicData-Births78
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births78.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Births78.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Births78.csv"))

(defn mosaicData-BirthsCDC
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/BirthsCDC.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/BirthsCDC.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/BirthsCDC.csv"))

(defn mosaicData-BirthsSSA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/BirthsSSA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/BirthsSSA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/BirthsSSA.csv"))

(defn mosaicData-Cards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Cards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Cards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Cards.csv"))

(defn mosaicData-CoolingWater
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/CoolingWater.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/CoolingWater.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/CoolingWater.csv"))

(defn mosaicData-Countries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Countries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Countries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Countries.csv"))

(defn mosaicData-CPS85
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/CPS85.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/CPS85.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/CPS85.csv"))

(defn mosaicData-Dimes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Dimes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Dimes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Dimes.csv"))

(defn mosaicData-Galton
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Galton.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Galton.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Galton.csv"))

(defn mosaicData-Gestation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Gestation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Gestation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Gestation.csv"))

(defn mosaicData-GoosePermits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/GoosePermits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/GoosePermits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/GoosePermits.csv"))

(defn mosaicData-HeatX
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HeatX.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HeatX.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/HeatX.csv"))

(defn mosaicData-HELPfull
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPfull.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPfull.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/HELPfull.csv"))

(defn mosaicData-HELPmiss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPmiss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPmiss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/HELPmiss.csv"))

(defn mosaicData-HELPrct
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPrct.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/HELPrct.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/HELPrct.csv"))

(defn mosaicData-KidsFeet
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/KidsFeet.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/KidsFeet.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/KidsFeet.csv"))

(defn mosaicData-Marriage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Marriage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Marriage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Marriage.csv"))

(defn mosaicData-Mites
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Mites.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Mites.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Mites.csv"))

(defn mosaicData-RailTrail
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/RailTrail.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/RailTrail.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/RailTrail.csv"))

(defn mosaicData-Riders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Riders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Riders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Riders.csv"))

(defn mosaicData-SaratogaHouses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SaratogaHouses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SaratogaHouses.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/SaratogaHouses.csv"))

(defn mosaicData-SAT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SAT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SAT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/SAT.csv"))

(defn mosaicData-SnowGR
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SnowGR.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SnowGR.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/SnowGR.csv"))

(defn mosaicData-SwimRecords
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SwimRecords.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/SwimRecords.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/SwimRecords.csv"))

(defn mosaicData-TenMileRace
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/TenMileRace.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/TenMileRace.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/TenMileRace.csv"))

(defn mosaicData-Utilities
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Utilities.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Utilities.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Utilities.csv"))

(defn mosaicData-Utilities2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Utilities2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Utilities2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Utilities2.csv"))

(defn mosaicData-Weather
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Weather.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Weather.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Weather.csv"))

(defn mosaicData-Whickham
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Whickham.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mosaicData/Whickham.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mosaicData/Whickham.csv"))

(defn mstate-aidssi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/aidssi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/aidssi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/aidssi.csv"))

(defn mstate-aidssi2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/aidssi2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/aidssi2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/aidssi2.csv"))

(defn mstate-bmt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/bmt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/bmt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/bmt.csv"))

(defn mstate-ebmt1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/ebmt1.csv"))

(defn mstate-ebmt2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/ebmt2.csv"))

(defn mstate-ebmt3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/ebmt3.csv"))

(defn mstate-ebmt4
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt4.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/ebmt4.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/ebmt4.csv"))

(defn mstate-prothr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/mstate/prothr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/mstate/prothr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/mstate/prothr.csv"))

(defn multgee-arthritis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/multgee/arthritis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/multgee/arthritis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/multgee/arthritis.csv"))

(defn multgee-housing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/multgee/housing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/multgee/housing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/multgee/housing.csv"))

(defn nlme-Alfalfa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Alfalfa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Alfalfa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Alfalfa.csv"))

(defn nlme-Assay
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Assay.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Assay.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Assay.csv"))

(defn nlme-bdf
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/bdf.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/bdf.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/bdf.csv"))

(defn nlme-BodyWeight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/BodyWeight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/BodyWeight.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/BodyWeight.csv"))

(defn nlme-Cefamandole
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Cefamandole.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Cefamandole.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Cefamandole.csv"))

(defn nlme-Dialyzer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Dialyzer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Dialyzer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Dialyzer.csv"))

(defn nlme-Earthquake
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Earthquake.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Earthquake.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Earthquake.csv"))

(defn nlme-ergoStool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/ergoStool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/ergoStool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/ergoStool.csv"))

(defn nlme-Fatigue
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Fatigue.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Fatigue.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Fatigue.csv"))

(defn nlme-Gasoline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Gasoline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Gasoline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Gasoline.csv"))

(defn nlme-Glucose
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Glucose.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Glucose.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Glucose.csv"))

(defn nlme-Glucose2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Glucose2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Glucose2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Glucose2.csv"))

(defn nlme-Gun
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Gun.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Gun.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Gun.csv"))

(defn nlme-IGF
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/IGF.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/IGF.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/IGF.csv"))

(defn nlme-Machines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Machines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Machines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Machines.csv"))

(defn nlme-MathAchieve
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/MathAchieve.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/MathAchieve.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/MathAchieve.csv"))

(defn nlme-MathAchSchool
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/MathAchSchool.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/MathAchSchool.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/MathAchSchool.csv"))

(defn nlme-Meat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Meat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Meat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Meat.csv"))

(defn nlme-Milk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Milk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Milk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Milk.csv"))

(defn nlme-Muscle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Muscle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Muscle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Muscle.csv"))

(defn nlme-Nitrendipene
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Nitrendipene.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Nitrendipene.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Nitrendipene.csv"))

(defn nlme-Oats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oats.csv"))

(defn nlme-Orthodont
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Orthodont.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Orthodont.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Orthodont.csv"))

(defn nlme-Ovary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Ovary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Ovary.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Ovary.csv"))

(defn nlme-Oxboys
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oxboys.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oxboys.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oxboys.csv"))

(defn nlme-Oxide
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oxide.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Oxide.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Oxide.csv"))

(defn nlme-PBG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/PBG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/PBG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/PBG.csv"))

(defn nlme-Phenobarb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Phenobarb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Phenobarb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Phenobarb.csv"))

(defn nlme-Pixel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Pixel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Pixel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Pixel.csv"))

(defn nlme-Quinidine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Quinidine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Quinidine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Quinidine.csv"))

(defn nlme-Rail
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Rail.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Rail.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Rail.csv"))

(defn nlme-RatPupWeight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/RatPupWeight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/RatPupWeight.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/RatPupWeight.csv"))

(defn nlme-Relaxin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Relaxin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Relaxin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Relaxin.csv"))

(defn nlme-Remifentanil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Remifentanil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Remifentanil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Remifentanil.csv"))

(defn nlme-Soybean
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Soybean.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Soybean.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Soybean.csv"))

(defn nlme-Spruce
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Spruce.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Spruce.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Spruce.csv"))

(defn nlme-Tetracycline1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Tetracycline1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Tetracycline1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Tetracycline1.csv"))

(defn nlme-Tetracycline2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Tetracycline2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Tetracycline2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Tetracycline2.csv"))

(defn nlme-Wafer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wafer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wafer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Wafer.csv"))

(defn nlme-Wheat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wheat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wheat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Wheat.csv"))

(defn nlme-Wheat2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wheat2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nlme/Wheat2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nlme/Wheat2.csv"))

(defn nycflights13-airlines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/airlines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/airlines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/airlines.csv"))

(defn nycflights13-airports
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/airports.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/airports.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/airports.csv"))

(defn nycflights13-flights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/flights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/flights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/flights.csv"))

(defn nycflights13-planes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/planes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/planes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/planes.csv"))

(defn nycflights13-weather
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/weather.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/nycflights13/weather.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/nycflights13/weather.csv"))

(defn openintro-absenteeism
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/absenteeism.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/absenteeism.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/absenteeism.csv"))

(defn openintro-acs12
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/acs12.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/acs12.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/acs12.csv"))

(defn openintro-age_at_mar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/age_at_mar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/age_at_mar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/age_at_mar.csv"))

(defn openintro-ames
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ames.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ames.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ames.csv"))

(defn openintro-ami_occurrences
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ami_occurrences.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ami_occurrences.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ami_occurrences.csv"))

(defn openintro-antibiotics
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/antibiotics.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/antibiotics.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/antibiotics.csv"))

(defn openintro-arbuthnot
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/arbuthnot.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/arbuthnot.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/arbuthnot.csv"))

(defn openintro-ask
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ask.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ask.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ask.csv"))

(defn openintro-association
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/association.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/association.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/association.csv"))

(defn openintro-assortative_mating
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/assortative_mating.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/assortative_mating.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/assortative_mating.csv"))

(defn openintro-assortive_mating
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/assortive_mating.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/assortive_mating.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/assortive_mating.csv"))

(defn openintro-avandia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/avandia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/avandia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/avandia.csv"))

(defn openintro-babies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/babies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/babies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/babies.csv"))

(defn openintro-babies_crawl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/babies_crawl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/babies_crawl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/babies_crawl.csv"))

(defn openintro-bac
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/bac.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/bac.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/bac.csv"))

(defn openintro-ball_bearing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ball_bearing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ball_bearing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ball_bearing.csv"))

(defn openintro-bdims
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/bdims.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/bdims.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/bdims.csv"))

(defn openintro-biontech_adolescents
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/biontech_adolescents.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/biontech_adolescents.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/biontech_adolescents.csv"))

(defn openintro-birds
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/birds.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/birds.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/birds.csv"))

(defn openintro-births
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/births.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/births.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/births.csv"))

(defn openintro-births14
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/births14.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/births14.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/births14.csv"))

(defn openintro-blizzard_salary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/blizzard_salary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/blizzard_salary.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/blizzard_salary.csv"))

(defn openintro-books
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/books.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/books.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/books.csv"))

(defn openintro-burger
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/burger.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/burger.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/burger.csv"))

(defn openintro-cancer_in_dogs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cancer_in_dogs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cancer_in_dogs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cancer_in_dogs.csv"))

(defn openintro-cards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cards.csv"))

(defn openintro-cars04
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cars04.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cars04.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cars04.csv"))

(defn openintro-cars93
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cars93.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cars93.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cars93.csv"))

(defn openintro-cchousing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cchousing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cchousing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cchousing.csv"))

(defn openintro-census
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/census.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/census.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/census.csv"))

(defn openintro-cherry
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cherry.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cherry.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cherry.csv"))

(defn openintro-china
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/china.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/china.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/china.csv"))

(defn openintro-cia_factbook
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cia_factbook.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cia_factbook.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cia_factbook.csv"))

(defn openintro-classdata
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/classdata.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/classdata.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/classdata.csv"))

(defn openintro-cle_sac
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cle_sac.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cle_sac.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cle_sac.csv"))

(defn openintro-climate70
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/climate70.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/climate70.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/climate70.csv"))

(defn openintro-climber_drugs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/climber_drugs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/climber_drugs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/climber_drugs.csv"))

(defn openintro-coast_starlight
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/coast_starlight.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/coast_starlight.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/coast_starlight.csv"))

(defn openintro-COL
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/COL.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/COL.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/COL.csv"))

(defn openintro-comics
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/comics.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/comics.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/comics.csv"))

(defn openintro-corr_match
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/corr_match.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/corr_match.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/corr_match.csv"))

(defn openintro-country_iso
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/country_iso.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/country_iso.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/country_iso.csv"))

(defn openintro-cpr
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cpr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cpr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cpr.csv"))

(defn openintro-cpu
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cpu.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/cpu.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/cpu.csv"))

(defn openintro-credits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/credits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/credits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/credits.csv"))

(defn openintro-daycare_fines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/daycare_fines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/daycare_fines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/daycare_fines.csv"))

(defn openintro-diabetes2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/diabetes2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/diabetes2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/diabetes2.csv"))

(defn openintro-dream
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/dream.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/dream.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/dream.csv"))

(defn openintro-drone_blades
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/drone_blades.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/drone_blades.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/drone_blades.csv"))

(defn openintro-drug_use
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/drug_use.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/drug_use.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/drug_use.csv"))

(defn openintro-duke_forest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/duke_forest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/duke_forest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/duke_forest.csv"))

(defn openintro-earthquakes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/earthquakes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/earthquakes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/earthquakes.csv"))

(defn openintro-ebola_survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ebola_survey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ebola_survey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ebola_survey.csv"))

(defn openintro-elmhurst
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/elmhurst.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/elmhurst.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/elmhurst.csv"))

(defn openintro-email
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/email.csv"))

(defn openintro-email_test
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email_test.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email_test.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/email_test.csv"))

(defn openintro-email50
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email50.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/email50.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/email50.csv"))

(defn openintro-env_regulation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/env_regulation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/env_regulation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/env_regulation.csv"))

(defn openintro-epa2012
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/epa2012.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/epa2012.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/epa2012.csv"))

(defn openintro-epa2021
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/epa2021.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/epa2021.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/epa2021.csv"))

(defn openintro-esi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/esi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/esi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/esi.csv"))

(defn openintro-ethanol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ethanol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ethanol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ethanol.csv"))

(defn openintro-evals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/evals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/evals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/evals.csv"))

(defn openintro-exam_grades
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exam_grades.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exam_grades.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/exam_grades.csv"))

(defn openintro-exams
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exams.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exams.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/exams.csv"))

(defn openintro-exclusive_relationship
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exclusive_relationship.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/exclusive_relationship.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/exclusive_relationship.csv"))

(defn openintro-fact_opinion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fact_opinion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fact_opinion.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fact_opinion.csv"))

(defn openintro-family_college
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/family_college.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/family_college.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/family_college.csv"))

(defn openintro-fastfood
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fastfood.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fastfood.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fastfood.csv"))

(defn openintro-fcid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fcid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fcid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fcid.csv"))

(defn openintro-fheights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fheights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fheights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fheights.csv"))

(defn openintro-fish_age
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fish_age.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fish_age.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fish_age.csv"))

(defn openintro-fish_oil_18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fish_oil_18.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/fish_oil_18.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/fish_oil_18.csv"))

(defn openintro-flow_rates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/flow_rates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/flow_rates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/flow_rates.csv"))

(defn openintro-friday
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/friday.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/friday.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/friday.csv"))

(defn openintro-full_body_scan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/full_body_scan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/full_body_scan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/full_body_scan.csv"))

(defn openintro-gdp_countries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gdp_countries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gdp_countries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gdp_countries.csv"))

(defn openintro-gear_company
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gear_company.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gear_company.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gear_company.csv"))

(defn openintro-gender_discrimination
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gender_discrimination.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gender_discrimination.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gender_discrimination.csv"))

(defn openintro-get_it_dunn_run
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/get_it_dunn_run.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/get_it_dunn_run.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/get_it_dunn_run.csv"))

(defn openintro-gifted
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gifted.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gifted.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gifted.csv"))

(defn openintro-global_warming_pew
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/global_warming_pew.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/global_warming_pew.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/global_warming_pew.csv"))

(defn openintro-goog
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/goog.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/goog.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/goog.csv"))

(defn openintro-gov_poll
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gov_poll.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gov_poll.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gov_poll.csv"))

(defn openintro-gpa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gpa.csv"))

(defn openintro-gpa_iq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa_iq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa_iq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gpa_iq.csv"))

(defn openintro-gpa_study_hours
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa_study_hours.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gpa_study_hours.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gpa_study_hours.csv"))

(defn openintro-gradestv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gradestv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gradestv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gradestv.csv"))

(defn openintro-gsearch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gsearch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gsearch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gsearch.csv"))

(defn openintro-gss_wordsum_class
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gss_wordsum_class.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gss_wordsum_class.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gss_wordsum_class.csv"))

(defn openintro-gss2010
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gss2010.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/gss2010.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/gss2010.csv"))

(defn openintro-health_coverage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/health_coverage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/health_coverage.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/health_coverage.csv"))

(defn openintro-healthcare_law_survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/healthcare_law_survey.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/healthcare_law_survey.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/healthcare_law_survey.csv"))

(defn openintro-heart_transplant
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/heart_transplant.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/heart_transplant.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/heart_transplant.csv"))

(defn openintro-helium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/helium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/helium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/helium.csv"))

(defn openintro-helmet
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/helmet.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/helmet.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/helmet.csv"))

(defn openintro-hfi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/hfi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/hfi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/hfi.csv"))

(defn openintro-house
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/house.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/house.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/house.csv"))

(defn openintro-housing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/housing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/housing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/housing.csv"))

(defn openintro-hsb2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/hsb2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/hsb2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/hsb2.csv"))

(defn openintro-husbands_wives
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/husbands_wives.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/husbands_wives.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/husbands_wives.csv"))

(defn openintro-immigration
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/immigration.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/immigration.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/immigration.csv"))

(defn openintro-IMSCOL
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/IMSCOL.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/IMSCOL.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/IMSCOL.csv"))

(defn openintro-infmortrate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/infmortrate.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/infmortrate.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/infmortrate.csv"))

(defn openintro-iowa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/iowa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/iowa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/iowa.csv"))

(defn openintro-ipod
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ipod.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ipod.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ipod.csv"))

(defn openintro-iran
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/iran.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/iran.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/iran.csv"))

(defn openintro-jury
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/jury.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/jury.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/jury.csv"))

(defn openintro-kobe_basket
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/kobe_basket.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/kobe_basket.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/kobe_basket.csv"))

(defn openintro-labor_market_discrimination
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/labor_market_discrimination.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/labor_market_discrimination.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/labor_market_discrimination.csv"))

(defn openintro-LAhomes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/LAhomes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/LAhomes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/LAhomes.csv"))

(defn openintro-law_resume
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/law_resume.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/law_resume.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/law_resume.csv"))

(defn openintro-lecture_learning
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lecture_learning.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lecture_learning.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/lecture_learning.csv"))

(defn openintro-leg_mari
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/leg_mari.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/leg_mari.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/leg_mari.csv"))

(defn openintro-lego_population
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lego_population.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lego_population.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/lego_population.csv"))

(defn openintro-lego_sample
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lego_sample.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lego_sample.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/lego_sample.csv"))

(defn openintro-life_exp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/life_exp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/life_exp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/life_exp.csv"))

(defn openintro-lizard_habitat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lizard_habitat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lizard_habitat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/lizard_habitat.csv"))

(defn openintro-lizard_run
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lizard_run.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/lizard_run.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/lizard_run.csv"))

(defn openintro-loan50
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/loan50.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/loan50.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/loan50.csv"))

(defn openintro-loans_full_schema
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/loans_full_schema.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/loans_full_schema.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/loans_full_schema.csv"))

(defn openintro-london_boroughs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/london_boroughs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/london_boroughs.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/london_boroughs.csv"))

(defn openintro-london_murders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/london_murders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/london_murders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/london_murders.csv"))

(defn openintro-mail_me
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mail_me.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mail_me.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mail_me.csv"))

(defn openintro-major_survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/major_survey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/major_survey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/major_survey.csv"))

(defn openintro-malaria
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/malaria.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/malaria.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/malaria.csv"))

(defn openintro-male_heights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/male_heights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/male_heights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/male_heights.csv"))

(defn openintro-male_heights_fcid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/male_heights_fcid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/male_heights_fcid.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/male_heights_fcid.csv"))

(defn openintro-mammals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mammals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mammals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mammals.csv"))

(defn openintro-mammogram
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mammogram.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mammogram.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mammogram.csv"))

(defn openintro-manhattan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/manhattan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/manhattan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/manhattan.csv"))

(defn openintro-marathon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/marathon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/marathon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/marathon.csv"))

(defn openintro-mariokart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mariokart.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mariokart.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mariokart.csv"))

(defn openintro-mcu_films
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mcu_films.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mcu_films.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mcu_films.csv"))

(defn openintro-midterms_house
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/midterms_house.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/midterms_house.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/midterms_house.csv"))

(defn openintro-migraine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/migraine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/migraine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/migraine.csv"))

(defn openintro-military
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/military.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/military.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/military.csv"))

(defn openintro-mlb
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mlb.csv"))

(defn openintro-mlb_players_18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb_players_18.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb_players_18.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mlb_players_18.csv"))

(defn openintro-mlb_teams
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb_teams.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlb_teams.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mlb_teams.csv"))

(defn openintro-mlbbat10
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlbbat10.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mlbbat10.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mlbbat10.csv"))

(defn openintro-mn_police_use_of_force
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mn_police_use_of_force.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mn_police_use_of_force.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mn_police_use_of_force.csv"))

(defn openintro-movies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/movies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/movies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/movies.csv"))

(defn openintro-mtl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mtl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/mtl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/mtl.csv"))

(defn openintro-murders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/murders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/murders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/murders.csv"))

(defn openintro-nba_finals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_finals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_finals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nba_finals.csv"))

(defn openintro-nba_finals_teams
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_finals_teams.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_finals_teams.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nba_finals_teams.csv"))

(defn openintro-nba_heights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_heights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_heights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nba_heights.csv"))

(defn openintro-nba_players_19
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_players_19.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nba_players_19.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nba_players_19.csv"))

(defn openintro-ncbirths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ncbirths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ncbirths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ncbirths.csv"))

(defn openintro-nuclear_survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nuclear_survey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nuclear_survey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nuclear_survey.csv"))

(defn openintro-nyc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nyc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nyc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nyc.csv"))

(defn openintro-nyc_marathon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nyc_marathon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nyc_marathon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nyc_marathon.csv"))

(defn openintro-nycflights
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nycflights.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/nycflights.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/nycflights.csv"))

(defn openintro-offshore_drilling
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/offshore_drilling.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/offshore_drilling.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/offshore_drilling.csv"))

(defn openintro-openintro_colors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/openintro_colors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/openintro_colors.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/openintro_colors.csv"))

(defn openintro-opportunity_cost
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/opportunity_cost.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/opportunity_cost.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/opportunity_cost.csv"))

(defn openintro-orings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/orings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/orings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/orings.csv"))

(defn openintro-oscars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/oscars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/oscars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/oscars.csv"))

(defn openintro-outliers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/outliers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/outliers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/outliers.csv"))

(defn openintro-paralympic_1500
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/paralympic_1500.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/paralympic_1500.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/paralympic_1500.csv"))

(defn openintro-penelope
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penelope.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penelope.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/penelope.csv"))

(defn openintro-penetrating_oil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penetrating_oil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penetrating_oil.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/penetrating_oil.csv"))

(defn openintro-penny_ages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penny_ages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/penny_ages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/penny_ages.csv"))

(defn openintro-pew_energy_2018
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pew_energy_2018.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pew_energy_2018.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/pew_energy_2018.csv"))

(defn openintro-photo_classify
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/photo_classify.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/photo_classify.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/photo_classify.csv"))

(defn openintro-piracy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/piracy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/piracy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/piracy.csv"))

(defn openintro-playing_cards
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/playing_cards.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/playing_cards.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/playing_cards.csv"))

(defn openintro-pm25_2011_durham
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pm25_2011_durham.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pm25_2011_durham.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/pm25_2011_durham.csv"))

(defn openintro-pm25_2022_durham
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pm25_2022_durham.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/pm25_2022_durham.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/pm25_2022_durham.csv"))

(defn openintro-poker
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/poker.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/poker.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/poker.csv"))

(defn openintro-possum
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/possum.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/possum.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/possum.csv"))

(defn openintro-ppp_201503
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ppp_201503.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ppp_201503.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ppp_201503.csv"))

(defn openintro-present
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/present.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/present.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/present.csv"))

(defn openintro-president
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/president.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/president.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/president.csv"))

(defn openintro-prison
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/prison.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/prison.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/prison.csv"))

(defn openintro-prius_mpg
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/prius_mpg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/prius_mpg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/prius_mpg.csv"))

(defn openintro-race_justice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/race_justice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/race_justice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/race_justice.csv"))

(defn openintro-reddit_finance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/reddit_finance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/reddit_finance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/reddit_finance.csv"))

(defn openintro-res_demo_1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/res_demo_1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/res_demo_1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/res_demo_1.csv"))

(defn openintro-res_demo_2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/res_demo_2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/res_demo_2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/res_demo_2.csv"))

(defn openintro-resume
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/resume.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/resume.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/resume.csv"))

(defn openintro-rosling_responses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/rosling_responses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/rosling_responses.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/rosling_responses.csv"))

(defn openintro-russian_influence_on_us_election_2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/russian_influence_on_us_election_2016.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/russian_influence_on_us_election_2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/russian_influence_on_us_election_2016.csv"))

(defn openintro-sa_gdp_elec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sa_gdp_elec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sa_gdp_elec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sa_gdp_elec.csv"))

(defn openintro-salinity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/salinity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/salinity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/salinity.csv"))

(defn openintro-sat_improve
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sat_improve.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sat_improve.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sat_improve.csv"))

(defn openintro-satgpa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/satgpa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/satgpa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/satgpa.csv"))

(defn openintro-scotus_healthcare
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/scotus_healthcare.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/scotus_healthcare.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/scotus_healthcare.csv"))

(defn openintro-seattlepets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/seattlepets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/seattlepets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/seattlepets.csv"))

(defn openintro-sex_discrimination
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sex_discrimination.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sex_discrimination.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sex_discrimination.csv"))

(defn openintro-simpsons_paradox_covid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simpsons_paradox_covid.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simpsons_paradox_covid.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/simpsons_paradox_covid.csv"))

(defn openintro-simulated_normal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simulated_normal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simulated_normal.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/simulated_normal.csv"))

(defn openintro-simulated_scatter
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simulated_scatter.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/simulated_scatter.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/simulated_scatter.csv"))

(defn openintro-sinusitis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sinusitis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sinusitis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sinusitis.csv"))

(defn openintro-sleep_deprivation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sleep_deprivation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sleep_deprivation.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sleep_deprivation.csv"))

(defn openintro-smallpox
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/smallpox.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/smallpox.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/smallpox.csv"))

(defn openintro-smoking
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/smoking.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/smoking.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/smoking.csv"))

(defn openintro-snowfall
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/snowfall.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/snowfall.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/snowfall.csv"))

(defn openintro-socialexp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/socialexp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/socialexp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/socialexp.csv"))

(defn openintro-soda
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/soda.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/soda.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/soda.csv"))

(defn openintro-solar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/solar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/solar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/solar.csv"))

(defn openintro-sowc_child_mortality
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_child_mortality.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_child_mortality.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sowc_child_mortality.csv"))

(defn openintro-sowc_demographics
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_demographics.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_demographics.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sowc_demographics.csv"))

(defn openintro-sowc_maternal_newborn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_maternal_newborn.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sowc_maternal_newborn.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sowc_maternal_newborn.csv"))

(defn openintro-sp500
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sp500.csv"))

(defn openintro-sp500_1950_2018
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500_1950_2018.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500_1950_2018.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sp500_1950_2018.csv"))

(defn openintro-sp500_seq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500_seq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sp500_seq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sp500_seq.csv"))

(defn openintro-speed_gender_height
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/speed_gender_height.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/speed_gender_height.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/speed_gender_height.csv"))

(defn openintro-ssd_speed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ssd_speed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ssd_speed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ssd_speed.csv"))

(defn openintro-starbucks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/starbucks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/starbucks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/starbucks.csv"))

(defn openintro-stats_scores
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stats_scores.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stats_scores.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/stats_scores.csv"))

(defn openintro-stem_cell
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stem_cell.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stem_cell.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/stem_cell.csv"))

(defn openintro-stent30
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stent30.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stent30.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/stent30.csv"))

(defn openintro-stent365
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stent365.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stent365.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/stent365.csv"))

(defn openintro-stocks_18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stocks_18.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/stocks_18.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/stocks_18.csv"))

(defn openintro-student_housing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/student_housing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/student_housing.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/student_housing.csv"))

(defn openintro-student_sleep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/student_sleep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/student_sleep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/student_sleep.csv"))

(defn openintro-sulphinpyrazone
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sulphinpyrazone.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/sulphinpyrazone.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/sulphinpyrazone.csv"))

(defn openintro-supreme_court
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/supreme_court.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/supreme_court.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/supreme_court.csv"))

(defn openintro-teacher
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/teacher.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/teacher.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/teacher.csv"))

(defn openintro-textbooks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/textbooks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/textbooks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/textbooks.csv"))

(defn openintro-thanksgiving_spend
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/thanksgiving_spend.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/thanksgiving_spend.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/thanksgiving_spend.csv"))

(defn openintro-tips
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/tips.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/tips.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/tips.csv"))

(defn openintro-toohey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/toohey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/toohey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/toohey.csv"))

(defn openintro-tourism
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/tourism.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/tourism.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/tourism.csv"))

(defn openintro-toy_anova
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/toy_anova.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/toy_anova.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/toy_anova.csv"))

(defn openintro-transplant
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/transplant.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/transplant.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/transplant.csv"))

(defn openintro-twins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/twins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/twins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/twins.csv"))

(defn openintro-ucb_admit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucb_admit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucb_admit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ucb_admit.csv"))

(defn openintro-ucla_f18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucla_f18.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucla_f18.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ucla_f18.csv"))

(defn openintro-ucla_textbooks_f18
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucla_textbooks_f18.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ucla_textbooks_f18.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ucla_textbooks_f18.csv"))

(defn openintro-ukdemo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ukdemo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/ukdemo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/ukdemo.csv"))

(defn openintro-unempl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/unempl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/unempl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/unempl.csv"))

(defn openintro-unemploy_pres
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/unemploy_pres.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/unemploy_pres.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/unemploy_pres.csv"))

(defn openintro-us_temperature
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/us_temperature.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/us_temperature.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/us_temperature.csv"))

(defn openintro-winery_cars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/winery_cars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/winery_cars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/winery_cars.csv"))

(defn openintro-world_pop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/world_pop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/world_pop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/world_pop.csv"))

(defn openintro-xom
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/xom.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/xom.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/xom.csv"))

(defn openintro-yawn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yawn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yawn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/yawn.csv"))

(defn openintro-yrbss
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yrbss.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yrbss.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/yrbss.csv"))

(defn openintro-yrbss_samp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yrbss_samp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/openintro/yrbss_samp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/openintro/yrbss_samp.csv"))

(defn ordinal-income
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/income.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/income.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ordinal/income.csv"))

(defn ordinal-soup
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/soup.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/soup.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ordinal/soup.csv"))

(defn ordinal-wine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/wine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ordinal/wine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ordinal/wine.csv"))

(defn palmerpenguins-penguins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/palmerpenguins/penguins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/palmerpenguins/penguins.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins.csv"))

(defn palmerpenguins-penguins_rawpenguins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/palmerpenguins/penguins_raw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/palmerpenguins/penguins_raw.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/palmerpenguins/penguins_raw.csv"))

(defn plm-Cigar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Cigar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Cigar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Cigar.csv"))

(defn plm-Crime
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Crime.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Crime.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Crime.csv"))

(defn plm-EmplUK
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/EmplUK.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/EmplUK.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/EmplUK.csv"))

(defn plm-Gasoline
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Gasoline.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Gasoline.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Gasoline.csv"))

(defn plm-Grunfeld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Grunfeld.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Grunfeld.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Grunfeld.csv"))

(defn plm-Hedonic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Hedonic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Hedonic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Hedonic.csv"))

(defn plm-LaborSupply
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/LaborSupply.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/LaborSupply.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/LaborSupply.csv"))

(defn plm-Males
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Males.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Males.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Males.csv"))

(defn plm-Parity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Parity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Parity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Parity.csv"))

(defn plm-Produc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Produc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Produc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Produc.csv"))

(defn plm-RiceFarms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/RiceFarms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/RiceFarms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/RiceFarms.csv"))

(defn plm-Snmesp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Snmesp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Snmesp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Snmesp.csv"))

(defn plm-SumHes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/SumHes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/SumHes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/SumHes.csv"))

(defn plm-Wages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plm/Wages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plm/Wages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plm/Wages.csv"))

(defn plyr-baseball
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plyr/baseball.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plyr/baseball.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plyr/baseball.csv"))

(defn plyr-ozone
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/plyr/ozone.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/plyr/ozone.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/plyr/ozone.csv"))

(defn pscl-absentee
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/absentee.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/absentee.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/absentee.csv"))

(defn pscl-admit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/admit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/admit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/admit.csv"))

(defn pscl-AustralianElectionPolling
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/AustralianElectionPolling.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/AustralianElectionPolling.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/AustralianElectionPolling.csv"))

(defn pscl-AustralianElections
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/AustralianElections.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/AustralianElections.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/AustralianElections.csv"))

(defn pscl-bioChemists
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/bioChemists.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/bioChemists.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/bioChemists.csv"))

(defn pscl-ca2006
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/ca2006.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/ca2006.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/ca2006.csv"))

(defn pscl-EfronMorris
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/EfronMorris.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/EfronMorris.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/EfronMorris.csv"))

(defn pscl-iraqVote
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/iraqVote.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/iraqVote.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/iraqVote.csv"))

(defn pscl-partycodes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/partycodes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/partycodes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/partycodes.csv"))

(defn pscl-politicalInformation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/politicalInformation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/politicalInformation.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/politicalInformation.csv"))

(defn pscl-presidentialElections
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/presidentialElections.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/presidentialElections.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/presidentialElections.csv"))

(defn pscl-prussian
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/prussian.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/prussian.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/prussian.csv"))

(defn pscl-RockTheVote
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/RockTheVote.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/RockTheVote.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/RockTheVote.csv"))

(defn pscl-state.info
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/state.info.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/state.info.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/state.info.csv"))

(defn pscl-UKHouseOfCommons
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/UKHouseOfCommons.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/UKHouseOfCommons.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/UKHouseOfCommons.csv"))

(defn pscl-unionDensity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/unionDensity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/unionDensity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/unionDensity.csv"))

(defn pscl-vote92
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/pscl/vote92.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/pscl/vote92.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/pscl/vote92.csv"))

(defn psych-Bechtoldt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Bechtoldt.csv"))

(defn psych-Bechtoldt.1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Bechtoldt.1.csv"))

(defn psych-Bechtoldt.2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Bechtoldt.2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Bechtoldt.2.csv"))

(defn psych-bfi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/bfi.csv"))

(defn psych-bfi.dictionary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.dictionary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.dictionary.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/bfi.dictionary.csv"))

(defn psych-bfi.keysbfi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.keys.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/bfi.keys.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/bfi.keys.csv"))

(defn psych-bock.tablebock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/bock.table.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/bock.table.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/bock.table.csv"))

(defn psych-cattell
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/cattell.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/cattell.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/cattell.csv"))

(defn psych-ChenSchmid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Chen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Chen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Chen.csv"))

(defn psych-Dwyer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Dwyer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Dwyer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Dwyer.csv"))

(defn psych-GarciaGSBE
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Garcia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Garcia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Garcia.csv"))

(defn psych-Gleser
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Gleser.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Gleser.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Gleser.csv"))

(defn psych-Gorsuch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Gorsuch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Gorsuch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Gorsuch.csv"))

(defn psych-Harman.5
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.5.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.5.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Harman.5.csv"))

(defn psych-Harman.8
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.8.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.8.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Harman.8.csv"))

(defn psych-Harman.BurtHarman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.Burt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.Burt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Harman.Burt.csv"))

(defn psych-Harman.HolzingerHarman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.Holzinger.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.Holzinger.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Harman.Holzinger.csv"))

(defn psych-Harman.political
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.political.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Harman.political.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Harman.political.csv"))

(defn psych-Holzinger
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Holzinger.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Holzinger.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Holzinger.csv"))

(defn psych-Holzinger.9
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Holzinger.9.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Holzinger.9.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Holzinger.9.csv"))

(defn psych-lsat6bock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/lsat6.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/lsat6.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/lsat6.csv"))

(defn psych-lsat7bock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/lsat7.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/lsat7.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/lsat7.csv"))

(defn psych-Reise
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Reise.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Reise.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Reise.csv"))

(defn psych-sat.act
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/sat.act.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/sat.act.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/sat.act.csv"))

(defn psych-Schmid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Schmid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Schmid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Schmid.csv"))

(defn psych-schmid.leimanSchmid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/schmid.leiman.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/schmid.leiman.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/schmid.leiman.csv"))

(defn psych-small.msq
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/small.msq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/small.msq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/small.msq.csv"))

(defn psych-Tal_Ortal_or
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tal_Or.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tal_Or.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Tal_Or.csv"))

(defn psych-Tal.Or
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tal.Or.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tal.Or.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Tal.Or.csv"))

(defn psych-Thurstone
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Thurstone.csv"))

(defn psych-Thurstone.33
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.33.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.33.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Thurstone.33.csv"))

(defn psych-Thurstone.33G
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.33G.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.33G.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Thurstone.33G.csv"))

(defn psych-Thurstone.9
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.9.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Thurstone.9.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Thurstone.9.csv"))

(defn psych-Tucker
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tucker.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/Tucker.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/Tucker.csv"))

(defn psych-WestSchmid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/West.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/West.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/West.csv"))

(defn psych-withinBetween
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/psych/withinBetween.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/psych/withinBetween.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/psych/withinBetween.csv"))

(defn quantreg-barro
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/barro.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/barro.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/barro.csv"))

(defn quantreg-Bosco
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/Bosco.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/Bosco.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/Bosco.csv"))

(defn quantreg-CobarOre
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/CobarOre.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/CobarOre.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/CobarOre.csv"))

(defn quantreg-engel
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/engel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/engel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/engel.csv"))

(defn quantreg-gasprice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/gasprice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/gasprice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/gasprice.csv"))

(defn quantreg-Mammals
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/Mammals.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/Mammals.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/Mammals.csv"))

(defn quantreg-MelTemp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/MelTemp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/MelTemp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/MelTemp.csv"))

(defn quantreg-uis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/uis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/quantreg/uis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/quantreg/uis.csv"))

(defn ratdat-complete
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/complete.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/complete.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ratdat/complete.csv"))

(defn ratdat-complete_old
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/complete_old.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/complete_old.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ratdat/complete_old.csv"))

(defn ratdat-plots
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/plots.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/plots.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ratdat/plots.csv"))

(defn ratdat-species
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/species.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/species.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ratdat/species.csv"))

(defn ratdat-surveys
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/surveys.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/ratdat/surveys.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/ratdat/surveys.csv"))

(defn reshape2-french_fries
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/french_fries.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/french_fries.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/reshape2/french_fries.csv"))

(defn reshape2-smiths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/smiths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/smiths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/reshape2/smiths.csv"))

(defn reshape2-tips
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/reshape2/tips.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/reshape2/tips.csv"))

(defn robustbase-aircraft
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/aircraft.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/aircraft.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/aircraft.csv"))

(defn robustbase-airmay
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/airmay.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/airmay.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/airmay.csv"))

(defn robustbase-alcohol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/alcohol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/alcohol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/alcohol.csv"))

(defn robustbase-ambientNOxCH
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/ambientNOxCH.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/ambientNOxCH.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/ambientNOxCH.csv"))

(defn robustbase-Animals2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/Animals2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/Animals2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/Animals2.csv"))

(defn robustbase-biomassTill
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/biomassTill.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/biomassTill.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/biomassTill.csv"))

(defn robustbase-bushfire
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/bushfire.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/bushfire.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/bushfire.csv"))

(defn robustbase-carrots
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/carrots.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/carrots.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/carrots.csv"))

(defn robustbase-cloud
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/cloud.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/cloud.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/cloud.csv"))

(defn robustbase-coleman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/coleman.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/coleman.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/coleman.csv"))

(defn robustbase-condroz
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/condroz.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/condroz.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/condroz.csv"))

(defn robustbase-CrohnD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/CrohnD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/CrohnD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/CrohnD.csv"))

(defn robustbase-cushny
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/cushny.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/cushny.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/cushny.csv"))

(defn robustbase-delivery
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/delivery.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/delivery.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/delivery.csv"))

(defn robustbase-education
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/education.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/education.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/education.csv"))

(defn robustbase-epilepsy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/epilepsy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/epilepsy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/epilepsy.csv"))

(defn robustbase-exAM
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/exAM.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/exAM.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/exAM.csv"))

(defn robustbase-foodstamp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/foodstamp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/foodstamp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/foodstamp.csv"))

(defn robustbase-hbk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/hbk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/hbk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/hbk.csv"))

(defn robustbase-heart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/heart.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/heart.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/heart.csv"))

(defn robustbase-kootenay
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/kootenay.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/kootenay.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/kootenay.csv"))

(defn robustbase-lactic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/lactic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/lactic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/lactic.csv"))

(defn robustbase-los
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/los.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/los.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/los.csv"))

(defn robustbase-milk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/milk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/milk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/milk.csv"))

(defn robustbase-NOxEmissions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/NOxEmissions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/NOxEmissions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/NOxEmissions.csv"))

(defn robustbase-pension
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pension.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pension.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/pension.csv"))

(defn robustbase-phosphor
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/phosphor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/phosphor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/phosphor.csv"))

(defn robustbase-pilot
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pilot.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pilot.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/pilot.csv"))

(defn robustbase-possum.matpossumDiv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/possum.mat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/possum.mat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/possum.mat.csv"))

(defn robustbase-possumDiv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/possumDiv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/possumDiv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/possumDiv.csv"))

(defn robustbase-pulpfiber
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pulpfiber.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/pulpfiber.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/pulpfiber.csv"))

(defn robustbase-radarImage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/radarImage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/radarImage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/radarImage.csv"))

(defn robustbase-salinity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/salinity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/salinity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/salinity.csv"))

(defn robustbase-SiegelsEx
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/SiegelsEx.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/SiegelsEx.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/SiegelsEx.csv"))

(defn robustbase-starsCYG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/starsCYG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/starsCYG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/starsCYG.csv"))

(defn robustbase-steamUse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/steamUse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/steamUse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/steamUse.csv"))

(defn robustbase-telef
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/telef.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/telef.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/telef.csv"))

(defn robustbase-toxicity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/toxicity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/toxicity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/toxicity.csv"))

(defn robustbase-vaso
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/vaso.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/vaso.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/vaso.csv"))

(defn robustbase-wagnerGrowth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/wagnerGrowth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/wagnerGrowth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/wagnerGrowth.csv"))

(defn robustbase-wood
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/wood.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/wood.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/wood.csv"))

(defn robustbase-x30o50
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/x30o50.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/robustbase/x30o50.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/robustbase/x30o50.csv"))

(defn rpart-car.test.frame
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/car.test.frame.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/car.test.frame.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/car.test.frame.csv"))

(defn rpart-car90
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/car90.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/car90.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/car90.csv"))

(defn rpart-cu.summary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/cu.summary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/cu.summary.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/cu.summary.csv"))

(defn rpart-kyphosis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/kyphosis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/kyphosis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/kyphosis.csv"))

(defn rpart-solder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/solder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/solder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/solder.csv"))

(defn rpart-solder.balancesolder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/solder.balance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/solder.balance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/solder.balance.csv"))

(defn rpart-stagec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/rpart/stagec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/rpart/stagec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/rpart/stagec.csv"))

(defn sampleSelection-Mroz87
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/Mroz87.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/Mroz87.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sampleSelection/Mroz87.csv"))

(defn sampleSelection-nlswork
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/nlswork.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/nlswork.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sampleSelection/nlswork.csv"))

(defn sampleSelection-RandHIE
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/RandHIE.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/RandHIE.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sampleSelection/RandHIE.csv"))

(defn sampleSelection-Smoke
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/Smoke.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sampleSelection/Smoke.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sampleSelection/Smoke.csv"))

(defn sandwich-InstInnovation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/InstInnovation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/InstInnovation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sandwich/InstInnovation.csv"))

(defn sandwich-Investment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/Investment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/Investment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sandwich/Investment.csv"))

(defn sandwich-PetersenCL
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/PetersenCL.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/PetersenCL.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sandwich/PetersenCL.csv"))

(defn sandwich-PublicSchools
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/PublicSchools.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sandwich/PublicSchools.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sandwich/PublicSchools.csv"))

(defn sem-Bollen
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/Bollen.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/Bollen.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/Bollen.csv"))

(defn sem-CNES
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/CNES.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/CNES.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/CNES.csv"))

(defn sem-HS.data
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/HS.data.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/HS.data.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/HS.data.csv"))

(defn sem-Klein
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/Klein.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/Klein.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/Klein.csv"))

(defn sem-Kmenta
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/Kmenta.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/Kmenta.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/Kmenta.csv"))

(defn sem-Tests
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/sem/Tests.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/sem/Tests.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/sem/Tests.csv"))

(defn Stat2Data-AccordPrice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AccordPrice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AccordPrice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AccordPrice.csv"))

(defn Stat2Data-AHCAvote2017
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AHCAvote2017.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AHCAvote2017.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AHCAvote2017.csv"))

(defn Stat2Data-Airlines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Airlines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Airlines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Airlines.csv"))

(defn Stat2Data-Alfalfa
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Alfalfa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Alfalfa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Alfalfa.csv"))

(defn Stat2Data-AlitoConfirmation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AlitoConfirmation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AlitoConfirmation.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AlitoConfirmation.csv"))

(defn Stat2Data-Amyloid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Amyloid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Amyloid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Amyloid.csv"))

(defn Stat2Data-AppleStock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AppleStock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AppleStock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AppleStock.csv"))

(defn Stat2Data-ArcheryData
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ArcheryData.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ArcheryData.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ArcheryData.csv"))

(defn Stat2Data-AthleteGrad
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AthleteGrad.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AthleteGrad.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AthleteGrad.csv"))

(defn Stat2Data-AudioVisual
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AudioVisual.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AudioVisual.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AudioVisual.csv"))

(defn Stat2Data-AutoPollution
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AutoPollution.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/AutoPollution.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/AutoPollution.csv"))

(defn Stat2Data-Backpack
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Backpack.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Backpack.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Backpack.csv"))

(defn Stat2Data-BaseballTimes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BaseballTimes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BaseballTimes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BaseballTimes.csv"))

(defn Stat2Data-BaseballTimes2017
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BaseballTimes2017.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BaseballTimes2017.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BaseballTimes2017.csv"))

(defn Stat2Data-BeeStings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BeeStings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BeeStings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BeeStings.csv"))

(defn Stat2Data-BirdCalcium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BirdCalcium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BirdCalcium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BirdCalcium.csv"))

(defn Stat2Data-BirdNest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BirdNest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BirdNest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BirdNest.csv"))

(defn Stat2Data-Blood1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Blood1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Blood1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Blood1.csv"))

(defn Stat2Data-BlueJays
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BlueJays.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BlueJays.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BlueJays.csv"))

(defn Stat2Data-BrainpH
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BrainpH.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BrainpH.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BrainpH.csv"))

(defn Stat2Data-BreesPass
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BreesPass.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BreesPass.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BreesPass.csv"))

(defn Stat2Data-BritishUnions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BritishUnions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/BritishUnions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/BritishUnions.csv"))

(defn Stat2Data-ButterfliesBc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ButterfliesBc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ButterfliesBc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ButterfliesBc.csv"))

(defn Stat2Data-CAFE
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CAFE.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CAFE.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CAFE.csv"))

(defn Stat2Data-CalciumBP
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CalciumBP.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CalciumBP.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CalciumBP.csv"))

(defn Stat2Data-CanadianDrugs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CanadianDrugs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CanadianDrugs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CanadianDrugs.csv"))

(defn Stat2Data-CancerSurvival
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CancerSurvival.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CancerSurvival.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CancerSurvival.csv"))

(defn Stat2Data-Caterpillars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Caterpillars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Caterpillars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Caterpillars.csv"))

(defn Stat2Data-CavsShooting
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CavsShooting.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CavsShooting.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CavsShooting.csv"))

(defn Stat2Data-Cereal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Cereal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Cereal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Cereal.csv"))

(defn Stat2Data-ChemoTHC
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ChemoTHC.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ChemoTHC.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ChemoTHC.csv"))

(defn Stat2Data-ChildSpeaks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ChildSpeaks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ChildSpeaks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ChildSpeaks.csv"))

(defn Stat2Data-ClintonSanders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ClintonSanders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ClintonSanders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ClintonSanders.csv"))

(defn Stat2Data-Clothing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Clothing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Clothing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Clothing.csv"))

(defn Stat2Data-CloudSeeding
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CloudSeeding.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CloudSeeding.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CloudSeeding.csv"))

(defn Stat2Data-CloudSeeding2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CloudSeeding2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CloudSeeding2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CloudSeeding2.csv"))

(defn Stat2Data-CO2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CO2.csv"))

(defn Stat2Data-CO2Germany
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2Germany.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2Germany.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CO2Germany.csv"))

(defn Stat2Data-CO2Hawaii
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2Hawaii.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2Hawaii.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CO2Hawaii.csv"))

(defn Stat2Data-CO2SouthPole
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2SouthPole.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CO2SouthPole.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CO2SouthPole.csv"))

(defn Stat2Data-Contraceptives
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Contraceptives.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Contraceptives.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Contraceptives.csv"))

(defn Stat2Data-CountyHealth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CountyHealth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CountyHealth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CountyHealth.csv"))

(defn Stat2Data-CrabShip
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CrabShip.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CrabShip.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CrabShip.csv"))

(defn Stat2Data-CrackerFiber
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CrackerFiber.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CrackerFiber.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CrackerFiber.csv"))

(defn Stat2Data-CreditRisk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CreditRisk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/CreditRisk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/CreditRisk.csv"))

(defn Stat2Data-Cuckoo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Cuckoo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Cuckoo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Cuckoo.csv"))

(defn Stat2Data-Day1Survey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Day1Survey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Day1Survey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Day1Survey.csv"))

(defn Stat2Data-DiabeticDogs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/DiabeticDogs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/DiabeticDogs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/DiabeticDogs.csv"))

(defn Stat2Data-Diamonds
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Diamonds.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Diamonds.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Diamonds.csv"))

(defn Stat2Data-Diamonds2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Diamonds2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Diamonds2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Diamonds2.csv"))

(defn Stat2Data-Dinosaurs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Dinosaurs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Dinosaurs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Dinosaurs.csv"))

(defn Stat2Data-Election08
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Election08.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Election08.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv"))

(defn Stat2Data-Election16
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Election16.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Election16.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election16.csv"))

(defn Stat2Data-ElephantsFB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ElephantsFB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ElephantsFB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ElephantsFB.csv"))

(defn Stat2Data-ElephantsMF
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ElephantsMF.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ElephantsMF.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ElephantsMF.csv"))

(defn Stat2Data-Ethanol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Ethanol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Ethanol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Ethanol.csv"))

(defn Stat2Data-Eyes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Eyes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Eyes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Eyes.csv"))

(defn Stat2Data-Faces
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Faces.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Faces.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Faces.csv"))

(defn Stat2Data-FaithfulFaces
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FaithfulFaces.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FaithfulFaces.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FaithfulFaces.csv"))

(defn Stat2Data-FantasyBaseball
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FantasyBaseball.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FantasyBaseball.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FantasyBaseball.csv"))

(defn Stat2Data-FatRats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FatRats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FatRats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FatRats.csv"))

(defn Stat2Data-Fertility
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fertility.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fertility.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Fertility.csv"))

(defn Stat2Data-FGByDistance
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FGByDistance.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FGByDistance.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FGByDistance.csv"))

(defn Stat2Data-Film
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Film.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Film.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Film.csv"))

(defn Stat2Data-FinalFourIzzo
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourIzzo.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourIzzo.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourIzzo.csv"))

(defn Stat2Data-FinalFourIzzo17
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourIzzo17.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourIzzo17.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourIzzo17.csv"))

(defn Stat2Data-FinalFourLong
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourLong.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourLong.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourLong.csv"))

(defn Stat2Data-FinalFourLong17
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourLong17.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourLong17.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourLong17.csv"))

(defn Stat2Data-FinalFourShort
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourShort.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourShort.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourShort.csv"))

(defn Stat2Data-FinalFourShort17
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourShort17.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FinalFourShort17.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FinalFourShort17.csv"))

(defn Stat2Data-Fingers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fingers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fingers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Fingers.csv"))

(defn Stat2Data-FirstYearGPA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FirstYearGPA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FirstYearGPA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FirstYearGPA.csv"))

(defn Stat2Data-FishEggs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FishEggs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FishEggs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FishEggs.csv"))

(defn Stat2Data-Fitch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fitch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fitch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Fitch.csv"))

(defn Stat2Data-FlightResponse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FlightResponse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FlightResponse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FlightResponse.csv"))

(defn Stat2Data-FloridaDP
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FloridaDP.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FloridaDP.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FloridaDP.csv"))

(defn Stat2Data-Fluorescence
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fluorescence.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Fluorescence.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Fluorescence.csv"))

(defn Stat2Data-FranticFingers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FranticFingers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FranticFingers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FranticFingers.csv"))

(defn Stat2Data-FruitFlies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FruitFlies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FruitFlies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FruitFlies.csv"))

(defn Stat2Data-FruitFlies2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FruitFlies2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FruitFlies2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FruitFlies2.csv"))

(defn Stat2Data-FunnelDrop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FunnelDrop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/FunnelDrop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/FunnelDrop.csv"))

(defn Stat2Data-GlowWorms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/GlowWorms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/GlowWorms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/GlowWorms.csv"))

(defn Stat2Data-Goldenrod
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Goldenrod.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Goldenrod.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Goldenrod.csv"))

(defn Stat2Data-GrinnellHouses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/GrinnellHouses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/GrinnellHouses.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/GrinnellHouses.csv"))

(defn Stat2Data-Grocery
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Grocery.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Grocery.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Grocery.csv"))

(defn Stat2Data-Gunnels
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Gunnels.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Gunnels.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Gunnels.csv"))

(defn Stat2Data-Handwriting
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Handwriting.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Handwriting.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Handwriting.csv"))

(defn Stat2Data-Hawks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Hawks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Hawks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Hawks.csv"))

(defn Stat2Data-HawkTail
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HawkTail.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HawkTail.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HawkTail.csv"))

(defn Stat2Data-HawkTail2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HawkTail2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HawkTail2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HawkTail2.csv"))

(defn Stat2Data-HearingTest
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HearingTest.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HearingTest.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HearingTest.csv"))

(defn Stat2Data-HeatingOil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HeatingOil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HeatingOil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HeatingOil.csv"))

(defn Stat2Data-HighPeaks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HighPeaks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HighPeaks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HighPeaks.csv"))

(defn Stat2Data-Hoops
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Hoops.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Hoops.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Hoops.csv"))

(defn Stat2Data-HorsePrices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HorsePrices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HorsePrices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HorsePrices.csv"))

(defn Stat2Data-Houses
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Houses.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Houses.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Houses.csv"))

(defn Stat2Data-HousesNY
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HousesNY.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/HousesNY.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/HousesNY.csv"))

(defn Stat2Data-ICU
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ICU.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ICU.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ICU.csv"))

(defn Stat2Data-InfantMortality2010
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/InfantMortality2010.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/InfantMortality2010.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/InfantMortality2010.csv"))

(defn Stat2Data-Inflation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Inflation.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Inflation.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Inflation.csv"))

(defn Stat2Data-InsuranceVote
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/InsuranceVote.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/InsuranceVote.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/InsuranceVote.csv"))

(defn Stat2Data-IQGuessing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/IQGuessing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/IQGuessing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/IQGuessing.csv"))

(defn Stat2Data-Jurors
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Jurors.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Jurors.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Jurors.csv"))

(defn Stat2Data-Kershaw
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Kershaw.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Kershaw.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Kershaw.csv"))

(defn Stat2Data-KeyWestWater
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/KeyWestWater.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/KeyWestWater.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/KeyWestWater.csv"))

(defn Stat2Data-Kids198
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Kids198.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Kids198.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Kids198.csv"))

(defn Stat2Data-Leafhoppers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Leafhoppers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Leafhoppers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Leafhoppers.csv"))

(defn Stat2Data-LeafWidth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LeafWidth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LeafWidth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LeafWidth.csv"))

(defn Stat2Data-Leukemia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Leukemia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Leukemia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Leukemia.csv"))

(defn Stat2Data-LeveeFailures
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LeveeFailures.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LeveeFailures.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LeveeFailures.csv"))

(defn Stat2Data-LewyBody2Groups
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LewyBody2Groups.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LewyBody2Groups.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LewyBody2Groups.csv"))

(defn Stat2Data-LewyDLBad
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LewyDLBad.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LewyDLBad.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LewyDLBad.csv"))

(defn Stat2Data-LongJumpOlympics
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LongJumpOlympics.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LongJumpOlympics.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LongJumpOlympics.csv"))

(defn Stat2Data-LongJumpOlympics2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LongJumpOlympics2016.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LongJumpOlympics2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LongJumpOlympics2016.csv"))

(defn Stat2Data-LosingSleep
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LosingSleep.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LosingSleep.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LosingSleep.csv"))

(defn Stat2Data-LostLetter
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LostLetter.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/LostLetter.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/LostLetter.csv"))

(defn Stat2Data-Marathon
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Marathon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Marathon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Marathon.csv"))

(defn Stat2Data-Markets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Markets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Markets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Markets.csv"))

(defn Stat2Data-MathEnrollment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MathEnrollment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MathEnrollment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MathEnrollment.csv"))

(defn Stat2Data-MathPlacement
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MathPlacement.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MathPlacement.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MathPlacement.csv"))

(defn Stat2Data-MedGPA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MedGPA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MedGPA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MedGPA.csv"))

(defn Stat2Data-Meniscus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Meniscus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Meniscus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Meniscus.csv"))

(defn Stat2Data-MentalHealth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MentalHealth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MentalHealth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MentalHealth.csv"))

(defn Stat2Data-MetabolicRate
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetabolicRate.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetabolicRate.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MetabolicRate.csv"))

(defn Stat2Data-MetroCommutes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetroCommutes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetroCommutes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MetroCommutes.csv"))

(defn Stat2Data-MetroHealth83
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetroHealth83.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MetroHealth83.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MetroHealth83.csv"))

(defn Stat2Data-Migraines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Migraines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Migraines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Migraines.csv"))

(defn Stat2Data-Milgram
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Milgram.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Milgram.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Milgram.csv"))

(defn Stat2Data-MLB2007Standings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MLB2007Standings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MLB2007Standings.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MLB2007Standings.csv"))

(defn Stat2Data-MLBStandings2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MLBStandings2016.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MLBStandings2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MLBStandings2016.csv"))

(defn Stat2Data-MothEggs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MothEggs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MothEggs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MothEggs.csv"))

(defn Stat2Data-MouseBrain
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MouseBrain.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MouseBrain.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MouseBrain.csv"))

(defn Stat2Data-MusicTime
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MusicTime.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/MusicTime.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MusicTime.csv"))

(defn Stat2Data-NCbirths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NCbirths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NCbirths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/NCbirths.csv"))

(defn Stat2Data-NFL2007Standings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NFL2007Standings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NFL2007Standings.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/NFL2007Standings.csv"))

(defn Stat2Data-NFLStandings2016
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NFLStandings2016.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/NFLStandings2016.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/NFLStandings2016.csv"))

(defn Stat2Data-Nursing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Nursing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Nursing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Nursing.csv"))

(defn Stat2Data-OilDeapsorbtion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/OilDeapsorbtion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/OilDeapsorbtion.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/OilDeapsorbtion.csv"))

(defn Stat2Data-Olives
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Olives.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Olives.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Olives.csv"))

(defn Stat2Data-Orings
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Orings.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Orings.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Orings.csv"))

(defn Stat2Data-Overdrawn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Overdrawn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Overdrawn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Overdrawn.csv"))

(defn Stat2Data-Oysters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Oysters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Oysters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Oysters.csv"))

(defn Stat2Data-PalmBeach
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PalmBeach.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PalmBeach.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PalmBeach.csv"))

(defn Stat2Data-PeaceBridge2003
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PeaceBridge2003.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PeaceBridge2003.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PeaceBridge2003.csv"))

(defn Stat2Data-PeaceBridge2012
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PeaceBridge2012.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PeaceBridge2012.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PeaceBridge2012.csv"))

(defn Stat2Data-Pedometer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pedometer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pedometer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Pedometer.csv"))

(defn Stat2Data-Perch
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Perch.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Perch.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Perch.csv"))

(defn Stat2Data-PigFeed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PigFeed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PigFeed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PigFeed.csv"))

(defn Stat2Data-Pines
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pines.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pines.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Pines.csv"))

(defn Stat2Data-PKU
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PKU.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PKU.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PKU.csv"))

(defn Stat2Data-Political
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Political.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Political.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Political.csv"))

(defn Stat2Data-Pollster08
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pollster08.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pollster08.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Pollster08.csv"))

(defn Stat2Data-Popcorn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Popcorn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Popcorn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Popcorn.csv"))

(defn Stat2Data-PorscheJaguar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PorscheJaguar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PorscheJaguar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PorscheJaguar.csv"))

(defn Stat2Data-PorschePrice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PorschePrice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/PorschePrice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/PorschePrice.csv"))

(defn Stat2Data-Pulse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pulse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Pulse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Pulse.csv"))

(defn Stat2Data-Putts1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Putts1.csv"))

(defn Stat2Data-Putts2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Putts2.csv"))

(defn Stat2Data-Putts3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Putts3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Putts3.csv"))

(defn Stat2Data-RacialAnimus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RacialAnimus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RacialAnimus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RacialAnimus.csv"))

(defn Stat2Data-RadioactiveTwins
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RadioactiveTwins.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RadioactiveTwins.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RadioactiveTwins.csv"))

(defn Stat2Data-RailsTrails
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RailsTrails.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RailsTrails.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RailsTrails.csv"))

(defn Stat2Data-Rectangles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Rectangles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Rectangles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Rectangles.csv"))

(defn Stat2Data-ReligionGDP
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ReligionGDP.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ReligionGDP.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ReligionGDP.csv"))

(defn Stat2Data-RepeatedPulse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RepeatedPulse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RepeatedPulse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RepeatedPulse.csv"))

(defn Stat2Data-ResidualOil
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ResidualOil.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ResidualOil.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ResidualOil.csv"))

(defn Stat2Data-Retirement
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Retirement.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Retirement.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Retirement.csv"))

(defn Stat2Data-Ricci
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Ricci.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Ricci.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Ricci.csv"))

(defn Stat2Data-RiverElements
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RiverElements.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RiverElements.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RiverElements.csv"))

(defn Stat2Data-RiverIron
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RiverIron.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/RiverIron.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/RiverIron.csv"))

(defn Stat2Data-SampleFG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SampleFG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SampleFG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SampleFG.csv"))

(defn Stat2Data-SandwichAnts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SandwichAnts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SandwichAnts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SandwichAnts.csv"))

(defn Stat2Data-SATGPA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SATGPA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SATGPA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SATGPA.csv"))

(defn Stat2Data-SeaIce
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SeaIce.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SeaIce.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SeaIce.csv"))

(defn Stat2Data-SeaSlugs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SeaSlugs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SeaSlugs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SeaSlugs.csv"))

(defn Stat2Data-SleepingShrews
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SleepingShrews.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SleepingShrews.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SleepingShrews.csv"))

(defn Stat2Data-Sparrows
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Sparrows.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Sparrows.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Sparrows.csv"))

(defn Stat2Data-SpeciesArea
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SpeciesArea.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SpeciesArea.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SpeciesArea.csv"))

(defn Stat2Data-Speed
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Speed.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Speed.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Speed.csv"))

(defn Stat2Data-SugarEthanol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SugarEthanol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SugarEthanol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SugarEthanol.csv"))

(defn Stat2Data-SuicideChina
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SuicideChina.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/SuicideChina.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/SuicideChina.csv"))

(defn Stat2Data-Swahili
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Swahili.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Swahili.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Swahili.csv"))

(defn Stat2Data-Tadpoles
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Tadpoles.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Tadpoles.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Tadpoles.csv"))

(defn Stat2Data-TechStocks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TechStocks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TechStocks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TechStocks.csv"))

(defn Stat2Data-TeenPregnancy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TeenPregnancy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TeenPregnancy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TeenPregnancy.csv"))

(defn Stat2Data-TextPrices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TextPrices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TextPrices.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TextPrices.csv"))

(defn Stat2Data-ThomasConfirmation
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThomasConfirmation.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThomasConfirmation.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ThomasConfirmation.csv"))

(defn Stat2Data-ThreeCars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThreeCars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThreeCars.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ThreeCars.csv"))

(defn Stat2Data-ThreeCars2017
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThreeCars2017.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/ThreeCars2017.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/ThreeCars2017.csv"))

(defn Stat2Data-TipJoke
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TipJoke.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TipJoke.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TipJoke.csv"))

(defn Stat2Data-Titanic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Titanic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Titanic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Titanic.csv"))

(defn Stat2Data-TMS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TMS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TMS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TMS.csv"))

(defn Stat2Data-TomlinsonRush
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TomlinsonRush.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TomlinsonRush.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TomlinsonRush.csv"))

(defn Stat2Data-TwinsLungs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TwinsLungs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/TwinsLungs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/TwinsLungs.csv"))

(defn Stat2Data-Undoing
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Undoing.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Undoing.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Undoing.csv"))

(defn Stat2Data-USstamps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/USstamps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/USstamps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/USstamps.csv"))

(defn Stat2Data-VisualVerbal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/VisualVerbal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/VisualVerbal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/VisualVerbal.csv"))

(defn Stat2Data-Volts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Volts.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Volts.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Volts.csv"))

(defn Stat2Data-WalkingBabies
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WalkingBabies.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WalkingBabies.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WalkingBabies.csv"))

(defn Stat2Data-WalkTheDogs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WalkTheDogs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WalkTheDogs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WalkTheDogs.csv"))

(defn Stat2Data-WeightLossIncentive
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WeightLossIncentive.csv"))

(defn Stat2Data-WeightLossIncentive4
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive4.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive4.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WeightLossIncentive4.csv"))

(defn Stat2Data-WeightLossIncentive7
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive7.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WeightLossIncentive7.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WeightLossIncentive7.csv"))

(defn Stat2Data-Whickham2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Whickham2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Whickham2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Whickham2.csv"))

(defn Stat2Data-WordMemory
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WordMemory.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WordMemory.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WordMemory.csv"))

(defn Stat2Data-WordsWithFriends
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WordsWithFriends.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/WordsWithFriends.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/WordsWithFriends.csv"))

(defn Stat2Data-Wrinkle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Wrinkle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Wrinkle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Wrinkle.csv"))

(defn Stat2Data-YouthRisk
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/YouthRisk.csv"))

(defn Stat2Data-YouthRisk2007
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk2007.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk2007.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/YouthRisk2007.csv"))

(defn Stat2Data-YouthRisk2009
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk2009.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/YouthRisk2009.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/YouthRisk2009.csv"))

(defn Stat2Data-Zimmerman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Zimmerman.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/Stat2Data/Zimmerman.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Zimmerman.csv"))

(defn stevedata-af_crime93
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/af_crime93.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/af_crime93.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/af_crime93.csv"))

(defn stevedata-african_coups
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/african_coups.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/african_coups.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/african_coups.csv"))

(defn stevedata-aluminum_premiums
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/aluminum_premiums.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/aluminum_premiums.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/aluminum_premiums.csv"))

(defn stevedata-anes_partytherms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_partytherms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_partytherms.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/anes_partytherms.csv"))

(defn stevedata-anes_prochoice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_prochoice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_prochoice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/anes_prochoice.csv"))

(defn stevedata-anes_vote84
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_vote84.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/anes_vote84.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/anes_vote84.csv"))

(defn stevedata-Arca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Arca.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Arca.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Arca.csv"))

(defn stevedata-arcticseaice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/arcticseaice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/arcticseaice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/arcticseaice.csv"))

(defn stevedata-arg_tariff
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/arg_tariff.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/arg_tariff.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/arg_tariff.csv"))

(defn stevedata-asn_stats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/asn_stats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/asn_stats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/asn_stats.csv"))

(defn stevedata-CFT15
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/CFT15.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/CFT15.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/CFT15.csv"))

(defn stevedata-clemson_temps
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/clemson_temps.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/clemson_temps.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/clemson_temps.csv"))

(defn stevedata-co2emissions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/co2emissions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/co2emissions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/co2emissions.csv"))

(defn stevedata-coffee_imports
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/coffee_imports.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/coffee_imports.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/coffee_imports.csv"))

(defn stevedata-coffee_price
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/coffee_price.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/coffee_price.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/coffee_price.csv"))

(defn stevedata-commodity_prices
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/commodity_prices.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/commodity_prices.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/commodity_prices.csv"))

(defn stevedata-CP77
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/CP77.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/CP77.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/CP77.csv"))

(defn stevedata-DAPO
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DAPO.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DAPO.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/DAPO.csv"))

(defn stevedata-Datasaurus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Datasaurus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Datasaurus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Datasaurus.csv"))

(defn stevedata-DCE12
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DCE12.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DCE12.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/DCE12.csv"))

(defn stevedata-Dee04
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Dee04.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Dee04.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Dee04.csv"))

(defn stevedata-DJIA
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DJIA.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DJIA.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/DJIA.csv"))

(defn stevedata-DST
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DST.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/DST.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/DST.csv"))

(defn stevedata-EBJ
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/EBJ.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/EBJ.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/EBJ.csv"))

(defn stevedata-eight_schools
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eight_schools.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eight_schools.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/eight_schools.csv"))

(defn stevedata-election_turnout
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/election_turnout.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/election_turnout.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/election_turnout.csv"))

(defn stevedata-eq_passengercars
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eq_passengercars.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eq_passengercars.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/eq_passengercars.csv"))

(defn stevedata-ESS10NO
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESS10NO.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESS10NO.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ESS10NO.csv"))

(defn stevedata-ESS9GB
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESS9GB.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESS9GB.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ESS9GB.csv"))

(defn stevedata-ESSBE5
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESSBE5.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ESSBE5.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ESSBE5.csv"))

(defn stevedata-eurostat_codes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eurostat_codes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eurostat_codes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/eurostat_codes.csv"))

(defn stevedata-eustates
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eustates.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/eustates.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/eustates.csv"))

(defn stevedata-fakeAPI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeAPI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeAPI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/fakeAPI.csv"))

(defn stevedata-fakeHappiness
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeHappiness.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeHappiness.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/fakeHappiness.csv"))

(defn stevedata-fakeLogit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeLogit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeLogit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/fakeLogit.csv"))

(defn stevedata-fakeTSCS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeTSCS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeTSCS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/fakeTSCS.csv"))

(defn stevedata-fakeTSD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeTSD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/fakeTSD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/fakeTSD.csv"))

(defn stevedata-ghp100k
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ghp100k.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ghp100k.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ghp100k.csv"))

(defn stevedata-GHR04
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/GHR04.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/GHR04.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/GHR04.csv"))

(defn stevedata-gss_abortion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_abortion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_abortion.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/gss_abortion.csv"))

(defn stevedata-gss_spending
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_spending.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_spending.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/gss_spending.csv"))

(defn stevedata-gss_wages
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_wages.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/gss_wages.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/gss_wages.csv"))

(defn stevedata-Guber99
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Guber99.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Guber99.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Guber99.csv"))

(defn stevedata-illiteracy30
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/illiteracy30.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/illiteracy30.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/illiteracy30.csv"))

(defn stevedata-inglehart03
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/inglehart03.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/inglehart03.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/inglehart03.csv"))

(defn stevedata-Lipset59
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Lipset59.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Lipset59.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Lipset59.csv"))

(defn stevedata-LOTI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LOTI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LOTI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/LOTI.csv"))

(defn stevedata-LTPT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LTPT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LTPT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/LTPT.csv"))

(defn stevedata-LTWT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LTWT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/LTWT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/LTWT.csv"))

(defn stevedata-min_wage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/min_wage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/min_wage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/min_wage.csv"))

(defn stevedata-mm_mlda
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mm_mlda.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mm_mlda.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/mm_mlda.csv"))

(defn stevedata-mm_nhis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mm_nhis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mm_nhis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/mm_nhis.csv"))

(defn stevedata-mvprod
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mvprod.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/mvprod.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/mvprod.csv"))

(defn stevedata-nesarc_drinkspd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/nesarc_drinkspd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/nesarc_drinkspd.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/nesarc_drinkspd.csv"))

(defn stevedata-Newhouse77
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Newhouse77.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Newhouse77.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Newhouse77.csv"))

(defn stevedata-ODGI
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ODGI.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ODGI.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ODGI.csv"))

(defn stevedata-OODTPT
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/OODTPT.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/OODTPT.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/OODTPT.csv"))

(defn stevedata-PPGE
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/PPGE.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/PPGE.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/PPGE.csv"))

(defn stevedata-PRDEG
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/PRDEG.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/PRDEG.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/PRDEG.csv"))

(defn stevedata-Presidents
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Presidents.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/Presidents.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/Presidents.csv"))

(defn stevedata-pwt_sample
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/pwt_sample.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/pwt_sample.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/pwt_sample.csv"))

(defn stevedata-quartets
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/quartets.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/quartets.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/quartets.csv"))

(defn stevedata-recessions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/recessions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/recessions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/recessions.csv"))

(defn stevedata-SBCD
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/SBCD.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/SBCD.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/SBCD.csv"))

(defn stevedata-scb_regions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/scb_regions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/scb_regions.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/scb_regions.csv"))

(defn stevedata-SCP16
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/SCP16.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/SCP16.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/SCP16.csv"))

(defn stevedata-sealevels
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sealevels.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sealevels.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/sealevels.csv"))

(defn stevedata-so2concentrations
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/so2concentrations.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/so2concentrations.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/so2concentrations.csv"))

(defn stevedata-states_war
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/states_war.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/states_war.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/states_war.csv"))

(defn stevedata-steves_clothes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/steves_clothes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/steves_clothes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/steves_clothes.csv"))

(defn stevedata-sugar_price
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sugar_price.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sugar_price.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/sugar_price.csv"))

(defn stevedata-sweden_counties
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sweden_counties.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/sweden_counties.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/sweden_counties.csv"))

(defn stevedata-thatcher_approval
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/thatcher_approval.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/thatcher_approval.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/thatcher_approval.csv"))

(defn stevedata-therms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/therms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/therms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/therms.csv"))

(defn stevedata-turnips
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/turnips.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/turnips.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/turnips.csv"))

(defn stevedata-TV16
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/TV16.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/TV16.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/TV16.csv"))

(defn stevedata-ukg_eeri
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ukg_eeri.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/ukg_eeri.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/ukg_eeri.csv"))

(defn stevedata-uniondensity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/uniondensity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/uniondensity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/uniondensity.csv"))

(defn stevedata-usa_chn_gdp_forecasts
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_chn_gdp_forecasts.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_chn_gdp_forecasts.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/usa_chn_gdp_forecasts.csv"))

(defn stevedata-usa_computers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_computers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_computers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/usa_computers.csv"))

(defn stevedata-usa_migration
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_migration.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_migration.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/usa_migration.csv"))

(defn stevedata-usa_states
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_states.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_states.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/usa_states.csv"))

(defn stevedata-usa_tradegdp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_tradegdp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/usa_tradegdp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/usa_tradegdp.csv"))

(defn stevedata-USFAHR
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/USFAHR.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/USFAHR.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/USFAHR.csv"))

(defn stevedata-voteincome
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/voteincome.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/voteincome.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/voteincome.csv"))

(defn stevedata-wbd_example
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wbd_example.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wbd_example.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wbd_example.csv"))

(defn stevedata-wvs_ccodes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_ccodes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_ccodes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_ccodes.csv"))

(defn stevedata-wvs_immig
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_immig.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_immig.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_immig.csv"))

(defn stevedata-wvs_justifbribe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_justifbribe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_justifbribe.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_justifbribe.csv"))

(defn stevedata-wvs_usa_abortion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_abortion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_abortion.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_usa_abortion.csv"))

(defn stevedata-wvs_usa_educat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_educat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_educat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_usa_educat.csv"))

(defn stevedata-wvs_usa_regions
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_regions.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/wvs_usa_regions.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/wvs_usa_regions.csv"))

(defn stevedata-yugo_sales
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/yugo_sales.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/stevedata/yugo_sales.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/stevedata/yugo_sales.csv"))

(defn survival-amlcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/aml.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/aml.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/aml.csv"))

(defn survival-bladdercancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/bladder.csv"))

(defn survival-bladder1cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/bladder1.csv"))

(defn survival-bladder2cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/bladder2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/bladder2.csv"))

(defn survival-cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/cancer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/cancer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cancer.csv"))

(defn survival-capacitorreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/capacitor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/capacitor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/capacitor.csv"))

(defn survival-cgd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/cgd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/cgd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cgd.csv"))

(defn survival-cgd0cgd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/cgd0.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/cgd0.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cgd0.csv"))

(defn survival-coloncancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/colon.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/colon.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/colon.csv"))

(defn survival-cracksreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/cracks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/cracks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/cracks.csv"))

(defn survival-diabetic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/diabetic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/diabetic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/diabetic.csv"))

(defn survival-flchain
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/flchain.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/flchain.csv"))

(defn survival-gbsgcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/gbsg.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/gbsg.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/gbsg.csv"))

(defn survival-genfanreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/genfan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/genfan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/genfan.csv"))

(defn survival-heart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/heart.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/heart.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/heart.csv"))

(defn survival-hoelcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/hoel.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/hoel.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/hoel.csv"))

(defn survival-ifluidreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/ifluid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/ifluid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/ifluid.csv"))

(defn survival-imotorreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/imotor.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/imotor.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/imotor.csv"))

(defn survival-jasaheart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/jasa.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/jasa.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/jasa.csv"))

(defn survival-jasa1heart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/jasa1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/jasa1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/jasa1.csv"))

(defn survival-kidneycancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/kidney.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/kidney.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/kidney.csv"))

(defn survival-leukemiacancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/leukemia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/leukemia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/leukemia.csv"))

(defn survival-logan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/logan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/logan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/logan.csv"))

(defn survival-lungcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/lung.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/lung.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/lung.csv"))

(defn survival-mguscancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/mgus.csv"))

(defn survival-mgus1cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/mgus1.csv"))

(defn survival-mgus2cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/mgus2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/mgus2.csv"))

(defn survival-myeloidcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/myeloid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/myeloid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/myeloid.csv"))

(defn survival-myelomacancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/myeloma.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/myeloma.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/myeloma.csv"))

(defn survival-nafld1nafld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/nafld1.csv"))

(defn survival-nafld2nafld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/nafld2.csv"))

(defn survival-nafld3nafld
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/nafld3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/nafld3.csv"))

(defn survival-nwtco
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/nwtco.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/nwtco.csv"))

(defn survival-ovariancancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/ovarian.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/ovarian.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/ovarian.csv"))

(defn survival-pbc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/pbc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/pbc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbc.csv"))

(defn survival-pbcseqpbc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/pbcseq.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/pbcseq.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/pbcseq.csv"))

(defn survival-ratscancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/rats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/rats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rats.csv"))

(defn survival-rats2cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/rats2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/rats2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rats2.csv"))

(defn survival-retinopathy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/retinopathy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/retinopathy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/retinopathy.csv"))

(defn survival-rhDNase
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/rhDNase.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/rhDNase.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rhDNase.csv"))

(defn survival-rotterdamcancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/rotterdam.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/rotterdam.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/rotterdam.csv"))

(defn survival-solder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/solder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/solder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/solder.csv"))

(defn survival-stanford2heart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/stanford2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/stanford2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/stanford2.csv"))

(defn survival-survexp.mnsurvexp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.mn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.mn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/survexp.mn.csv"))

(defn survival-survexp.ussurvexp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.us.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.us.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/survexp.us.csv"))

(defn survival-survexp.usrsurvexp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.usr.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/survexp.usr.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/survexp.usr.csv"))

(defn survival-tobin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/tobin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/tobin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/tobin.csv"))

(defn survival-transplant
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/transplant.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/transplant.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/transplant.csv"))

(defn survival-turbinereliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/turbine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/turbine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/turbine.csv"))

(defn survival-udca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/udca.csv"))

(defn survival-udca1udca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/udca1.csv"))

(defn survival-udca2udca
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/udca2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/udca2.csv"))

(defn survival-uspop2survexp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/uspop2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/uspop2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/uspop2.csv"))

(defn survival-valveSeatreliability
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/valveSeat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/valveSeat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/valveSeat.csv"))

(defn survival-veterancancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/survival/veteran.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/survival/veteran.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/survival/veteran.csv"))

(defn texmex-liver
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/liver.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/liver.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/liver.csv"))

(defn texmex-nidd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/nidd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/nidd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/nidd.csv"))

(defn texmex-portpirie
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/portpirie.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/portpirie.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/portpirie.csv"))

(defn texmex-rain
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/rain.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/rain.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/rain.csv"))

(defn texmex-summer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/summer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/summer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/summer.csv"))

(defn texmex-wavesurge
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/wavesurge.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/wavesurge.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/wavesurge.csv"))

(defn texmex-winter
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/texmex/winter.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/texmex/winter.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/texmex/winter.csv"))

(defn tidyr-billboard
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/billboard.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/billboard.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/billboard.csv"))

(defn tidyr-cms_patient_care
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/cms_patient_care.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/cms_patient_care.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/cms_patient_care.csv"))

(defn tidyr-cms_patient_experience
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/cms_patient_experience.html"
  {:doc-link
   "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/cms_patient_experience.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/cms_patient_experience.csv"))

(defn tidyr-construction
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/construction.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/construction.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/construction.csv"))

(defn tidyr-fish_encounters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/fish_encounters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/fish_encounters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/fish_encounters.csv"))

(defn tidyr-household
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/household.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/household.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/household.csv"))

(defn tidyr-population
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/population.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/population.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/population.csv"))

(defn tidyr-relig_income
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/relig_income.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/relig_income.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/relig_income.csv"))

(defn tidyr-smiths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/smiths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/smiths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/smiths.csv"))

(defn tidyr-table1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table1.csv"))

(defn tidyr-table2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table2.csv"))

(defn tidyr-table3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table3.csv"))

(defn tidyr-table4a
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table4a.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table4a.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table4a.csv"))

(defn tidyr-table4b
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table4b.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table4b.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table4b.csv"))

(defn tidyr-table5
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table5.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/table5.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/table5.csv"))

(defn tidyr-us_rent_income
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/us_rent_income.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/us_rent_income.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/us_rent_income.csv"))

(defn tidyr-who
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/who.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/who.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/who.csv"))

(defn tidyr-who2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/who2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/who2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/who2.csv"))

(defn tidyr-world_bank_pop
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/world_bank_pop.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tidyr/world_bank_pop.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tidyr/world_bank_pop.csv"))

(defn tsibble-pedestrian
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibble/pedestrian.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibble/pedestrian.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibble/pedestrian.csv"))

(defn tsibble-tourism
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibble/tourism.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibble/tourism.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibble/tourism.csv"))

(defn tsibbledata-ansett
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/ansett.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/ansett.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/ansett.csv"))

(defn tsibbledata-aus_livestock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_livestock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_livestock.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/aus_livestock.csv"))

(defn tsibbledata-aus_production
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_production.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_production.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/aus_production.csv"))

(defn tsibbledata-aus_retail
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_retail.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/aus_retail.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/aus_retail.csv"))

(defn tsibbledata-gafa_stock
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/gafa_stock.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/gafa_stock.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/gafa_stock.csv"))

(defn tsibbledata-global_economy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/global_economy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/global_economy.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/global_economy.csv"))

(defn tsibbledata-hh_budget
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/hh_budget.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/hh_budget.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/hh_budget.csv"))

(defn tsibbledata-nyc_bikes
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/nyc_bikes.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/nyc_bikes.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/nyc_bikes.csv"))

(defn tsibbledata-olympic_running
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/olympic_running.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/olympic_running.html"}
  []
  (fetch-dataset
    "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/olympic_running.csv"))

(defn tsibbledata-PBS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/PBS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/PBS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/PBS.csv"))

(defn tsibbledata-pelt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/pelt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/pelt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/pelt.csv"))

(defn tsibbledata-vic_elec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/vic_elec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/tsibbledata/vic_elec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/tsibbledata/vic_elec.csv"))

(defn validate-nace_rev2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/validate/nace_rev2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/validate/nace_rev2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/validate/nace_rev2.csv"))

(defn validate-retailers
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/validate/retailers.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/validate/retailers.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/validate/retailers.csv"))

(defn validate-samplonomy
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/validate/samplonomy.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/validate/samplonomy.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/validate/samplonomy.csv"))

(defn validate-SBS2000
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/validate/SBS2000.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/validate/SBS2000.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/validate/SBS2000.csv"))

(defn vcd-Arthritis
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Arthritis.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Arthritis.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Arthritis.csv"))

(defn vcd-Baseball
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Baseball.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Baseball.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Baseball.csv"))

(defn vcd-BrokenMarriage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/BrokenMarriage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/BrokenMarriage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/BrokenMarriage.csv"))

(defn vcd-Bundesliga
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Bundesliga.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Bundesliga.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Bundesliga.csv"))

(defn vcd-Bundestag2005
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Bundestag2005.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Bundestag2005.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Bundestag2005.csv"))

(defn vcd-Butterfly
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Butterfly.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Butterfly.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Butterfly.csv"))

(defn vcd-CoalMiners
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/CoalMiners.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/CoalMiners.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/CoalMiners.csv"))

(defn vcd-DanishWelfare
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/DanishWelfare.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/DanishWelfare.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/DanishWelfare.csv"))

(defn vcd-Employment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Employment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Employment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Employment.csv"))

(defn vcd-Federalist
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Federalist.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Federalist.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Federalist.csv"))

(defn vcd-Hitters
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Hitters.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Hitters.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Hitters.csv"))

(defn vcd-HorseKicks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/HorseKicks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/HorseKicks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/HorseKicks.csv"))

(defn vcd-Hospital
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Hospital.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Hospital.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Hospital.csv"))

(defn vcd-JobSatisfaction
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/JobSatisfaction.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/JobSatisfaction.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/JobSatisfaction.csv"))

(defn vcd-JointSports
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/JointSports.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/JointSports.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/JointSports.csv"))

(defn vcd-Lifeboats
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Lifeboats.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Lifeboats.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Lifeboats.csv"))

(defn vcd-MSPatients
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/MSPatients.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/MSPatients.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/MSPatients.csv"))

(defn vcd-NonResponse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/NonResponse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/NonResponse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/NonResponse.csv"))

(defn vcd-OvaryCancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/OvaryCancer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/OvaryCancer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/OvaryCancer.csv"))

(defn vcd-PreSex
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/PreSex.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/PreSex.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/PreSex.csv"))

(defn vcd-Punishment
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Punishment.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Punishment.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Punishment.csv"))

(defn vcd-RepVict
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/RepVict.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/RepVict.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/RepVict.csv"))

(defn vcd-Rochdale
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Rochdale.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Rochdale.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Rochdale.csv"))

(defn vcd-Saxony
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Saxony.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Saxony.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Saxony.csv"))

(defn vcd-SexualFun
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/SexualFun.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/SexualFun.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/SexualFun.csv"))

(defn vcd-SpaceShuttle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/SpaceShuttle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/SpaceShuttle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/SpaceShuttle.csv"))

(defn vcd-Suicide
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Suicide.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Suicide.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Suicide.csv"))

(defn vcd-Trucks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Trucks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/Trucks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/Trucks.csv"))

(defn vcd-UKSoccer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/UKSoccer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/UKSoccer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/UKSoccer.csv"))

(defn vcd-VisualAcuity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/VisualAcuity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/VisualAcuity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/VisualAcuity.csv"))

(defn vcd-VonBort
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/VonBort.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/VonBort.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/VonBort.csv"))

(defn vcd-WeldonDice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/WeldonDice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/WeldonDice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/WeldonDice.csv"))

(defn vcd-WomenQueue
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcd/WomenQueue.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcd/WomenQueue.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcd/WomenQueue.csv"))

(defn vcdExtra-Abortion
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Abortion.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Abortion.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Abortion.csv"))

(defn vcdExtra-Accident
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Accident.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Accident.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Accident.csv"))

(defn vcdExtra-AirCrash
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/AirCrash.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/AirCrash.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/AirCrash.csv"))

(defn vcdExtra-Alligator
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Alligator.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Alligator.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Alligator.csv"))

(defn vcdExtra-Asbestos
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Asbestos.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Asbestos.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Asbestos.csv"))

(defn vcdExtra-Bartlett
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Bartlett.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Bartlett.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Bartlett.csv"))

(defn vcdExtra-Burt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Burt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Burt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Burt.csv"))

(defn vcdExtra-Caesar
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Caesar.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Caesar.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Caesar.csv"))

(defn vcdExtra-Cancer
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Cancer.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Cancer.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Cancer.csv"))

(defn vcdExtra-Cormorants
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Cormorants.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Cormorants.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Cormorants.csv"))

(defn vcdExtra-CyclingDeaths
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/CyclingDeaths.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/CyclingDeaths.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/CyclingDeaths.csv"))

(defn vcdExtra-DaytonSurvey
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/DaytonSurvey.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/DaytonSurvey.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/DaytonSurvey.csv"))

(defn vcdExtra-Depends
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Depends.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Depends.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Depends.csv"))

(defn vcdExtra-Detergent
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Detergent.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Detergent.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Detergent.csv"))

(defn vcdExtra-Donner
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Donner.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Donner.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Donner.csv"))

(defn vcdExtra-Draft1970
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Draft1970.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Draft1970.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Draft1970.csv"))

(defn vcdExtra-Draft1970table
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Draft1970table.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Draft1970table.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Draft1970table.csv"))

(defn vcdExtra-Dyke
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Dyke.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Dyke.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Dyke.csv"))

(defn vcdExtra-Fungicide
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Fungicide.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Fungicide.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Fungicide.csv"))

(defn vcdExtra-Geissler
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Geissler.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Geissler.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Geissler.csv"))

(defn vcdExtra-Gilby
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Gilby.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Gilby.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Gilby.csv"))

(defn vcdExtra-Glass
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Glass.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Glass.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Glass.csv"))

(defn vcdExtra-GSS
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/GSS.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/GSS.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/GSS.csv"))

(defn vcdExtra-HairEyePlace
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HairEyePlace.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HairEyePlace.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/HairEyePlace.csv"))

(defn vcdExtra-Hauser79
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Hauser79.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Hauser79.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Hauser79.csv"))

(defn vcdExtra-Heart
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Heart.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Heart.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Heart.csv"))

(defn vcdExtra-Heckman
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Heckman.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Heckman.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Heckman.csv"))

(defn vcdExtra-HospVisits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HospVisits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HospVisits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/HospVisits.csv"))

(defn vcdExtra-HouseTasks
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HouseTasks.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/HouseTasks.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/HouseTasks.csv"))

(defn vcdExtra-Hoyt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Hoyt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Hoyt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Hoyt.csv"))

(defn vcdExtra-ICU
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/ICU.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/ICU.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/ICU.csv"))

(defn vcdExtra-JobSat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/JobSat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/JobSat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/JobSat.csv"))

(defn vcdExtra-Mammograms
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mammograms.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mammograms.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Mammograms.csv"))

(defn vcdExtra-Mental
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mental.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mental.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Mental.csv"))

(defn vcdExtra-Mice
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mice.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mice.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Mice.csv"))

(defn vcdExtra-Mobility
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mobility.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Mobility.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Mobility.csv"))

(defn vcdExtra-PhdPubs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/PhdPubs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/PhdPubs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/PhdPubs.csv"))

(defn vcdExtra-ShakeWords
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/ShakeWords.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/ShakeWords.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/ShakeWords.csv"))

(defn vcdExtra-Titanicp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Titanicp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Titanicp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Titanicp.csv"))

(defn vcdExtra-Toxaemia
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Toxaemia.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Toxaemia.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Toxaemia.csv"))

(defn vcdExtra-TV
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/TV.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/TV.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/TV.csv"))

(defn vcdExtra-Vietnam
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Vietnam.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Vietnam.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Vietnam.csv"))

(defn vcdExtra-Vote1980
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Vote1980.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Vote1980.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Vote1980.csv"))

(defn vcdExtra-WorkerSat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/WorkerSat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/WorkerSat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/WorkerSat.csv"))

(defn vcdExtra-Yamaguchi87
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Yamaguchi87.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/vcdExtra/Yamaguchi87.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/vcdExtra/Yamaguchi87.csv"))

(defn wooldridge-admnrev
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/admnrev.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/admnrev.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/admnrev.csv"))

(defn wooldridge-affairs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/affairs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/affairs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/affairs.csv"))

(defn wooldridge-airfare
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/airfare.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/airfare.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/airfare.csv"))

(defn wooldridge-alcohol
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/alcohol.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/alcohol.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/alcohol.csv"))

(defn wooldridge-apple
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/apple.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/apple.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/apple.csv"))

(defn wooldridge-approval
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/approval.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/approval.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/approval.csv"))

(defn wooldridge-athlet1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/athlet1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/athlet1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/athlet1.csv"))

(defn wooldridge-athlet2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/athlet2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/athlet2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/athlet2.csv"))

(defn wooldridge-attend
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/attend.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/attend.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/attend.csv"))

(defn wooldridge-audit
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/audit.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/audit.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/audit.csv"))

(defn wooldridge-barium
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/barium.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/barium.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/barium.csv"))

(defn wooldridge-beauty
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/beauty.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/beauty.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/beauty.csv"))

(defn wooldridge-benefits
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/benefits.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/benefits.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/benefits.csv"))

(defn wooldridge-beveridge
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/beveridge.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/beveridge.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/beveridge.csv"))

(defn wooldridge-big9salary
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/big9salary.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/big9salary.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/big9salary.csv"))

(defn wooldridge-bwght
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/bwght.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/bwght.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/bwght.csv"))

(defn wooldridge-bwght2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/bwght2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/bwght2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/bwght2.csv"))

(defn wooldridge-campus
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/campus.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/campus.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/campus.csv"))

(defn wooldridge-card
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/card.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/card.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/card.csv"))

(defn wooldridge-catholic
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/catholic.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/catholic.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/catholic.csv"))

(defn wooldridge-cement
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cement.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cement.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/cement.csv"))

(defn wooldridge-census2000
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/census2000.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/census2000.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/census2000.csv"))

(defn wooldridge-ceosal1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ceosal1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ceosal1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/ceosal1.csv"))

(defn wooldridge-ceosal2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ceosal2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ceosal2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/ceosal2.csv"))

(defn wooldridge-charity
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/charity.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/charity.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/charity.csv"))

(defn wooldridge-consump
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/consump.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/consump.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/consump.csv"))

(defn wooldridge-corn
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/corn.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/corn.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/corn.csv"))

(defn wooldridge-countymurders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/countymurders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/countymurders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/countymurders.csv"))

(defn wooldridge-cps78_85
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cps78_85.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cps78_85.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/cps78_85.csv"))

(defn wooldridge-cps91
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cps91.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/cps91.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/cps91.csv"))

(defn wooldridge-crime1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/crime1.csv"))

(defn wooldridge-crime2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/crime2.csv"))

(defn wooldridge-crime3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/crime3.csv"))

(defn wooldridge-crime4
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime4.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/crime4.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/crime4.csv"))

(defn wooldridge-discrim
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/discrim.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/discrim.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/discrim.csv"))

(defn wooldridge-driving
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/driving.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/driving.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/driving.csv"))

(defn wooldridge-earns
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/earns.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/earns.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/earns.csv"))

(defn wooldridge-econmath
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/econmath.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/econmath.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/econmath.csv"))

(defn wooldridge-elem94_95
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/elem94_95.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/elem94_95.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/elem94_95.csv"))

(defn wooldridge-engin
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/engin.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/engin.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/engin.csv"))

(defn wooldridge-expendshares
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/expendshares.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/expendshares.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/expendshares.csv"))

(defn wooldridge-ezanders
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ezanders.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ezanders.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/ezanders.csv"))

(defn wooldridge-ezunem
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ezunem.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ezunem.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/ezunem.csv"))

(defn wooldridge-fair
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fair.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fair.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fair.csv"))

(defn wooldridge-fertil1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fertil1.csv"))

(defn wooldridge-fertil2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fertil2.csv"))

(defn wooldridge-fertil3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fertil3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fertil3.csv"))

(defn wooldridge-fish
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fish.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fish.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fish.csv"))

(defn wooldridge-fringe
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fringe.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/fringe.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/fringe.csv"))

(defn wooldridge-gpa1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/gpa1.csv"))

(defn wooldridge-gpa2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/gpa2.csv"))

(defn wooldridge-gpa3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/gpa3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/gpa3.csv"))

(defn wooldridge-happiness
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/happiness.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/happiness.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/happiness.csv"))

(defn wooldridge-hprice1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/hprice1.csv"))

(defn wooldridge-hprice2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/hprice2.csv"))

(defn wooldridge-hprice3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hprice3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/hprice3.csv"))

(defn wooldridge-hseinv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hseinv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/hseinv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/hseinv.csv"))

(defn wooldridge-htv
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/htv.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/htv.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/htv.csv"))

(defn wooldridge-infmrt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/infmrt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/infmrt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/infmrt.csv"))

(defn wooldridge-injury
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/injury.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/injury.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/injury.csv"))

(defn wooldridge-intdef
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/intdef.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/intdef.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/intdef.csv"))

(defn wooldridge-intqrt
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/intqrt.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/intqrt.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/intqrt.csv"))

(defn wooldridge-inven
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/inven.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/inven.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/inven.csv"))

(defn wooldridge-jtrain
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/jtrain.csv"))

(defn wooldridge-jtrain2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/jtrain2.csv"))

(defn wooldridge-jtrain3
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain3.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain3.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/jtrain3.csv"))

(defn wooldridge-jtrain98
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain98.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/jtrain98.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/jtrain98.csv"))

(defn wooldridge-k401k
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/k401k.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/k401k.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/k401k.csv"))

(defn wooldridge-k401ksubs
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/k401ksubs.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/k401ksubs.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/k401ksubs.csv"))

(defn wooldridge-kielmc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/kielmc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/kielmc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/kielmc.csv"))

(defn wooldridge-labsup
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/labsup.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/labsup.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/labsup.csv"))

(defn wooldridge-lawsch85
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/lawsch85.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/lawsch85.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/lawsch85.csv"))

(defn wooldridge-loanapp
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/loanapp.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/loanapp.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/loanapp.csv"))

(defn wooldridge-lowbrth
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/lowbrth.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/lowbrth.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/lowbrth.csv"))

(defn wooldridge-mathpnl
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mathpnl.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mathpnl.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/mathpnl.csv"))

(defn wooldridge-meap00_01
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap00_01.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap00_01.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/meap00_01.csv"))

(defn wooldridge-meap01
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap01.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap01.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/meap01.csv"))

(defn wooldridge-meap93
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap93.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meap93.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/meap93.csv"))

(defn wooldridge-meapsingle
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meapsingle.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/meapsingle.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/meapsingle.csv"))

(defn wooldridge-minwage
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/minwage.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/minwage.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/minwage.csv"))

(defn wooldridge-mlb1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mlb1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mlb1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/mlb1.csv"))

(defn wooldridge-mroz
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mroz.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/mroz.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/mroz.csv"))

(defn wooldridge-murder
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/murder.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/murder.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/murder.csv"))

(defn wooldridge-nbasal
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/nbasal.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/nbasal.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/nbasal.csv"))

(defn wooldridge-ncaa_rpi
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ncaa_rpi.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/ncaa_rpi.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/ncaa_rpi.csv"))

(defn wooldridge-nyse
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/nyse.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/nyse.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/nyse.csv"))

(defn wooldridge-okun
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/okun.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/okun.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/okun.csv"))

(defn wooldridge-openness
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/openness.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/openness.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/openness.csv"))

(defn wooldridge-pension
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/pension.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/pension.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/pension.csv"))

(defn wooldridge-phillips
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/phillips.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/phillips.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/phillips.csv"))

(defn wooldridge-pntsprd
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/pntsprd.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/pntsprd.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/pntsprd.csv"))

(defn wooldridge-prison
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/prison.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/prison.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/prison.csv"))

(defn wooldridge-prminwge
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/prminwge.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/prminwge.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/prminwge.csv"))

(defn wooldridge-rdchem
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rdchem.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rdchem.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/rdchem.csv"))

(defn wooldridge-rdtelec
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rdtelec.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rdtelec.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/rdtelec.csv"))

(defn wooldridge-recid
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/recid.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/recid.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/recid.csv"))

(defn wooldridge-rental
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rental.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/rental.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/rental.csv"))

(defn wooldridge-return
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/return.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/return.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/return.csv"))

(defn wooldridge-saving
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/saving.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/saving.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/saving.csv"))

(defn wooldridge-school93_98
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/school93_98.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/school93_98.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/school93_98.csv"))

(defn wooldridge-sleep75
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/sleep75.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/sleep75.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/sleep75.csv"))

(defn wooldridge-slp75_81
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/slp75_81.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/slp75_81.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/slp75_81.csv"))

(defn wooldridge-smoke
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/smoke.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/smoke.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/smoke.csv"))

(defn wooldridge-traffic1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/traffic1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/traffic1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/traffic1.csv"))

(defn wooldridge-traffic2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/traffic2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/traffic2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/traffic2.csv"))

(defn wooldridge-twoyear
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/twoyear.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/twoyear.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/twoyear.csv"))

(defn wooldridge-volat
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/volat.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/volat.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/volat.csv"))

(defn wooldridge-vote1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/vote1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/vote1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/vote1.csv"))

(defn wooldridge-vote2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/vote2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/vote2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/vote2.csv"))

(defn wooldridge-voucher
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/voucher.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/voucher.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/voucher.csv"))

(defn wooldridge-wage1
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wage1.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wage1.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/wage1.csv"))

(defn wooldridge-wage2
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wage2.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wage2.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/wage2.csv"))

(defn wooldridge-wagepan
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wagepan.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wagepan.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/wagepan.csv"))

(defn wooldridge-wageprc
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wageprc.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wageprc.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/wageprc.csv"))

(defn wooldridge-wine
  "Data description: https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wine.html"
  {:doc-link "https://vincentarelbundock.github.io/Rdatasets/doc/wooldridge/wine.html"}
  []
  (fetch-dataset "https://vincentarelbundock.github.io/Rdatasets/csv/wooldridge/wine.csv"))

(defn dataset-descriptions->doc-strings!
  "Run this function to attach the dataset descriptions as doc string to the fetch fns"
  []
  (run!
    (fn [v]
      (-> (symbol "scicloj.metamorph.ml.rdatasets" (name v))
       (find-var)
       (alter-meta!
         (fn [var-m]
           (if (:doc-link var-m)
             (do
               (println :fetch (:doc-link var-m))
               (assoc var-m :doc (doc-url->md (:doc-link var-m))))
             var-m)))))
    (keys (ns-publics (find-ns (symbol "scicloj.metamorph.ml.rdatasets"))))))


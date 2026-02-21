(ns scicloj.metamorph.ml.toydata.ggplot
  "Deprecated ns. Use scicloj.metamorph.ml.rdatasets instead"
  {:deprecated "1.1"}
  (:require [tech.v3.dataset]
            [clojure.java.io :as io]
            [scicloj.metamorph.ml.rdatasets :as rdatasets]
            ))

(def diamonds
  "ggplot2 diamonds dataset with prices and attributes of ~54,000 diamonds.

  Contains features like carat, cut, color, clarity, depth, table, and price."
  (rdatasets/ggplot2-diamonds))

(def ecomonics
  "ggplot2 economics dataset with US economic time series data.

  Contains monthly data on personal consumption expenditures, population,
  savings rate, unemployment, and median unemployment duration."
  (rdatasets/ggplot2-economics))

(def ecomonics_long
  "ggplot2 economics dataset in long format for time series visualization.

  Reshaped version of the economics dataset with columns: date, variable, value."
  (rdatasets/ggplot2-economics_long))

(def faithfuld
  "ggplot2 faithfuld dataset with 2D density estimates of Old Faithful eruptions.

  Contains eruption duration, waiting time, and density estimates. Useful for
  contour plots and density visualization examples."
  (rdatasets/ggplot2-faithfuld))

(def luv_colours
  "ggplot2 luv_colours dataset with colors in Luv color space.

  Contains RGB color names with their corresponding L, u, v coordinates in
  the perceptually uniform Luv color space."
  (rdatasets/ggplot2-luv_colours))

(def midwest
  "ggplot2 midwest dataset with demographic data for Midwest US counties.

  Contains population statistics, area, and demographic percentages for counties
  in IL, IN, MI, OH, and WI."
  (rdatasets/ggplot2-midwest))

(def mpg
  "ggplot2 mpg dataset with fuel economy data for 38 car models (1999-2008).

  Contains manufacturer, model, engine specifications, drivetrain, and city/highway
  fuel efficiency ratings."
  (rdatasets/ggplot2-mpg))

(def msleep
  "ggplot2 msleep dataset with sleep patterns of 83 mammal species.

  Contains total sleep time, REM sleep, sleep cycle, awake time, brain weight,
  body weight, and dietary information."
  (rdatasets/ggplot2-msleep))

(def presidential
  "ggplot2 presidential dataset with US presidential terms (1953-2021).

  Contains name, party affiliation, and start/end dates for each presidential term."
  (rdatasets/ggplot2-presidential))

(def seals
  "ggplot2 seals dataset with vector field of seal movements.

  Contains latitude, longitude, and movement vectors (delta_long, delta_lat) for
  visualizing seal migration patterns."
  (rdatasets/ggplot2-seals))

(def txhousing
  "ggplot2 txhousing dataset with Texas housing market data (2000-2015).

  Contains monthly housing statistics (sales, volume, median price, listings,
  inventory) for major Texas cities."
  (rdatasets/ggplot2-txhousing))

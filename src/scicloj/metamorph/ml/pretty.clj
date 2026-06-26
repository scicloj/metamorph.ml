(ns scicloj.metamorph.ml.pretty
  (:require
   [scicloj.metamorph.ml.r :as r]
   [tablecloth.column.api :as tcc]
   [wadogo.scale :as s]))

(defn- pretty--wadogo [s proposed-ticks]
  (let [min (tcc/reduce-min s)
        max (tcc/reduce-max s)
        scale (s/scale :linear {:domain [min max]})]
    (s/ticks scale proposed-ticks)))



(defn pretty
  "Compute pretty breaks using R function base::pretty.

   Parameters:
    
   - `s`         sequence of values
   - `opts`      options for pretty, 
        - `:n` number of breask is supported by all implementations
   
   - `impl`       An implementation keyword, either

       - `:ocpu`    Uses an online service https://www.opencpu.org/api.html (server: cloud.opencpu.org)
       - `:renjine` Uses https://renjin.org/   
       - `:clojisr` Uses https://github.com/scicloj/clojisr, which requires a local R installation 
       - `:wadogo`  Uses plotje/wadogo (which has quite different notion of 'pretty' compared to R)
   
    
   Each implementation requires dependencies to be added:
    
   - `:ocpu` :  [opencpu-clj/opencpu-clj \"0.3.1\"] 
   - `:renjin` : [org.renjin/renjin-script-engine \"3.5-beta76\"]
   - `:clojisr` : [scicloj/clojisr \"1.1.0\"]


   Returns seq of the breaks, which R considers 'pretty'
    
    "

  [s opts impl]

  (let [result
        (case impl
          :ocpu (r/pretty s opts impl)
          :renjin (r/pretty s opts impl)
          :clojisr (r/pretty s opts impl)
          :wadogo (pretty--wadogo s (get opts :n 5))
          )]
    result
    ))
 






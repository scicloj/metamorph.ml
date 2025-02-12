#!/bin/sh -e

clj -X:smoke-test:dev text-perf/tfidf :max-lines 100000
clj -X:smoke-test:dev 'text-perf/df->tidy' :num-rows 100000
clj -X:smoke-test:dev text-perf/tidy :max-lines 100000

#real    1m27.365s
#user    3m56.936s
#sys     0m3.729s
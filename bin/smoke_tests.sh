#!/bin/sh -e

clj -X:smoke-test text-perf/tfidf :max-lines 100000
clj -X:smoke-test 'text-perf/df->tidy' :num-rows 100000
clj -X:smoke-test text-perf/tidy :max-lines 100000

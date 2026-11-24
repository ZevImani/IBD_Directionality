#!/bin/bash
# usage: memwatch.sh <command...>  -> runs it, samples summed PSS of its process tree every 0.5 s
"$@" &
pid=$!
peak=0; peak_n=0
tree() { local p=$1; echo $p; for c in $(pgrep -P $p); do tree $c; done; }
while kill -0 $pid 2>/dev/null; do
    tot=0; n=0
    for p in $(tree $pid); do
        v=$(awk '/^Pss:/{print $2}' /proc/$p/smaps_rollup 2>/dev/null); [ -n "$v" ] && tot=$((tot+v)) && n=$((n+1))
    done
    [ $tot -gt $peak ] && peak=$tot && peak_n=$n
    sleep 0.5
done
wait $pid; rc=$?
echo "MEMWATCH peak_pss_mb=$((peak/1024)) processes_at_peak=$peak_n exit=$rc"

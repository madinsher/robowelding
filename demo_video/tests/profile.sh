#!/bin/bash
# Steady-state per-frame cost: render 3 consecutive frames, report each.  usage: tests/profile.sh "<extra args>" label
cd "$(dirname "$0")/.."
rm -rf out/prof_$2; mkdir -p out/prof_$2
xvfb-run -a -s "-screen 0 1920x1080x24" python3 build.py --render --start 300 --end 302 --outdir out/prof_$2 $1 2>&1 | grep -E "\[render\] frame|fx off" | sed "s/^/[$2] /"

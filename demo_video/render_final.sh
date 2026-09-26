#!/bin/bash
# Final render in chunks (resumable: existing frames are skipped).  usage: ./render_final.sh [start] [end] [chunk] [extra build.py args...]
cd "$(dirname "$0")"
START=${1:-1}; END=${2:-1008}; CHUNK=${3:-72}; shift 3 2>/dev/null
mkdir -p out/frames out/logs
f=$START
while [ $f -le $END ]; do
  t=$((f + CHUNK - 1)); [ $t -gt $END ] && t=$END
  echo "[chunk] $f-$t start $(date +%H:%M:%S)"
  xvfb-run -a -s "-screen 0 1920x1080x24" python3 build.py --render --start $f --end $t --outdir out/frames "$@" > out/logs/chunk_${f}_${t}.log 2>&1
  n=$(grep -c "\[render\] frame" out/logs/chunk_${f}_${t}.log)
  echo "[chunk] $f-$t done $(date +%H:%M:%S) rendered=$n $(grep -E 'Traceback|Error' out/logs/chunk_${f}_${t}.log | head -1)"
  f=$((t + 1))
done
echo "[final] all chunks done $(date +%H:%M:%S), frames: $(ls out/frames | wc -l)"

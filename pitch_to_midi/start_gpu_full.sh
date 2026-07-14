#!/usr/bin/env bash
set -e
cd /root/audio-classification/pitch_to_midi
rm -f gpu_full.log gpu_full.pid gpu_full.exit
nohup ./run_gpu_full.sh > gpu_full.log 2>&1 < /dev/null &
echo $! > gpu_full.pid
cat gpu_full.pid

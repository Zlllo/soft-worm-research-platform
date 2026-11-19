#!/bin/bash
set -eu
# Run inside the configured focal container with a private virtual display.
lab=$(cd "$(dirname "$0")" && pwd)
export THERMAL_TRIAL=$(realpath "$1")
for file in frames.csv network.npz done.json; do
    if test -e "$THERMAL_TRIAL/$file"; then
        echo "Existing trial data in $THERMAL_TRIAL; use a fresh trial directory." >&2
        exit 1
    fi
done
cd /work/BAAIWorm/build2
export PYTHONPATH="$lab:/work/BAAIWorm:/work/BAAIWorm/build2:/work/temperature"
export LD_LIBRARY_PATH=/tmp/nv-gl/usr/lib/x86_64-linux-gnu:/tmp/nv-gpucomp/usr/lib/x86_64-linux-gnu:/opt/boost/lib:.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
ulimit -c 0
./neuronXcore -data ../eworm/ghost_in_mesh_sim/data/tuned/video_offline/video_offline_neuronX >"$THERMAL_TRIAL/native.log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT
for tick in $(seq 1 100); do
    grep -q "cuda cannot share graphics resources" "$THERMAL_TRIAL/native.log" && break
    kill -0 "$pid" || exit 1
    sleep 1
done
window=$(xdotool search --name owl | tail -1)
xdotool windowfocus "$window"
xdotool key space
for tick in $(seq 1 650); do
    test -s "$THERMAL_TRIAL/done.json" && break
    kill -0 "$pid" || exit 1
    sleep 1
done
test -s "$THERMAL_TRIAL/done.json" && test -s "$THERMAL_TRIAL/network.npz" || exit 1
xdotool key q
wait "$pid" || echo "Native viewer exit status $?" >&2

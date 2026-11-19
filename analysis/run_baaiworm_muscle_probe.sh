#!/bin/bash
set -eu
# Run inside the project's configured baaiworm-focal container.
# /work/run.sh launches unmodified build2/neuronXcore on display :88.
assay=$(mktemp -d /work/muscle-assay.XXXXXX)
echo "Results: $assay"
export DISPLAY=:88
cd /work/temperature
cp record_head.py "$assay/passive_recorder.py"
trap 'cp "$assay/passive_recorder.py" /work/temperature/record_head.py' EXIT
printf 'from baaiworm_muscle_probe import install\n' > record_head.py
Xvfb :88 -screen 0 1024x768x24 >"$assay/x.log" 2>&1 &
for group in sham right left dorsal ventral sham2; do
    export WORM_GROUP="$group" WORM_OUTPUT="$assay/$group"
    mkdir -p "$WORM_OUTPUT"
    bash /work/run.sh > "$WORM_OUTPUT/run.log" 2>&1 &
    pid=$!
    for tick in $(seq 1 120); do
        grep -q 'cuda cannot share graphics resources' "$WORM_OUTPUT/run.log" && break
        kill -0 "$pid" || exit 1
        sleep 1
    done
    xdotool search --name owl | tail -1 | xargs -r xdotool windowfocus
    xdotool key space
    for tick in $(seq 1 300); do
        test -s "$WORM_OUTPUT/completed.json" && break
        kill -0 "$pid" || exit 1
        sleep 1
    done
    test -s "$WORM_OUTPUT/completed.json"
    xdotool key q
    wait "$pid" || true
    echo "completed $group"
done
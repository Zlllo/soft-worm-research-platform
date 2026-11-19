"""Instrument the BAAIWorm demo for a paired open-loop muscle perturbation."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

GROUPS = {
    "sham": (),
    "sham2": (),
    "right": (0, 1),
    "left": (2, 3),
    "dorsal": (0, 2),
    "ventral": (1, 3),
}


def install(worm_class, interact, output, mesh_dir):
    # This hook has the same signature as the passive head recorder in build b99ca5a.
    condition = os.environ["WORM_GROUP"]
    root = Path(os.environ["WORM_OUTPUT"])
    root.mkdir(parents=True, exist_ok=True)
    original = worm_class.control_muscle
    digest = hashlib.sha256()
    snapshot = root / "mesh.json"
    stream = (root / "trajectory.csv").open("w", newline="")
    writer = csv.writer(stream)
    writer.writerow(
        ["frame", "time_s", "head_x", "head_y", "head_z", "center_x", "center_y", "center_z"]
    )

    def control(worm):
        # step() has incremented step_count, but the next native body update has yet to run.
        frame = worm.step_count - 1
        if frame <= 150 and frame % 3 == 0:
            interact.save_worm(0, str(snapshot))
            state = json.loads(snapshot.read_text())
            pos = np.array(state["Position"]).reshape(-1, 3)
            assert len(pos) == state["nVertices"] == 984 and np.isfinite(pos).all()
            center = interact.worm_local_to_world(0, pos.mean(axis=0).astype(np.float32))
            head = interact.worm_local_to_world(0, interact.get_head_location(0))
            writer.writerow([frame, frame / 30.0, *map(float, head), *map(float, center)])
            stream.flush()
        if frame == 150:
            stream.close()
            (root / "completed.json").write_text(
                json.dumps(
                    dict(
                        original_muscle_sha256=digest.hexdigest(),
                        command_frames=150,
                        condition=condition,
                    )
                )
            )
        signals = original(worm)
        assert len(signals) == 1
        command = np.asarray(signals[0], dtype=np.float32).reshape(4, 24).copy()
        assert np.isfinite(command).all()
        if frame < 150:
            digest.update(command.tobytes())
        if 60 <= frame < 120:
            for group in GROUPS[condition]:
                command[group] *= 0.8
        return [command.reshape(-1)]

    worm_class.control_muscle = control

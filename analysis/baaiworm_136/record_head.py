"""Native video_offline recorder hook. Temperature changes only sensory current."""
import csv
import json
import os
from pathlib import Path

import numpy as np
from thermotaxis.baaiworm_temperature import BAAIWormTemperatureField


def install(worm_class, interact, output, mesh_dir):
    root = Path(os.environ["THERMAL_TRIAL"])
    spec = json.loads((root / "design.json").read_text())
    field = BAAIWormTemperatureField(spec["origin"], spec["length"], spec["normal"])
    original = worm_class.control_muscle
    stream = (root / "frames.csv").open("x", newline="")
    writer = csv.writer(stream)
    writer.writerow(["frame", "body_s", "neural_s", "head_x", "head_y", "head_z",
                     "center_x", "center_y", "center_z", "head_C", "center_C",
                     "sensed_C", "base_nA", "thermal_nA", "clamp_nA"])
    muscles, volts = [], []
    reference = None

    def control(worm):
        nonlocal reference
        frame = worm.step_count - 1
        if frame > spec["last_frame"]:
            return original(worm)
        assert worm.worm_num == 1 and worm.worm_0.neuron_num == 136
        assert len(worm.worm_0.worm.iclamp) == 15 and worm.config["detail_control"] == "fc"
        assert worm.config["interaction_mode"] == "offline"
        head = interact.worm_local_to_world(0, interact.get_head_location(0))
        mesh = root / "mesh.json"
        interact.save_worm(0, str(mesh))
        vertices = np.array(json.loads(mesh.read_text())["Position"]).reshape(984, 3)
        center = interact.worm_local_to_world(0, vertices.mean(axis=0).astype(np.float32))
        head_c, center_c = field.temperature_c(np.stack([head, center]))
        sensed = head_c if spec["feedback"] else field.temperature_c(field.origin_world)
        value = -abs(spec["target_C"] - sensed)
        if reference is None or frame <= spec["onset_frame"]:
            reference = value
        thermal = spec["gain_nA"] * np.tanh((value-reference) / spec["scale_C"])
        reference += -np.expm1(-1/(30*spec["adaptation_s"])) * (value-reference)
        substeps = int(worm.config["motion_sim_dt"] / worm.config["neuron_sim_dt"])
        index = worm.step_count * substeps
        base = worm.standard_input_signal[0, index]
        worm.standard_input_signal[0, index] = base + thermal
        try:
            signals = original(worm)
            actual = np.array([clamp.amp for clamp in worm.worm_0.worm.iclamp])
            np.testing.assert_allclose(actual, base + thermal, atol=1e-8, rtol=1e-6)
        finally:
            worm.standard_input_signal[0, index] = base
        muscles.append(np.array(signals[0], dtype=np.float32).reshape(96))
        volts.append(np.asarray(worm.mnv_0, dtype=np.float32).copy())
        writer.writerow([frame, frame/30, worm.step_count * substeps * worm.config["neuron_sim_dt"]/1000,
                         *head.tolist(), *center.tolist(), head_c, center_c, sensed, base, thermal, actual[0]])
        stream.flush()
        if frame == spec["last_frame"]:
            stream.close()
            np.savez_compressed(root / "network.npz", muscle=muscles, motor_mV=volts)
            (root / "done.json").write_text(json.dumps(dict(cells=136, input_cells=worm.config["input_neuron"],
                    motor_cells=worm.config["output_neuron"], samples=frame+1, substeps=substeps)))
        return signals
    worm_class.control_muscle = control
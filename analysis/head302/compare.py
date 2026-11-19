"""Train on the single source and run paired closed-loop tests on four repository maps."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "程序")]
from core.utils import create_temperature_environment
from thermotaxis.head302 import HeadSteering, masks, neuron_graph

OUT = ROOT / "docs/reports/assets/head302-thermotaxis"
FIELDS = ("single_center", "linear_gradient", "dual_center", "ring_hotspot")
TRAIN_SEEDS = (11, 22, 33, 44, 55, 66)
GRAPH_SEEDS = (101, 202, 303, 404)
LIMIT, SPEED, STEPS = .65, .65, 96
CASES = 64
UPDATES = 128


def sample(field, x, y):
    """Bilinear head readings; a probe extending past the arena samples its edge."""
    y = np.clip(y, 0, field.shape[0] - 1)
    x = np.clip(x, 0, field.shape[1] - 1)
    i, j = x.astype(int), y.astype(int)
    u, v = x - i, y - j
    ii, jj = np.minimum(i + 1, field.shape[1] - 1), np.minimum(j + 1, field.shape[0] - 1)
    return ((1-v) * ((1-u)*field[j, i] + u*field[j, ii])
            + v * ((1-u)*field[jj, i] + u*field[jj, ii]))


def scan(field, x, y, heading, noise):
    # Same centre, left and right local samples in every architecture.
    angles = heading[:, None] + np.array([np.pi/4, -np.pi/4])
    sides = sample(field, x[:, None] + np.cos(angles), y[:, None] + np.sin(angles))
    center = sample(field, x, y) + noise[:, 0]
    return torch.from_numpy(((sides + noise[:, 1:] - center[:, None]) / 2).astype("float32"))


def demonstration(field, source, seed, count):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 39, size=(2, count))
    heading = rng.uniform(-np.pi, np.pi, count)
    obs = scan(field, x, y, heading, rng.normal(0, .08, (count, 3)))
    angle = np.arctan2(source[1] - y, source[0] - x) - heading
    target = np.clip(np.arctan2(np.sin(angle), np.cos(angle)) / LIMIT, -1, 1)
    return obs, torch.tensor(target, dtype=torch.float32)


def train(model, samples, validation, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    order = np.random.default_rng(seed + 1000)
    obs, target = samples
    curve = []
    for step in range(UPDATES + 1):
        if step in (0, 8, 32, UPDATES):
            with torch.no_grad():
                mae = (model(validation[0]) - validation[1]).abs().mean().item()
            curve.append(dict(update=step, action_mae=mae))
        if step < UPDATES:
            idx = order.integers(len(obs), size=64)
            loss = (model(obs[idx]) - target[idx]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
    return curve


@torch.no_grad()
def rollout(field, model):
    rng = np.random.default_rng(81002)
    x, y = rng.uniform(1, 38, (2, CASES))
    heading = rng.uniform(-np.pi, np.pi, CASES)
    matrix = model.recurrent_matrix() if model is not None else None
    scores, trace = [], [(x[0], y[0])]
    for _ in range(STEPS):
        obs = scan(field, x, y, heading, rng.normal(0, .08, (CASES, 3)))
        action = 0. if model is None else model(obs, matrix).numpy()
        heading = heading + LIMIT * action
        x = x + SPEED * np.cos(heading)
        y = y + SPEED * np.sin(heading)
        # Specular walls preserve distance and prevent a stopped head scoring at a wall.
        heading = np.where((x < 0) | (x > 39), np.pi - heading, heading)
        heading = np.where((y < 0) | (y > 39), -heading, heading)
        x, y = 39 - np.abs(39 - np.abs(x)), 39 - np.abs(39 - np.abs(y))
        scores.append((sample(field, x, y) - field.min()) / np.ptp(field))
        trace.append((x[0], y[0]))
    scores = np.stack(scores)
    return scores.mean(0), (scores >= .9).mean(0), np.asarray(trace)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out):
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    import scipy
    from scipy import ndimage  # require the repository ring generator's Gaussian path

    np.random.seed(41001)
    maps = {key: create_temperature_environment(40, 40, 41001, key) for key in FIELDS}
    graph = neuron_graph()
    designs = [("real", 0), ("dense", 0)] + [("rewired", g) for g in GRAPH_SEEDS]
    primary = maps["single_center"]
    validation = demonstration(*primary, seed=71001, count=1024)
    cases, curves, designs_info, traces = [], [], [], {}

    def evaluate(model, design):
        for field_id, (field, _) in maps.items():
            scores, residences, trace = rollout(field, model)
            for i, (score, residence) in enumerate(zip(scores, residences)):
                cases.append(dict(**design, field=field_id, case=i,
                                  mean_heat=round(score, 7), hot_residence=round(residence, 7)))
            if design["fit_seed"] in (0, TRAIN_SEEDS[0]) and design["graph_seed"] in (0, GRAPH_SEEDS[0]):
                traces[field_id + "_" + design["model"]] = trace

    evaluate(None, dict(model="straight", graph_seed=0, fit_seed=0))
    for topology, g in designs:
        chemical, gap = masks(graph, topology, g)
        for seed in TRAIN_SEEDS:
            torch.manual_seed(seed)
            model = HeadSteering(graph.names, chemical, gap)
            design = dict(model=topology, graph_seed=g, fit_seed=seed)
            for point in train(model, demonstration(*primary, seed, 4096), validation, seed):
                curves.append(dict(**design, **point))
            evaluate(model, design)
            print(topology, g, seed, flush=True)
        designs_info.append(dict(model=topology, graph_seed=g, states=model.size,
                                 parameters=sum(p.numel() for p in model.parameters()),
                                 chemical_edges=int(chemical.sum()), gap_pairs=int(np.triu(gap, 1).sum()),
                                 chemical_changed=int(np.sum(chemical != (graph.chemical > 0))),
                                 gap_changed=int(np.sum(gap != (graph.gap > 0)))))
    write_csv(out / "episodes.csv", cases)
    write_csv(out / "learning.csv", curves)
    np.savez_compressed(out / "example.npz", **traces, **{k: v[0] for k, v in maps.items()})
    manifest = dict(designs=designs_info, graph_sha256=graph.provenance["derived_data_sha256"],
                    torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__,
                    field_sha256={k: hashlib.sha256(v[0].astype("<f8").tobytes()).hexdigest()
                                  for k, v in maps.items()})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    run(parser.parse_args().out)

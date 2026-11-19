"""Compare bilateral pathway interventions using frozen, archived navigation actors.

Run from the repository root with PYTHONPATH=src. No fitting is performed.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from navigation_feasibility_replication import evaluate
from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.navigation import NavigationBatch, NavigationConfig
from thermotaxis.navigation_window_policy import WindowActorCritic

ARCHIVE = Path("docs/reports/assets/pathway-stage1-2026-11-03")
OUTPUT = Path("docs/reports/assets/pathway-specificity")
TARGET = "AFD-AIY"
MATCH = "AIZ-RIA"


def new_cases(config, first_seed=1911001):
    # Choose only by field and sign of initial temperature error.
    strata = {(f, s): [] for f in range(2) for s in (-1, 1)}
    for seed in range(first_seed, first_seed + 1000):
        field = (seed - first_seed) % 2
        env = NavigationBatch(config, [seed], fields=[field])
        bucket = strata[field, int(np.sign(env.offset[0]))]
        if len(bucket) < 16:
            bucket.append(seed)
        if all(len(b) == 16 for b in strata.values()):
            break
    assert all(len(b) == 16 for b in strata.values())
    return [strata[f, s][i] for i in range(16) for s in (-1, 1) for f in range(2)]


def bilateral_pairs(graph):
    result = {}
    for a in CORE[::2]:
        for b in CORE[::2]:
            if a == b:
                continue
            edges = [(CORE.index(a) + side, CORE.index(b) + side) for side in range(2)]
            if all(graph.chemical[i, j] > 0 for i, j in edges):
                result[a[:-1] + "-" + b[:-1]] = edges
    return result


def run(model_id, matched=False):
    torch.set_num_threads(1)
    graph = load_connectome().select(CORE)
    pairs = bilateral_pairs(graph)
    manifest = json.loads((ARCHIVE / "manifest.json").read_text())
    archive = ARCHIVE / "reproduction.zip"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    with zipfile.ZipFile(archive) as z:
        plan = json.loads(z.read("evidence/design.json"))
        spec = plan["specs"][model_id]
        assert spec["id"] == model_id
        member = f"evidence/fit-{model_id:03d}-All/model.pt"
        buf = z.read(member)
        assert hashlib.sha256(buf).hexdigest() == manifest["files"][member]
        state = torch.load(io.BytesIO(buf), map_location="cpu", weights_only=True)
    model = WindowActorCritic("real8", spec["interface"]).eval()
    config = NavigationConfig(**plan["config"])
    old = plan["fresh"]["seeds"]
    seeds = new_cases(config, 2911001) if matched else old + new_cases(config)
    assert len(set(seeds)) == (64 if matched else 96)
    treatments = [("intact", [], 1.0)]
    for name, edges in pairs.items():
        treatments.append((name, edges, 0.0))
    for name in (TARGET, MATCH):
        treatments.append((name + "-half", pairs[name], 0.5))
    if matched:
        # Match the absolute change on each homologous side, retaining each
        # edge sign. The smaller-magnitude edge on each side reaches zero.
        model.load_state_dict(state)
        magnitudes = []
        for name in (TARGET, MATCH):
            magnitudes.append(
                [
                    abs(model.chemical[(model.pre == i) & (model.post == j)].item())
                    for i, j in pairs[name]
                ]
            )
        common = np.min(magnitudes, axis=0)
        treatments = [("intact", [], 1.0)]
        for name, weights in zip((TARGET, MATCH), magnitudes):
            assert min(weights) > 0
            treatments.append((name + "-matched", pairs[name], 1 - common / weights))
    rows = []
    for label, edges, multiplier in treatments:
        model.load_state_dict(state)
        weights = []
        for edge_index, (i, j) in enumerate(edges):
            mask = (model.pre == i) & (model.post == j)
            assert mask.sum().item() == 1
            weights.append(float(model.chemical[mask].item()))
            with torch.no_grad():
                model.chemical[mask] *= float(
                    np.broadcast_to(multiplier, (len(edges),))[edge_index]
                )
        cases = evaluate(model, config, seeds)
        assert len(cases) == len(seeds)
        for i, case in enumerate(cases):
            assert case["seed"] == seeds[i]
            rows.append(
                dict(
                    interface=spec["interface"],
                    training_seed=spec["seed"],
                    treatment=label,
                    case_set="holdout" if matched else ("archive" if i < len(old) else "new"),
                    case_seed=seeds[i],
                    field=case["field"],
                    initial_error_K=case["initial_error_K"],
                    removed_weight_l2=float(np.linalg.norm(np.asarray(weights) * (1 - multiplier))),
                    complete=int(case["complete_success"]),
                    pre=int(case["pre_stable_s"] is not None),
                    post=int(case["post_stable_s"] is not None),
                    tracking=case["full_tracking"],
                    mae_K=case["center_error_mae_K"],
                )
            )
        print(spec["interface"], spec["seed"], label, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prefix = "matched" if matched else "cases"
    path = OUTPUT / f"{prefix}-{model_id}.csv"
    with path.with_suffix(".tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".tmp").replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", type=int, nargs="*", default=list(range(6)))
    parser.add_argument("--matched", action="store_true")
    args = parser.parse_args()
    for model_id in args.ids:
        run(model_id, args.matched)

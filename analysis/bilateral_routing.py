"""Equal-budget behavioral cloning on the two degree- and symmetry-matched masks."""

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile
from dataclasses import replace
from math import pi

import numpy as np
import torch

from navigation_window_ppo import fit_bc
from navigation_feasibility_replication import evaluate
from pathway_specificity import bilateral_pairs, new_cases, OUTPUT
from thermotaxis.connectome import CORE, load_connectome
from thermotaxis.navigation import NavigationConfig
from thermotaxis.navigation_window_policy import WindowActorCritic

SEEDS = (7, 17, 27, 37, 47, 57, 67, 77)
TEACHER = Path("docs/reports/assets/navigation-study-2026-10-25")
BUDGET = 2048


def alternative_mask():
    graph = load_connectome().select(CORE)
    original = graph.chemical > 0
    fixed = original.copy()
    pairs = bilateral_pairs(graph)
    for edges in pairs.values():
        for i, j in edges:
            fixed[i, j] = False
    options = [
        (i, j)
        for i in range(0, 8, 2)
        for j in range(0, 8, 2)
        if i != j and not (fixed[i, j] or fixed[i + 1, j + 1])
    ]
    alternatives = []
    for chosen in itertools.combinations(options, len(pairs)):
        mask = fixed.copy()
        for i, j in chosen:
            mask[i, j] = mask[i + 1, j + 1] = True
        if np.array_equal(mask.sum(0), original.sum(0)) and np.array_equal(
            mask.sum(1), original.sum(1)
        ):
            if not np.array_equal(original, mask):
                alternatives.append(mask)
    assert len(alternatives) == 1
    return alternatives[0]


def run(shard, shards):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = replace(NavigationConfig(), field_turn_rad=pi / 12)
    seeds = new_cases(config, 3911001)
    archive = TEACHER / "teacher-data.zip"
    manifest = json.loads((TEACHER / "manifest.json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest[archive.name]
    with zipfile.ZipFile(archive) as z:
        member = "results/navigation-feature-teacher-raw-2026-10-25.npz"
        with np.load(io.BytesIO(z.read(member)), allow_pickle=False) as data:
            train = {name: data[name] for name in ("x", "y")}
    alternative = alternative_mask()
    for seed in SEEDS[shard::shards]:
        for interface in ("restricted", "full"):
            for layout in ("original", "swap"):
                path = OUTPUT / f"fit-{interface}-{seed}-{layout}.json"
                torch.manual_seed(seed)
                actor = WindowActorCritic("real8", interface)
                if layout == "swap":
                    # Keep the same sampled weight on its source-side edge slot.
                    for k, (i, j) in enumerate(zip(actor.pre.tolist(), actor.post.tolist())):
                        if not alternative[i, j]:
                            (new_target,) = np.flatnonzero(
                                alternative[i]
                                & ~np.isin(np.arange(8), actor.post[actor.pre == i].tolist())
                            )
                            actor.post[k] = int(new_target)
                    reconstructed = np.zeros((8, 8), dtype=bool)
                    reconstructed[actor.pre.numpy(), actor.post.numpy()] = True
                    assert np.array_equal(reconstructed, alternative)
                curve = fit_bc(actor, train, seed, updates=BUDGET)
                cases = evaluate(actor.eval(), config, seeds)
                rows = [
                    {
                        k: r[k]
                        for k in (
                            "seed",
                            "field",
                            "initial_error_K",
                            "complete_success",
                            "full_tracking",
                            "center_error_mae_K",
                        )
                    }
                    for r in cases
                ]
                report = dict(
                    seed=seed,
                    interface=interface,
                    layout=layout,
                    updates=BUDGET,
                    batch_size=64,
                    curve=curve,
                    torch=torch.__version__,
                    numpy=np.__version__,
                    chemical_edges=list(zip(actor.pre.tolist(), actor.post.tolist())),
                    cases=rows,
                )
                path.with_suffix(".tmp").write_text(json.dumps(report))
                path.with_suffix(".tmp").replace(path)
                print("finished", seed, interface, layout, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(args.shard, args.shards)

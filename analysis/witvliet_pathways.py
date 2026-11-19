"""Independent chemical synapse counts from the authors' eight reconstructions."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from thermotaxis.connectome import CORE, DATA_ROOT

REVISION = "0646af9d25896ae660f97d462eab2d67282f5625"
ROOT = f"https://raw.githubusercontent.com/dwitvliet/nature2021/{REVISION}/data/synapses"
OUTPUT = Path("docs/reports/assets/pathway-specificity")


def main():
    cache = Path("results/witvliet")
    cache.mkdir(exist_ok=True, parents=True)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    neurons = set(json.loads((DATA_ROOT / "herm_neurons.json").read_text())["names"])
    rows, sources, sensory, routes = [], {}, [], []
    for dataset in range(1, 9):
        name = f"Dataset{dataset}_synapses.json"
        path = cache / name
        if not path.exists():
            with urlopen(f"{ROOT}/{name}", timeout=60) as response:
                raw = response.read()
            json.loads(raw)
            path.write_bytes(raw)
        raw = path.read_bytes()
        sources[name] = hashlib.sha256(raw).hexdigest()
        records = json.loads(raw)
        counts = Counter((r["pre"], post) for r in records for post in r["post"])
        seen = {n for edge in counts for n in edge}
        assert set(CORE) <= seen
        # Enumerate all two-hop chemical routes through named neurons, including
        # cross-side edges. Normalize each step by the sender's neuronal output.
        for minimum in (1, 2, 3):
            eligible = {
                (a, b): n
                for (a, b), n in counts.items()
                if a in neurons and b in neurons and n >= minimum
            }
            degree = Counter()
            for (a, b), n in eligible.items():
                degree[a] += n
            by_mid = Counter()
            npaths = Counter()
            for start in ("AFDL", "AFDR"):
                for end in ("RIAL", "RIAR"):
                    for mid in neurons - {"AFDL", "AFDR", "RIAL", "RIAR"}:
                        first, second = eligible.get((start, mid), 0), eligible.get((mid, end), 0)
                        if first and second:
                            npaths[mid] += 1
                            by_mid[mid] += 0.5 * first / degree[start] * second / degree[mid]
            total = sum(by_mid.values())
            routes.append(
                dict(
                    dataset=dataset,
                    min_synapses=minimum,
                    two_hop_paths=sum(npaths.values()),
                    via_aiy_paths=npaths["AIYL"] + npaths["AIYR"],
                    intermediate_neurons="|".join(sorted(by_mid)),
                    two_hop_mass=total,
                    aiy_fraction=(by_mid["AIYL"] + by_mid["AIYR"]) / total if total else None,
                )
            )
        # Twelve left/right amphid sensory classes. Compare fixed AIY targets
        # and the largest share going to any one named neuron on each side.
        for cls in "ADF ADL AFD ASE ASG ASH ASI ASJ ASK AWA AWB AWC".split():
            for side in "LR":
                pre = cls + side
                outgoing = {b: n for (a, b), n in counts.items() if a == pre and b in neurons}
                assert pre in seen
                sites = [r["post"] for r in records if r["pre"] == pre and r["post"]]
                fractional_aiy = sum(posts.count("AIY" + side) / len(posts) for posts in sites)
                maximum = max(outgoing.values(), default=0)
                sensory.append(
                    dict(
                        dataset=dataset,
                        source=pre,
                        total=sum(outgoing.values()),
                        ipsi_aiy=outgoing.get("AIY" + side, 0),
                        top_count=maximum,
                        raw_presynaptic_sites=len(sites),
                        fractional_aiy_sites=fractional_aiy,
                        top_targets="|".join(
                            sorted(b for b, n in outgoing.items() if n == maximum)
                        ),
                    )
                )
        for pre in CORE:
            # Restrict denominator to named neuronal targets in the project list.
            outputs = sum(v for (a, b), v in counts.items() if a == pre and b in neurons)
            for post in CORE:
                rows.append(
                    dict(
                        dataset=dataset,
                        pre=pre,
                        post=post,
                        synapses=counts[pre, post],
                        named_neuronal_outputs=outputs,
                    )
                )
    with (OUTPUT / "witvliet-core.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "amphid-outputs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sensory[0]))
        writer.writeheader()
        writer.writerows(sensory)
    with (OUTPUT / "two-hop-routes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(routes[0]))
        writer.writeheader()
        writer.writerows(routes)
    (OUTPUT / "witvliet-source.json").write_text(
        json.dumps(dict(source=ROOT, sha256=sources), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

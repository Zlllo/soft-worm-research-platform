"""Read the exact circuit distributed with the BAAIWorm video_offline demo."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import sys

DIGEST = "792a912ca1b466443d6fded8bd47cd4796a29aef044bb9c3c1c953965f1bd651"
CORE = ("AFDL", "AFDR", "AIYL", "AIYR", "AIZL", "AIZR", "RIAL", "RIAR")


def main(root):
    directory = root / "eworm/ghost_in_mesh_sim/data/tuned/video_offline"
    raw = (directory / "video_offline_abscircuit.pkl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST:
        raise ValueError("Expected the b99ca5a video_offline circuit")
    sys.path.insert(0, str(root / "eworm/network"))
    circuit = pickle.loads(raw)
    cells = {cell.name for cell in circuit.cells}
    edges = Counter(
        (x.pre_cell.name, x.post_cell.name, x.category)
        for x in circuit.connections
        if x.pre_cell and x.post_cell and x.pre_cell.name in CORE and x.post_cell.name in CORE
    )
    config = json.loads((directory / "video_offline_config.json").read_text())
    print(
        json.dumps(
            dict(
                upstream="b99ca5a",
                sha256=DIGEST,
                cells=len(cells),
                connection_records=len(circuit.connections),
                core_membership={n: n in cells for n in CORE},
                search_input=config["search_config"]["input_cell_names"],
                core_edges=[
                    dict(source=i, target=j, category=kind, count=n)
                    for (i, j, kind), n in sorted(edges.items())
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("upstream_root", type=Path)
    main(parser.parse_args().upstream_root)

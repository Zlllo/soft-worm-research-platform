"""Predeclared non-learning constant-heading negative control."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from thermotaxis.navigation import NavigationBatch,NavigationConfig
from navigation_preflight import summary


def main():
    plan=json.loads(Path('results/navigation-factorial-2026-10-25/design.json').read_text())
    for s in plan['specs']:assert (Path('results/navigation-factorial-2026-10-25')/f'fit-{s["id"]:03d}'/'record.json').exists()
    env=NavigationBatch(NavigationConfig(),plan['heldout_seeds']);env.initial_packet()
    while True:
        _,_,done=env.step(np.zeros((len(plan['heldout_seeds']),2)))
        if done:break
    rows=env.metrics()
    Path('results/navigation-blind-control-2026-10-25.json').write_text(json.dumps(dict(status='non_learning_negative_control',config=asdict(NavigationConfig()),cases=rows,summary=summary(rows)),indent=2)+'\n')


if __name__=='__main__':main()

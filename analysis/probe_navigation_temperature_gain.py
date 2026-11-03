"""Paired frozen fluctuation-gain assay after finding restricted low gain.

Amplifies observed fluctuations and their noise together; it does not add
information, retrain parameters, alter phase, or amplify the mean temperature.
"""
from dataclasses import replace
import json
import math
from pathlib import Path

import torch

from diagnose_navigation_failures import ROOT, check, digest, load_model, write
from navigation_feasibility_replication import evaluate
from navigation_preflight import summary
from thermotaxis.navigation import NavigationConfig


class FluctuationGain:
    def __init__(self, model, gain):
        self.model, self.gain = model, gain

    def actor_mean(self, x):
        scaled = x.clone()
        scaled[:, :, 0] *= self.gain
        return self.model.actor_mean(scaled)


def main():
    torch.set_num_threads(1)
    plan = json.loads((ROOT / 'design.json').read_text()); check(plan)
    dest = ROOT / 'temperature-gain.json'
    if dest.exists():
        raise RuntimeError('preserve frozen-gain result')
    design = dict(status='supplemental_after_low_gain_diagnosis', source_sha256=digest(Path(__file__)),
                  source_checkpoint='model-RL262144.pt', gains=[1., 3., 10.],
                  name='real8', interface='restricted', seeds=[7, 17, 27],
                  evaluation_seeds=list(range(71001, 71033)), field_turn_deg=15,
                  caveat='Noise amplified with signal. Previously inspected cases; not a new confirmation set.')
    write(ROOT / 'temperature-gain-design.json', design)
    config = replace(NavigationConfig(), field_turn_rad=math.pi/12)
    rows = []
    for s in plan['specs']:
        if s['name'] != design['name'] or s['interface'] != design['interface']:
            continue
        model = load_model(s, design['source_checkpoint'])
        before = {k:v.clone() for k,v in model.state_dict().items()}
        for gain in design['gains']:
            cases = evaluate(FluctuationGain(model, gain), config, design['evaluation_seeds'])
            rows.append(dict(spec=s, gain=gain, cases=cases, summary=summary(cases)))
            print('Frozen fluctuation gain', s['seed'], gain, summary(cases)['complete_success'], flush=True)
        assert all(torch.equal(v, model.state_dict()[k]) for k,v in before.items())
    write(dest, dict(design=design, runs=rows, complete=True))
    check(plan)


if __name__ == '__main__':
    main()

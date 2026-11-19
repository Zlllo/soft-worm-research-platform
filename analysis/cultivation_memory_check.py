"""Can the archived raw-window actors express different cultivation histories?"""

import hashlib
import io
import json
from math import log
import zipfile

import numpy as np
import torch

from pathway_specificity import ARCHIVE, OUTPUT
from thermotaxis.navigation_window_policy import NavigationFeatures, WindowActorCritic


def encoding_probe():
    # Short distinct prefixes are an encoding test, not a simulation of cultivation.
    feat = NavigationFeatures(2, "raw")
    for k in range(120):
        t = k / 10
        temperature = np.array([288.15, 298.15]) if k < 60 else np.full(2, 293.15 + 0.01 * (t - 6))
        packet = np.stack(
            [temperature, np.full(2, np.sin(t)), np.full(2, np.cos(t)), np.ones(2), np.ones(2)],
            axis=1,
        )[None].astype("float32")
        states = feat.observe(packet)
        if k == 59:
            assert not np.array_equal(states[0], states[1])
        if k == 100:
            np.testing.assert_array_equal(states[0], states[1])
    return states


def main():
    states = torch.from_numpy(encoding_probe())
    gaps = []
    with zipfile.ZipFile(ARCHIVE / "reproduction.zip") as z, torch.no_grad():
        for s in json.loads(z.read("evidence/design.json"))["specs"]:
            actor = WindowActorCritic("real8", s["interface"])
            member = f"evidence/fit-{s['id']:03d}-All/model.pt"
            raw = z.read(member)
            digest = json.loads((ARCHIVE / "manifest.json").read_text())["files"][member]
            assert hashlib.sha256(raw).hexdigest() == digest
            actor.load_state_dict(
                torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
            )
            a, b = actor.actor_mean(states)
            gaps.append(abs((a - b).item()))
    assert max(gaps) < 1e-6
    # Hawk et al., Fig. 2H and Results text; binomial counts for the two
    # opposite experiences at the same holding temperature. Natural logarithms.
    cohorts = [(27, 75), (44, 65)]
    pooled = sum(k for k, n in cohorts) / sum(n for k, n in cohorts)

    def loss(k, n, p):
        return -(k * log(p) + (n - k) * log(1 - p))

    common_loss = sum(loss(k, n, pooled) for k, n in cohorts)
    separate_loss = sum(loss(k, n, k / n) for k, n in cohorts)
    result = dict(
        source="https://pmc.ncbi.nlm.nih.gov/articles/PMC5806692/#F2",
        conditions=["holding_above_cultivation", "holding_below_cultivation"],
        counts=cohorts,
        raw_actor_absolute_logit_differences=gaps,
        pooled_probability=pooled,
        risk_difference=44 / 65 - 27 / 75,
        fitted_log_odds_difference=log(44 / 21) - log(27 / 48),
        in_sample_log_likelihood_gain_nats=common_loss - separate_loss,
    )
    (OUTPUT / "cultivation-memory.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

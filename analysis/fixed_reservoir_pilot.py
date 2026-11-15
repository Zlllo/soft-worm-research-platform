"""Reproduce the preregistered fixed-network engineering pilot from repository root."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from thermotaxis.connectome import load_connectome, rewire_directed

ROOT = Path('docs/reports/assets/fixed-reservoir-pilot-2026-11-15')
DT = .1

def graph_and_weights(graph, seed, nodes):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1., 1.], size=nodes)
    return (graph * signs[:, None] / max(1, graph.sum(axis=0).max())).astype(np.float64)

def simulate(u, weights, gap, inputs, step):
    n, samples = u.shape
    x = np.zeros((n, len(weights)))
    out = np.empty((n, samples, len(weights)))
    substeps = round(DT / step)
    assert abs(substeps * step - DT) < 1e-12
    for t in range(samples):
        for _ in range(substeps):
            drive = x @ weights + x @ gap - x * gap.sum(axis=0) + u[:, t, None] * inputs
            x += (step / .3) * (-x + np.tanh(drive))
        out[:, t] = x
    return out

def ridge(train, y, val, vy):
    m = train.mean(0); ym = y.mean()
    x = train - m; z = val - m
    # SVD reuses one decomposition for all regularization choices.
    U, s, Vt = np.linalg.svd(x, full_matrices=False)
    proj = U.T @ (y - ym)
    choices = []
    for penalty in (1e-4, 1e-2, 1., 100.):
        coef = Vt.T @ ((s / (s*s + penalty)) * proj)
        choices.append((float(np.mean((z @ coef + ym - vy)**2)), penalty, coef))
    _, penalty, coef = min(choices, key=lambda a: a[0])
    return m, ym, coef, penalty

def evaluate(features, y):
    X = features[:, 10:].reshape(-1, features.shape[-1])
    Y = y[:, 10:].reshape(-1)
    split1, split2 = 64*90, 80*90
    start = time.perf_counter()
    m, ym, coef, penalty = ridge(X[:split1], Y[:split1], X[split1:split2], Y[split1:split2])
    elapsed = time.perf_counter() - start
    pred = (X[split2:]-m) @ coef + ym
    truth = Y[split2:]
    mse = float(np.mean((pred-truth)**2))
    zero = float(np.mean(truth**2))
    return dict(mse=mse, zero_mse=zero, r2_vs_zero=1-mse/zero, penalty=penalty, fit_seconds=elapsed)

def main():
    protocol = json.loads((ROOT/'protocol.json').read_text())
    graph = load_connectome().select(json.loads(Path('data/connectome/herm_neurons.json').read_text())['names'])
    names = list(graph.names)
    chem = graph.chemical > 0
    gap = (graph.gap > 0).astype(float); np.fill_diagonal(gap, 0)
    gap *= .05 / max(1, gap.sum(0).max())
    inputs = np.zeros(len(names)); inputs[[names.index('AFDL'), names.index('AFDR')]] = 1.
    read = [names.index('RIAL'), names.index('RIAR')]
    rng = np.random.default_rng(4815)
    # New value every 0.5 seconds, each history independent.
    u = np.repeat(rng.normal(0, .35, (96, 20)), 5, axis=1)
    y = u - np.roll(u, 5, axis=1); y[:, :5] = 0
    results = []
    with threadpool_limits(limits=1):
        baseline = evaluate(u[:,:,None], y)
        graphs = [('real', chem, {})]
        for seed in range(101, 106):
            rewired, info = rewire_directed(chem, seed)
            assert np.array_equal(chem.sum(0), rewired.sum(0)) and np.array_equal(chem.sum(1), rewired.sum(1))
            graphs.append((f'rewired-{seed}', rewired, info))
        for name, matrix, info in graphs:
            for seed in (7, 17, 27):
                W = graph_and_weights(matrix, seed, len(names))
                start = time.perf_counter(); states = simulate(u,W,gap,inputs,.1); seconds = time.perf_counter()-start
                assert np.isfinite(states).all()
                fine = simulate(u,W,gap,inputs,.05)
                difference = float(np.sqrt(np.mean((states-fine)**2)))
                magnitude = float(np.sqrt(np.mean(fine**2)))
                result = dict(graph=name, seed=seed, rewiring=info, simulation_seconds=seconds,
                              state_max=float(np.abs(states).max()), dt_rms_difference=difference,
                              fine_state_rms=magnitude,
                              ria=evaluate(states[:,:,read], y), all_nodes=evaluate(states,y),
                              ria_fine=evaluate(fine[:,:,read],y), all_nodes_fine=evaluate(fine,y))
                results.append(result)
                print(name, seed, 'RIA', round(result['ria']['r2_vs_zero'],4), 'all', round(result['all_nodes']['r2_vs_zero'],4), flush=True)
    output = dict(protocol=protocol, connectome_sha256=graph.provenance['derived_data_sha256'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  current_input_baseline=baseline, runs=results,
                  timing_note='Wall-clock seconds on this instance, single BLAS thread, 96 histories of 10 seconds simulated in a batch; no warm-up exclusion.')
    (ROOT/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')

if __name__=='__main__': main()

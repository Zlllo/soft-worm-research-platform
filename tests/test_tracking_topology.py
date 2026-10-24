import importlib
from pathlib import Path

import numpy as np
import pytest


def test_budget_selection_is_nested_balanced_and_preserves_whole_episodes(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'analysis'))
    module = importlib.import_module('compare_tracking_topology')
    data = {k: np.arange(24) for k in ('x', 'y', 'times', 'fields')}
    data['episodes'] = [dict(field=field, seed=seed, labels=4)
                        for field in ('linear', 'radial') for seed in (10, 11, 12)]
    first = module.subset(data, 1)
    second = module.subset(data, 2)
    assert first['x'].tolist() == list(range(4)) + list(range(12, 16))
    assert second['x'].tolist() == list(range(8)) + list(range(12, 20))
    assert set(first['x']) < set(second['x'])
    assert [row['seed'] for row in first['episodes']] == [10, 10]
    assert len(first['episodes']) == 2
    with pytest.raises(ValueError, match='insufficient'):
        module.subset(data, 4)

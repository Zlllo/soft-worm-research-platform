"""Native counterfactual: reference replay preserves actuation, not geometry."""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from thermotaxis.neuromechanics.boyle_body import BoyleBody
from thermotaxis.neuromechanics.boyle_output import side_lengths

ANALYSIS=Path(__file__).resolve().parents[1]/'analysis/neuromechanics'
sys.path.insert(0,str(ANALYSIS))
spec=importlib.util.spec_from_file_location('boyle_load_builder',ANALYSIS/'build_boyle_load_worker.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def test_native_replay_identity_and_load_counterfactual(tmp_path):
    source=os.environ.get('BOYLE_UPSTREAM');base=os.environ.get('BOYLE_BASE_BUILD')
    if not source or not base:pytest.skip('set BOYLE_UPSTREAM and BOYLE_BASE_BUILD for native load check')
    reference=tmp_path/'reference.bin'
    def simulate(binary):
        with BoyleBody(binary) as body:
            body.reset();states=[body.advance(.02) for _ in range(50)]
        return {name:np.stack([getattr(s,name) for s in states]) for name in ('pose','neural','muscle','side_length')}
    nominal=simulate(builder.build(source,base,tmp_path/'nominal',record=reference))
    replay=simulate(builder.build(source,base,tmp_path/'replay',replay=reference))
    for name in nominal:np.testing.assert_array_equal(nominal[name],replay[name])
    loaded=simulate(builder.build(source,base,tmp_path/'loaded',factor=2,onset=.2,duration=.2,replay=reference))
    np.testing.assert_array_equal(nominal['neural'],loaded['neural'])
    np.testing.assert_array_equal(nominal['muscle'],loaded['muscle'])
    assert np.max(abs(nominal['pose'][:,:,:2]-loaded['pose'][:,:,:2]))>1e-8
    np.testing.assert_allclose(loaded['side_length'],side_lengths(loaded['pose']),atol=1e-18,rtol=0)
    trace=np.fromfile(tmp_path/'loaded/feedback-trace.bin').reshape(-1,194)
    assert np.max(abs(trace[:,1:97]-trace[:,97:193]))>1e-9

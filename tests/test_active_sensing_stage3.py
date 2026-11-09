import importlib.util
from pathlib import Path
import sys

import numpy as np

ANALYSIS=Path(__file__).resolve().parents[1]/'analysis';sys.path.insert(0,str(ANALYSIS))
spec=importlib.util.spec_from_file_location('active_sensing_stage3',ANALYSIS/'run_active_sensing_stage3.py')
stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)


def test_mismatched_past_reference_preserves_power_and_no_future_sample():
    rng=np.random.default_rng(5);motion=rng.normal(size=(6,41,2)).astype('float32')
    shifts=np.arange(1,7);altered=stage.circular_mismatch(motion,shifts)
    np.testing.assert_allclose(np.mean(motion.astype('float64')**2,axis=1),np.mean(altered.astype('float64')**2,axis=1),atol=1e-14)
    np.testing.assert_allclose(abs(np.fft.rfft(motion.astype('float64'),axis=1)),
                               abs(np.fft.rfft(altered.astype('float64'),axis=1)),atol=1e-12)
    for i in range(6):
        np.testing.assert_array_equal(np.sort(motion[i,:,0]),np.sort(altered[i,:,0]))


def test_conditions_keep_identical_temperature_and_phase_splits_disjoint():
    rng=np.random.default_rng(6)
    data=dict(head_temperature=rng.normal(size=(6,41)).astype('float32'),
        center_temperature=rng.normal(size=(6,41)).astype('float32'),
        motion=rng.normal(size=(6,41,2)).astype('float32'),shift=np.arange(1,7))
    features=[stage.make_features(data,np.arange(6),c) for c in stage.CONDITIONS]
    np.testing.assert_array_equal(features[0][:,:,:2],features[1][:,:,:2])
    np.testing.assert_array_equal(features[1][:,:,:2],features[2][:,:,:2])
    np.testing.assert_array_equal(features[1][:,:,4:6],features[3][:,:,4:6])
    assert np.all(features[0][:,:,4:6]==0)
    assert [stage.group_split(i) for i in range(8)]==[0,1,0,2,0,1,0,2]

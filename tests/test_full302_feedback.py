import numpy as np
import pytest

from thermotaxis.neuromechanics.full302_feedback import LocalAFD,BendingFeedback


def test_local_temperature_sensor_adapts_and_cools_without_privileged_inputs():
    sensor=LocalAFD(293.15)
    assert sensor.advance(293.15,.1)==0
    initial=sensor.advance(293.20,.1)
    assert 0<initial<5
    for _ in range(2000):last=sensor.advance(293.20,.1)
    assert last<initial/10
    for _ in range(20):last=sensor.advance(293.10,.1)
    assert last==0


def test_feedback_is_side_specific_and_does_not_confuse_extension_with_bending():
    feedback=BendingFeedback.anatomical_bins();rest=np.ones((48,2))
    assert len(feedback.names)==18
    np.testing.assert_allclose(feedback.weights.sum(1),1)
    np.testing.assert_array_equal(feedback.currents(rest,rest),np.zeros(18))
    np.testing.assert_array_equal(feedback.currents(rest*1.01,rest),np.zeros(18))
    bent=rest.copy();bent[:,0]*=.99;bent[:,1]*=1.01
    current=feedback.currents(bent,rest)
    assert np.all(current[feedback.sides==0]>0) and np.all(current[feedback.sides==1]<0)
    np.testing.assert_allclose(feedback.currents(bent[:,::-1],rest),-current)


def test_feedback_fields_are_explicitly_local_and_invalid_values_fail():
    f=BendingFeedback.anatomical_bins();rest=np.ones((48,2))
    for row,center in zip(f.weights,f.nominal_output_centers):
        assert 1<=np.count_nonzero(row)<=10
        assert np.flatnonzero(row).max()<max(1,int(np.floor(center)))
    with pytest.raises(ValueError):f.currents(rest,np.zeros_like(rest))
    with pytest.raises(ValueError):LocalAFD(293.15).advance(float('nan'),.1)

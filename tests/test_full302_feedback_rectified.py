import numpy as np

from thermotaxis.neuromechanics.full302_feedback_rectified import ExcitatoryBendingFeedback


def test_rectified_feedback_exposes_opposite_bending_without_negative_current():
    f=ExcitatoryBendingFeedback.anatomical_bins();rest=np.ones((48,2))
    dorsal=rest.copy();dorsal[:,0]=.99;dorsal[:,1]=1.01
    a=f.currents(dorsal,rest);b=f.currents(dorsal[:,::-1],rest)
    assert np.all(a>=0) and np.all(b>=0)
    assert np.all(a[f.sides==0]>0) and np.all(a[f.sides==1]==0)
    assert np.all(b[f.sides==1]>0) and np.all(b[f.sides==0]==0)
    np.testing.assert_array_equal(f.currents(rest,rest),np.zeros(18))

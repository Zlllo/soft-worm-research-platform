import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis'))
from reservoir_stability_stage import simulate

def test_heun_improves_known_solution_and_is_causal():
 u=np.ones((1,10)); W=np.zeros((1,1)); g=np.zeros((1,1)); inputs=np.ones(1)
 exact=np.tanh(1)*(1-np.exp(-1/.3))
 coarse=simulate(u,W,g,inputs,.1,'heun')[0,-1,0]
 fine=simulate(u,W,g,inputs,.05,'heun')[0,-1,0]
 assert abs(fine-exact)<abs(coarse-exact)
 changed=u.copy(); changed[:,5:]=0
 np.testing.assert_array_equal(simulate(u,W,g,inputs,.05,'heun')[:,:5],simulate(changed,W,g,inputs,.05,'heun')[:,:5])

import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
from reservoir_generalization_stage import fit, assess

def test_fit_recovers_linear_relation_and_keeps_input():
 x=np.linspace(-1,1,100)[:,None]; old=x.copy(); y=2*x[:,0]+3
 m,b,c,penalty=fit(x[:60],y[:60],x[60:80],y[60:80])
 np.testing.assert_allclose((x[80:]-m)@c+b,y[80:],atol=1e-5)
 np.testing.assert_array_equal(x,old)

def test_assess_separate_test_and_constant_perfect_target():
 rng=np.random.default_rng(11); x=rng.normal(size=(112,120,1)); y=2*x[:,:,0]+1
 result=assess(x,[x[80:]],y,16)
 assert result['scores'][0] > .99999
 assert np.isfinite(result['fit_seconds'])

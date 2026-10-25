import math
import numpy as np
from thermotaxis.motor_feedback import simulate_feedback


def test_port_matches_independent_scalar_source_equations_with_same_draws():
    rng=np.random.default_rng(7);noise=rng.normal(size=(60,1));turn=rng.uniform(size=(60,1));angle=.4;dt=200/1500
    hist,head=simulate_feedback(1,noise,turn,np.array([angle]))
    x=[np.array([1.,1.,0.,0.])];phi=angle
    for j in range(1,61):
        a,m,px,py=x[-1];old=x[max(0,j-2)][1];flag=int(old>=0)*10+int(m>=0)
        if flag==10:
            phi+=math.pi
            if phi>2*math.pi:phi-=2*math.pi
        elif flag==1:phi=turn[j-1,0]*2*math.pi
        sensory=(px-x[max(0,j-4)][2])*(-1.5)*(m>0)
        sigmoid=lambda v:1/(1+math.exp(-v))
        da=1.5*(sensory+.5+2*noise[j-1,0]+sigmoid(15*m)-a)*dt
        dm=1.5*(sigmoid(5*(a-1.5))-.7*sigmoid(-5*(a-1.5))-m)*dt
        x.append(np.array([a+da,m+dm,px+math.cos(phi)*dt,py+math.sin(phi)*dt]))
    assert np.allclose(hist[:,0],np.stack(x),atol=1e-12,rtol=0)

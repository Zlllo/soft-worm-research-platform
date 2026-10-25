"""Vectorized SI navigation: acquire, track, and reacquire after a field change.

Only local noisy head temperature and explicitly enabled proprioception enter
observations. Position, field state and reward geometry remain inside the env.
Old near-isotherm tasks are unchanged. Metrics use the declared control grid.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NavigationConfig:
    schema_version: int = 3
    duration_s: float = 240.
    event_s: float = 120.
    event_jitter_s: float = 20.
    initial_error_min_K: float = .06
    initial_error_max_K: float = .15
    preferred_K: float = 293.15
    comfort_K: float = .05
    gradient_K_m: float = 40.
    radius_m: float = .005
    field_turn_rad: float = math.pi / 3
    field_shift_K: float = .08
    radial_shift_m: float = .002
    speed_m_s: float = .0002
    turn_limit_rad_s: float = .3
    response_tau_s: float = .3
    length_m: float = .001
    sweep_m: float = .00015
    frequency_Hz: float = .5
    noise_K: float = .002
    physics_dt_s: float = .02
    sensor_dt_s: float = .1
    control_dt_s: float = .2
    stable_s: float = 10.
    angle_tolerance_rad: float = math.radians(25)
    allow_reverse: bool = False

    def __post_init__(self):
        if self.schema_version != 3:raise ValueError('navigation schema must be 3')
        for k,v in asdict(self).items():
            if k in ('schema_version','allow_reverse'):continue
            if not isinstance(v,(float,int)) or not math.isfinite(v):raise ValueError('nonfinite '+k)
        positive=['duration_s','initial_error_min_K','initial_error_max_K','comfort_K','gradient_K_m','radius_m',
            'speed_m_s','turn_limit_rad_s','length_m','frequency_Hz','physics_dt_s','sensor_dt_s','control_dt_s','stable_s']
        if any(getattr(self,k)<=0 for k in positive):raise ValueError('positive scales required')
        if self.initial_error_min_K<=self.comfort_K or self.initial_error_max_K<self.initial_error_min_K:raise ValueError('outside-band initial errors required')
        if self.radius_m<=self.initial_error_max_K/self.gradient_K_m:raise ValueError('radial initial radius must be positive')
        if min(self.response_tau_s,self.noise_K,self.sweep_m,self.event_jitter_s,self.radial_shift_m)<0:raise ValueError('negative scale')
        if not self.stable_s<self.event_s-self.event_jitter_s<=self.event_s+self.event_jitter_s<self.duration_s-self.stable_s:raise ValueError('event and stable intervals overlap boundary')
        for v in [self.sensor_dt_s,self.control_dt_s,self.duration_s,self.stable_s]:
            ratio=v/self.physics_dt_s
            if abs(ratio-round(ratio))>1e-9:raise ValueError('clocks must align')
        if abs(self.control_dt_s/self.sensor_dt_s-round(self.control_dt_s/self.sensor_dt_s))>1e-9:raise ValueError('sensor/control clocks must align')
        if self.sensor_dt_s*self.frequency_Hz>=.5:raise ValueError('undersampled sweep')


class NavigationBatch:
    """A synchronous episode batch; one reproducible seed per episode.

    Observations: [temperature K, sin phase, cos phase, signed speed ratio,
    valid sample]. Policy packets never include event time, x/y, or heading.
    """
    def __init__(self,config,seeds,fields=None,phase_mode='correct',record=False):
        self.config=config;self.seeds=list(map(int,seeds));self.n=len(self.seeds)
        if not self.n:raise ValueError('empty episode batch')
        self.fields=np.asarray(fields if fields is not None else [i%2 for i in range(self.n)],dtype=int)
        if self.fields.shape!=(self.n,) or np.any((self.fields!=0)&(self.fields!=1)):raise ValueError('field must be linear0/radial1')
        if phase_mode not in ('correct','none','shifted'):raise ValueError('invalid phase mode')
        self.phase_mode=phase_mode;self.record=record;self.time=0.;self.steps=0
        self.angle=np.empty(self.n);self.sign=np.empty(self.n);self.event=np.empty(self.n)
        self.x=np.empty(self.n);self.y=np.empty(self.n);self.heading=np.empty(self.n);self.phase0=np.empty(self.n)
        self.offset=np.empty(self.n);self.noise_rng=[]
        for i,seed in enumerate(self.seeds):
            env=np.random.default_rng(np.random.SeedSequence([seed,0]));init=np.random.default_rng(np.random.SeedSequence([seed,1]))
            self.noise_rng.append(np.random.default_rng(np.random.SeedSequence([seed,2])))
            self.angle[i]=env.uniform(-math.pi,math.pi);self.sign[i]=env.choice([-1.,1.])
            raw=config.event_s+env.uniform(-config.event_jitter_s,config.event_jitter_s)
            self.event[i]=round(raw/config.control_dt_s)*config.control_dt_s
            self.offset[i]=init.choice([-1.,1.])*init.uniform(config.initial_error_min_K,config.initial_error_max_K)
            a=self.angle[i] if self.fields[i]==0 else init.uniform(-math.pi,math.pi)
            r=self.offset[i]/config.gradient_K_m+(config.radius_m if self.fields[i] else 0.)
            self.x[i]=r*math.cos(a);self.y[i]=r*math.sin(a)
            self.heading[i]=init.uniform(-math.pi,math.pi);self.phase0[i]=init.uniform(-math.pi,math.pi)
        self.omega=np.zeros(self.n);self.velocity=np.ones(self.n)*config.speed_m_s
        self.history=[];self.actions=[]
        self.initial=dict(x_m=self.x.copy(),y_m=self.y.copy(),heading_rad=self.heading.copy(),offset_K=self.offset.copy(),event_s=self.event.copy())
        self._store()

    def field(self,x,y,time):
        c=self.config;after=np.asarray(time)>=self.event-1e-9
        angle=self.angle+after*self.sign*c.field_turn_rad
        cx=after*c.radial_shift_m*np.cos(self.angle);cy=after*c.radial_shift_m*np.sin(self.angle)
        linear=c.gradient_K_m*(x*np.cos(angle)+y*np.sin(angle))-after*self.sign*c.field_shift_K
        radial=c.gradient_K_m*(np.hypot(x-cx,y-cy)-c.radius_m)
        return np.where(self.fields==0,linear,radial)

    def _sample(self):
        c=self.config;phase=self.phase0+2*math.pi*c.frequency_Hz*self.time
        hx=self.x+c.length_m/2*np.cos(self.heading)-c.sweep_m*np.sin(phase)*np.sin(self.heading)
        hy=self.y+c.length_m/2*np.sin(self.heading)+c.sweep_m*np.sin(phase)*np.cos(self.heading)
        noise=np.array([r.normal(0,c.noise_K) for r in self.noise_rng])
        temp=c.preferred_K+self.field(hx,hy,self.time)+noise
        observed=phase if self.phase_mode=='correct' else phase+math.pi/2
        sine=np.sin(observed) if self.phase_mode!='none' else np.zeros(self.n)
        cosine=np.cos(observed) if self.phase_mode!='none' else np.zeros(self.n)
        return np.stack([temp,sine,cosine,self.velocity/c.speed_m_s,np.ones(self.n)],axis=1).astype(np.float32)

    def initial_packet(self):
        k=round(self.config.control_dt_s/self.config.sensor_dt_s)
        result=np.zeros((k,self.n,5),dtype=np.float32);result[-1]=self._sample();return result

    def _store(self):
        c=self.config;after=self.time>=self.event-1e-9
        normal=np.where(self.fields==0,self.angle+after*self.sign*c.field_turn_rad,
            np.arctan2(self.y-after*c.radial_shift_m*np.sin(self.angle),self.x-after*c.radial_shift_m*np.cos(self.angle)))
        error=self.field(self.x,self.y,self.time)
        tangent=np.arcsin(np.clip(np.abs(np.cos(self.heading-normal)),0,1))
        # Store true geometry for evaluator only.
        self.history.append(np.stack([np.full(self.n,self.time),error,tangent,self.x,self.y,self.heading,self.omega,self.velocity],axis=1))

    def step(self,actions):
        c=self.config;a=np.asarray(actions,dtype=float)
        if a.shape!=(self.n,2) or not np.isfinite(a).all() or (np.abs(a)>1+1e-8).any():raise ValueError('bounded actions [batch,2] required')
        if self.steps>=round(c.duration_s/c.control_dt_s):raise RuntimeError('episode ended')
        target=a[:,0]*c.turn_limit_rad_s
        speed=c.speed_m_s*(np.where(a[:,1]>=0,1.,-1.) if c.allow_reverse else np.ones(self.n))
        previous=self.field(self.x,self.y,self.time)
        samples=[];physics=round(c.control_dt_s/c.physics_dt_s);sample_every=round(c.sensor_dt_s/c.physics_dt_s)
        for j in range(physics):
            dt=c.physics_dt_s
            if c.response_tau_s:
                integral=lambda h:target*h+(self.omega-target)*c.response_tau_s*(-math.expm1(-h/c.response_tau_s))
                mid=self.heading+integral(dt/2);angle_step=integral(dt)
                self.omega=target+(self.omega-target)*math.exp(-dt/c.response_tau_s)
            else:
                mid=self.heading+target*dt/2;angle_step=target*dt;self.omega=target.copy()
            self.x+=speed*dt*np.cos(mid);self.y+=speed*dt*np.sin(mid);self.heading+=angle_step
            self.velocity=speed
            self.time=(self.steps*physics+j+1)*c.physics_dt_s
            if (j+1)%sample_every==0:samples.append(self._sample())
        self.steps+=1;self._store();self.actions.append(a.copy())
        error=self.field(self.x,self.y,self.time)
        potential=lambda e:-np.log1p(np.abs(e)/c.comfort_K)
        reward=(potential(error)-potential(previous))+.02*np.exp(-np.abs(error)/c.comfort_K)-.0002*a[:,0]**2
        done=self.steps>=round(c.duration_s/c.control_dt_s)
        return np.stack(samples),reward.astype(np.float32),done

    def metrics(self):
        c=self.config;h=np.stack(self.history);t=h[:-1,:,0];e=h[:-1,:,1];angle=h[:-1,:,2]
        comfortable=np.abs(e)<=c.comfort_K;tracking=comfortable&(angle<=c.angle_tolerance_rad)
        result=[];dwell=round(c.stable_s/c.control_dt_s)
        for i in range(self.n):
            pre=t[:,i]<self.event[i]-1e-9;post=~pre
            def first(mask,start):
                ids=np.flatnonzero(mask)
                return None if not len(ids) else float(t[ids[0],i]-start)
            def stable(valid,start):
                run=0
                for j in range(len(valid)):
                    run=run+1 if valid[j] else 0
                    if run>=dwell:return float(t[j-dwell+1,i]-start)
                return None
            acquisition=first(comfortable[:,i]&pre,0.)
            stable_pre=stable(tracking[:,i]&pre,0.)
            recovery=first(comfortable[:,i]&post,self.event[i])
            stable_post=stable(tracking[:,i]&post,self.event[i])
            result.append(dict(seed=self.seeds[i],field='linear' if self.fields[i]==0 else 'radial',
                initial_error_K=float(self.offset[i]),event_s=float(self.event[i]),
                first_entry_s=acquisition,pre_stable_s=stable_pre,reentry_s=recovery,post_stable_s=stable_post,
                pre_tracking=float(tracking[pre,i].mean()),post_tracking=float(tracking[post,i].mean()),
                full_tracking=float(tracking[:,i].mean()),residence=float(comfortable[:,i].mean()),
                complete_success=stable_pre is not None and stable_post is not None,
                center_error_mae_K=float(np.abs(e[:,i]).mean()),
                prescribed_distance_m=float(np.abs(h[:-1,i,7]).sum()*c.control_dt_s)))
        return result

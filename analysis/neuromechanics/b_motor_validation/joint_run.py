"""Frozen full302/body loop with path-only overrides and failure preservation."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/full302-joint-numerics-2026-11-12'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def driver_module():
    sys.path.insert(0,str(ROOT/'runtime/src'))
    spec=importlib.util.spec_from_file_location('joint_frozen_driver',ROOT/'runtime/driver.py')
    driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
    network_class=driver.C1Network;body_class=driver.BoyleMuscleBody
    class PinnedNetwork(network_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert self.h.secondorder==0 and self.h.cvode_active()==0
            assert len(self.plan['neurons'])==302 and len(self.plan['muscles'])==95
            assert len(self.chemical_connections)==4665 and len(self.gap_connections)==1192
    class PinnedBody(body_class):
        def __init__(self,ignored_path):super().__init__(ROOT/'runtime/body/worker')
    driver.C1Network=PinnedNetwork;driver.BoyleMuscleBody=PinnedBody;driver.ROOT=ROOT
    return driver


def args_for(mode,dt):
    return argparse.Namespace(name=f'{mode}_dt{dt:g}',model='prototype',feedback=mode,replay=None,
        gap_nS=.1,feedback_gain=100,feedback_clip=1,duration=8.,dt_ms=dt,seed=13,avb_on=True,
        calcium_scale=1,perturb=False,nmj_nS=.9)


def run(mode,dt):
    p=read(ROOT/'protocol.json');execution=read(ROOT/'execution-lock.json')
    assert sha(ROOT/'protocol.json')==execution['protocol_sha256']
    assert sha(Path(__file__))==execution['runner_sha256']
    assert read(ROOT/'preflight.json')['passed'] and read(ROOT/'generated-code-validation.json')['passed']
    for name,digest in execution['runtime_sha256'].items():assert sha(ROOT/name)==digest,name
    assert mode in ['off','live'] and dt in [.025,.0125,.00625]
    if dt==.00625:
        decision=read(ROOT/'refinement-decision.json')
        assert decision['fine_required'] and decision['protocol_sha256']==sha(ROOT/'protocol.json')
    # An attempted condition consumes its one directory; no retries in place.
    assert not (ROOT/f'{mode}_dt{dt:g}').exists()
    driver=driver_module();a=args_for(mode,dt);os.chdir(PROJECT)
    try:
        driver.run(a)
    except BaseException as error:
        # Preserve already-computed samples from the unmodified driver's frame.
        # This runs only after an exception, never inside the integration loop.
        import numpy as np
        tb=error.__traceback__;local=None
        while tb:
            if Path(tb.tb_frame.f_code.co_filename).resolve()==(ROOT/'runtime/driver.py').resolve() and tb.tb_frame.f_code.co_name=='run':local=tb.tb_frame.f_locals
            tb=tb.tb_next
        out=ROOT/a.name;out.mkdir(exist_ok=True)
        if local and not (out/'trace.npz').exists() and local.get('rows'):
            rows=local['rows'];values={k:np.array([r[k] for r in rows]) for k in rows[0]}
            fb=local.get('feedback_rows',[])
            values.update(exchange_time_s=np.arange(len(fb))*.001,feedback_pA=np.array(fb),
                B_names=np.array(local['names']),neuron_names=np.array(local['w'].neuron_names),
                feedback_weights=local['weights'],feedback_sides=local['sides'],nominal_centers=local['centers'])
            np.savez_compressed(out/'trace.npz',**values)
        record=dict(spec=vars(a),complete=False,failure=local.get('failure') if local and local.get('failure') else repr(error),
            exception=repr(error),traceback=traceback.format_exc(),condition_attempt_consumed=True,
            trace_sha256=sha(out/'trace.npz') if (out/'trace.npz').exists() else None,
            saved_state_samples=len(local.get('rows',[])) if local else 0,
            saved_exchange_samples=len(local.get('feedback_rows',[])) if local else 0)
        write(out/'failure.json',record)
        if not (out/'record.json').exists():write(out/'record.json',record)
        raise
    finally:
        out=ROOT/a.name
        if out.exists():
            import neuron,numpy,scipy
            write(out/'provenance.json',dict(protocol_sha256=sha(ROOT/'protocol.json'),execution_lock_sha256=sha(ROOT/'execution-lock.json'),
                runner_sha256=sha(Path(__file__)),driver_sha256=sha(ROOT/'runtime/driver.py'),candidate='joint_exp_direct',
                changes='ROOT/body path overrides; constructor-only population and default assertions; post-error serialization only',
                extra_fcurrent_calls=0,extra_rates_calls=0,loop_changed=False,
                versions=dict(neuron=neuron.__version__,numpy=numpy.__version__,scipy=scipy.__version__),python=sys.version,
                secondorder=0,secondorder_source='Asserted current runtime default; no assignment',
                OPENBLAS_NUM_THREADS=os.environ.get('OPENBLAS_NUM_THREADS')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--feedback',choices=['off','live'],required=True)
    p.add_argument('--dt',type=float,choices=[.025,.0125,.00625],required=True)
    a=p.parse_args();run(a.feedback,a.dt)

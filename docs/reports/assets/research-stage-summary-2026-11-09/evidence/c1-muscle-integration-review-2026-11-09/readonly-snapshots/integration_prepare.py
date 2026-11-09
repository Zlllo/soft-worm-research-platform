"""Preregister and build isolated, minimal C1 integration variants."""
import argparse
from datetime import datetime, timezone
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-muscle-integration-isolation-2026-11-09'
FROZEN=PROJECT/'results/b-motor-convergence-2026-11-09'
VARIANTS=['original','gates','pool','both']
STEPS=[.025,.0125,.00625]
GATES={'ca_boyle':{'e_q':'e','f_q':'f'},'k_fast':{'p_q':'p','q_q':'q'},'k_slow':{'n_q':'n'}}
HELPER='''TITLE Prescribed calcium current density for isolated concentration controls
NEURON {
    SUFFIX IntegrationCaInput
    USEION ca WRITE ica VALENCE 2
    RANGE commanded, ica
}
UNITS { (mA) = (milliamp) }
PARAMETER { commanded = 0 (mA/cm2) }
ASSIGNED { ica (mA/cm2) }
BREAKPOINT { ica = commanded }
'''


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d): p.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def read(p): return json.loads(p.read_text())


def prepare():
    import numpy as np
    ROOT.mkdir(exist_ok=True)
    assert not (ROOT/'protocol.json').exists()
    assert {p.name for p in ROOT.iterdir()} <= {'task-control.json'}, 'Preserve pre-existing work'
    sources={}
    old=read(FROZEN/'inputs.json')
    for source in sorted((FROZEN/'kernels').rglob('*')):
        if source.is_file():
            rel=str(source.relative_to(FROZEN));assert sha(source)==old[rel]['sha256']
            target=ROOT/'frozen'/source.relative_to(FROZEN/'kernels');target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target);sources[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(PROJECT)),sha256=sha(target))
    for name in ['summary.json','archive-validation.json','numerical-diagnostic.md','numerical-diagnostic-evidence.json']:
        source=FROZEN/name;target=ROOT/'historical'/name;target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target);sources[str(target.relative_to(ROOT))]=dict(source=str(source.relative_to(PROJECT)),sha256=sha(target))
    (ROOT/'harness').mkdir();(ROOT/'harness/IntegrationCaInput.mod').write_text(HELPER)
    write(ROOT/'sources.json',sources)
    # A fixed dense input, not a reconstruction of recorded full-network voltage.
    tick=np.arange(1280001,dtype=np.int64);t=tick*.00625
    phase=2*np.pi*1.9*t/1000
    blend=np.clip((t-250)/100,0,1);blend=blend*blend*(3-2*blend)
    voltage=-60+blend*((-32+6*np.sin(phase))-(-60))
    np.savez_compressed(ROOT/'presynaptic-input.npz',time_ms=t,voltage_mV=voltage)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),variants=VARIANTS,dt_ms=STEPS,
        boundary='Isolated C1 muscle and analytic component controls only; no full302/body runs, no physical parameter tuning',
        patches='Only five gate derivative RHS and/or CaPool concentration derivative RHS; retain rate observers, initialization, caches and ion/clipping order',
        main=dict(duration_ms=8000,window_ms=[4000,8000],record_ms=.025,conditions=12,
            geometry='Frozen GenericMuscleCell.hoc, 20 um length / 5 um diameter, cm=1 uF/cm2',
            init='Frozen -45 mV and cai=0/cao=2; original gate initialization, no extra reset',
            nmj_conductance_nS=.9,nmj_Vth_mV=-30,nmj_weight=1,
            input='Dense .00625 ms file from analytic -60 mV then cubic smooth blend over 250..350 ms into -32+6sin(2*pi*1.9*t/1000) mV; original frozen NMJ',
            input_convention='Each neural interval [k*dt,(k+1)*dt) uses input file at its left endpoint. At record endpoints record the currently set peer input and separately the actually used previous-interval input. Common endpoint peer inputs must be bitwise identical. Same-dt synapse gates must be bitwise identical across four variants.',
            refinement='Exactly the three stated steps for all four variants; no additional conditions or parameter search'),
        controls=dict(variants=['original','both'],dt_ms=STEPS,
            gates=dict(duration_ms=600,voltage_mV=-10,SEClamp_rs_MOhm=1e-9,
                gmax_all_zero=True,initial_states=[.8,.2,.7,.1,.6],
                reference='xinf+(x0-xinf)*exp(-t/tau), coefficients from frozen MOD and measured fixed voltage',
                voltage_guard_mV=1e-6,initial_distance_min=.05,max_state_abs_error=1e-9),
            pool=dict(duration_ms=105,voltage_mV=-45,SEClamp_rs_MOhm=1e-9,gmax_all_zero=True,
                initial_concentration_mM=2e-7,
                actual_ica_mA_cm2=[[-.002,0,50],[0,50,100],[.002,100,105]],
                reference_window_ms=[0,100],stress_window_ms=[100,105],
                reference='For actual measured source density J, dC/dt=(-1e6*J)*rho-(C-rest)/tau; piecewise analytic input/zero decay. Outward segment is sync/clipping stress only.',
                source_density_abs_tolerance_mA_cm2=1e-12,total_current_abs_tolerance_nA=1e-12,
                state_max_abs_error_mM=1e-15,ion_sync_abs_tolerance_mM=1e-15)),
        acceptance=dict(voltage_RMS_mV=.5,voltage_mean_abs_mV=.2,voltage_amplitude_relative=.05,voltage_amplitude_floor_mV=1,
            calcium_scaled_stats_and_RMS=.05,calcium_scale_floor_mM=1e-9,
            drive_scaled_stats_and_RMS=.05,drive_scale_floor=.001,gate_state_RMS=.01,
            scaling='Finer reference q95-q05 with floors; statistics = mean/std/q05/q95; report state concentration and cai separately',
            improvement='For combined vs original compare last-pair V RMS, STATE calcium scaled RMS, cai scaled RMS, drive scaled RMS. Report every ratio. Strong improvement requires all ratios<=0.5, except any original error already <=1/10 of its acceptance tolerance.',
            recommendation='Consider full-system numerical validation only if code and combined analytic STATE controls pass, combined last-pair main gates all pass, and strong improvement holds. Any cache/ion residual remains separately disclosed; full-system convergence is never inferred.'),
        input_sha256=sha(ROOT/'presynaptic-input.npz'),source_manifest_sha256=sha(ROOT/'sources.json'),
        helper_sha256=sha(ROOT/'harness/IntegrationCaInput.mod'),prepare_script_sha256=sha(Path(__file__)))
    write(ROOT/'protocol.json',protocol);write(ROOT/'protocol-lock.json',dict(protocol_sha256=sha(ROOT/'protocol.json')))
    print(json.dumps(dict(protocol_sha256=sha(ROOT/'protocol.json'),main_conditions=12,control_conditions=12)))


def state_function(path):
    text=path.read_text();start=text.index('static int states (')
    return text[start:text.index('static int  rates',start)]


def build():
    assert sha(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    for path,d in read(ROOT/'sources.json').items(): assert sha(ROOT/path)==d['sha256']
    (ROOT/'variants').mkdir(exist_ok=True);evidence={}
    for variant in VARIANTS:
        target=ROOT/'variants'/variant;target.mkdir(exist_ok=True)
        changed=[]
        for source in sorted((ROOT/'frozen').iterdir()):
            if source.suffix not in ['.mod','.hoc']:continue
            text=source.read_text();updated=text
            if variant in ['gates','both'] and source.stem in GATES:
                for state,prefix in GATES[source.stem].items():
                    old=f"{state}' = rate_{state}"
                    assert updated.count(old)==1
                    updated=updated.replace(old,f"{state}' = ({prefix}_inf-{state})/{prefix}_tau")
            if variant in ['pool','both'] and source.name=='CaPool.mod':
                old="concentration' = rate_concentration";assert updated.count(old)==1
                updated=updated.replace(old,"concentration' = (iCa/surfaceArea)*rho-(concentration-restingConc)/decayConstant")
            (target/source.name).write_text(updated)
            if updated!=text:
                changed.append(source.name)
        shutil.copy2(ROOT/'harness/IntegrationCaInput.mod',target/'IntegrationCaInput.mod')
        diff=[]
        for name in changed:
            diff.extend(difflib.unified_diff((ROOT/'frozen'/name).read_text().splitlines(True),(target/name).read_text().splitlines(True),fromfile='frozen/'+name,tofile=variant+'/'+name))
        (target/'patch.diff').write_text(''.join(diff))
        if not (target/'arm64/libnrnmech.dylib').exists():
            with (target/'build.log').open('w') as log:
                subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')],cwd=target,stdout=log,stderr=subprocess.STDOUT,check=True)
        states={}
        for mech in ['ca_boyle','k_fast','k_slow','CaPool']:
            block=state_function(target/'arm64'/f'{mech}.cpp')
            expected_exp=(variant in ['gates','both'] and mech!='CaPool') or (variant in ['pool','both'] and mech=='CaPool')
            assert ('exp(' in block)==expected_exp,(variant,mech,block)
            if expected_exp:
                assert ('decayConstant' in block if mech=='CaPool' else '_tau' in block)
                compact=re.sub(r'\s+','',block)
                assert ('-((1.0))/' in compact if mech=='CaPool' else '-1.0' in compact)
            states[mech]=dict(exponential_generated=expected_exp,code=block)
            if variant=='original':
                assert re.sub(r'\s+','',block)==re.sub(r'\s+','',state_function(ROOT/'frozen/arm64'/f'{mech}.cpp'))
        hashes={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}
        evidence[variant]=dict(changed_original_files=changed,generated_states=states,sha256=hashes)
        print('compiled',variant,changed,flush=True)
    for mech in GATES:
        assert state_function(ROOT/'variants/gates/arm64'/f'{mech}.cpp')==state_function(ROOT/'variants/both/arm64'/f'{mech}.cpp')
    assert state_function(ROOT/'variants/pool/arm64/CaPool.cpp')==state_function(ROOT/'variants/both/arm64/CaPool.cpp')
    write(ROOT/'build-evidence.json',evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build']);a=p.parse_args()
    prepare() if a.action=='prepare' else build()

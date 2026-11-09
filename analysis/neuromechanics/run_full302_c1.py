"""Build copied C1 kernels, verify a reference pair, run persistent full302 cases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from thermotaxis.neuromechanics.full302_c1 import C1Network, connection_plan
from thermotaxis.neuromechanics.anatomical_body import load_anatomical_body_wiring, PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_muscle_body import BoyleMuscleBody


REPO = Path(__file__).resolve().parents[2]
DEFAULT = Path('results/full302-c1-dynamics-2026-11-05')
UPSTREAM = Path('results/neuromechanics-stage0-2026-10-27/upstream/c302')
REFERENCE = Path('results/neuromechanics-stage0-2026-10-27/outputs/c302-c1-neuron-wrapper')
KERNEL_FILES = ['GenericNeuronCell.hoc', 'GenericMuscleCell.hoc', 'Leak.mod', 'k_fast.mod', 'k_slow.mod',
    'ca_boyle.mod', 'CaPool.mod', 'neuron_to_neuron_exc_syn.mod', 'neuron_to_neuron_inh_syn.mod', 'neuron_to_neuron_elec_syn.mod']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(root):
    if root.exists():
        raise FileExistsError('use a new root to preserve experiments')
    root.mkdir(parents=True)
    metadata = UPSTREAM / 'c302/data/owmeta_cache.json'
    plan = connection_plan(json.loads(metadata.read_text()), unresolved_polarity='exc')
    write(root / 'connection-plan.json', plan)
    files = [Path('src/thermotaxis/neuromechanics/full302_c1.py'), Path(__file__).resolve().relative_to(REPO),
        Path('data/connectome/cook2019_herm.json'), Path('data/connectome/herm_neurons.json'), metadata,
        Path('src/thermotaxis/neuromechanics/anatomical_body.py'),
        Path('src/thermotaxis/neuromechanics/boyle_muscle_body.py')]
    files += [REFERENCE / n for n in KERNEL_FILES]
    files += [REFERENCE / n for n in ['c302_C1_Syns.dat', 'c302_C1_Syns.muscles.dat', 'c302_C1_Syns.muscles.activity.dat']]
    files += [Path('results/full-network-body-interface-2026-11-05/build') / n for n in ['build.json', 'worker']]
    design = dict(status='declared_before_full302_conductance_model_execution',
        neural_model='copied pinned c302 C1 NEURON exports; single-compartment voltage/calcium, graded synapses',
        neurons=302, muscles=95, dt_ms=.05, duration_s=1., case_names=['background', 'AS2_pulse', 'AS2_nmj_lesion', 'AFD_pulse'],
        pulse=dict(start_s=.2, duration_s=.4, amplitude_pA=5.),
        chemistry='GABA-labelled sources inhibitory; every other source explicitly assumed excitatory, not receptor resolved',
        efficacy='unit anatomical mask times C1 base conductance; no EM-to-efficacy inference, no fitted weights',
        gaps='one undirected pair, two equal-conductance current mechanisms; include cross neuron-muscle and muscle-muscle',
        background='no offset current, no PLM demo stimulation, no hidden locomotor drive',
        isolated_pair='AS2->MDL07 weight3, exact historical stimuli1/5pA for C1 component validation only',
        coupled_body=dict(worker='results/full-network-body-interface-2026-11-05/build/worker', exchange_ms=1.,
            activation='0.3 * max(calcium,0) / (max(calcium,0) + 2.5e-7 mM)',
            scope='explicit uncalibrated calcium-to-drive map, existing100ms mechanical muscle filter retained',
            feedback='not injected into neurons in this stage; strain recorded'),
        c302_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),
        input_and_source_sha256={str(p): sha(p) for p in files},
        versions=dict(python=sys.version, numpy=np.__version__))
    write(root / 'design.json', design)
    kernel = root / 'kernels'; kernel.mkdir()
    for name in KERNEL_FILES:
        shutil.copyfile(REFERENCE / name, kernel / name)
    import neuron
    env = os.environ.copy()
    env['NEURON_HOME'] = str(Path(neuron.__file__).parent / '.data')
    with (root / 'kernel-build.log').open('w') as log:
        subprocess.run([str(Path(sys.executable).parent / 'nrnivmodl')], cwd=kernel, env=env,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    write(root / 'kernel-build.json', dict(files={p.name: sha(p) for p in kernel.iterdir() if p.is_file()},
        neuron_version=neuron.__version__, complete=True))


def check(root):
    d = json.loads((root / 'design.json').read_text())
    for path, h in d['input_and_source_sha256'].items():
        if sha(REPO / path) != h:
            raise RuntimeError('changed source/input: '+path)
    return d


def verify_pair(root):
    network = C1Network(root / 'kernels', dict(neurons=['AS2'], muscles=['MDL07'],
        chemical=[('AS2','MDL07','exc')], gap_pairs=[]))
    network.chemical_connections[('AS2','MDL07')].weight = 3
    network.add_current('AS2', start_ms=500, duration_ms=800, amplitude_pA=1)
    network.add_current('AS2', start_ms=1900, duration_ms=800, amplitude_pA=5)
    network.reset(); rows = []
    for i in range(72001):
        if i:
            network.advance(.05)
        s = network.state()
        rows.append([s['time_s'],s['neuron_voltage_mV'][0]/1000,s['muscle_voltage_mV'][0]/1000,s['muscle_calcium_mM'][0]])
    values = np.array(rows)
    reference_v = np.loadtxt(REFERENCE / 'c302_C1_Syns.dat')
    reference_m = np.loadtxt(REFERENCE / 'c302_C1_Syns.muscles.dat')
    reference_ca = np.loadtxt(REFERENCE / 'c302_C1_Syns.muscles.activity.dat')
    errors = dict(neuron_voltage_V=float(abs(values[:,1]-reference_v[:,2]).max()),
                  muscle_voltage_V=float(abs(values[:,2]-reference_m[:,1]).max()),
                  muscle_calcium_mM=float(abs(values[:,3]-reference_ca[:,1]).max()))
    np.savez_compressed(root / 'component-pair.npz', values=values)
    # Historical text output retains seven significant digits; allow that precision.
    passed = errors['neuron_voltage_V'] < 1e-7 and errors['muscle_voltage_V'] < 1e-7 and errors['muscle_calcium_mM'] < 1e-12
    write(root / 'component-pair.json', dict(max_abs_difference=errors, passed=passed,
        scope='two-cell kernel reconstruction, not the full302 test'))
    if not passed:
        raise RuntimeError('component reconstruction differs from the previous C1 reference')


def run_case(root, name, *, dt_ms=.05):
    d = check(root)
    assert json.loads((root / 'component-pair.json').read_text())['passed']
    plan = json.loads((root / 'connection-plan.json').read_text())
    folder = root / (name + ('-halfdt' if dt_ms != .05 else ''))
    folder.mkdir(exist_ok=False)
    lesion = []
    if name == 'AS2_nmj_lesion':
        lesion = [(pre,post) for pre,post,_ in plan['chemical'] if pre=='AS2' and post in plan['muscles']]
    started=time.monotonic()
    network = C1Network(root / 'kernels', plan, dt_ms=dt_ms, chemical_lesion=lesion)
    if name != 'background':
        cells = ['AFDL','AFDR'] if name=='AFD_pulse' else ['AS2']
        for cell in cells:
            network.add_current(cell,start_ms=200,duration_ms=400,amplitude_pA=5)
    assert len(network.cells)==397
    assert len(network.chemical_connections)==4665
    w=load_anatomical_body_wiring(); projection=PlanarMuscleProjection.rank_bins(w.muscle_names)
    network.reset()
    neural=[]; muscle_v=[]; calcium=[]; times=[]; poses=[]; body_m=[]; strains=[]; drives=[]
    body_binary=REPO / d['coupled_body']['worker']
    with BoyleMuscleBody(body_binary) as body:
        body.reset()
        for tick in range(1001):
            s=network.state()
            if tick%10==0:
                times.append(s['time_s']); neural.append(s['neuron_voltage_mV']); muscle_v.append(s['muscle_voltage_mV'])
                calcium.append(s['muscle_calcium_mM']);poses.append(body.state.pose.copy())
                body_m.append(body.state.muscle.copy());strains.append(body.state.stretch.copy())
            if tick==1000:
                break
            c=np.maximum(s['muscle_calcium_mM'],0)
            activation=.3*c/(c+2.5e-7)
            command=projection.to_segments(activation)
            # Both engines start from t; the command uses only the accepted t state.
            body.step(1,muscle_drive=command)
            network.advance(1.)
            assert abs(body.state.time_s-float(network.h.t)/1000)<1e-10
            if tick%10==0:
                drives.append(command)
    values=dict(time_s=np.array(times),neuron_voltage_mV=np.stack(neural),muscle_voltage_mV=np.stack(muscle_v),
        muscle_calcium_mM=np.stack(calcium),pose=np.stack(poses),body_muscle=np.stack(body_m),
        body_stretch=np.stack(strains),commanded_drive=np.stack(drives))
    np.savez_compressed(folder / 'trace.npz',**values)
    gap_currents=[float(abs(a.i+b.i)) for a,b in network.gap_connections]
    result=dict(name=name,dt_ms=dt_ms,duration_s=1.,neurons=302,muscles=95,
        chemical_connections=4665,gap_pairs=len(network.gap_connections),lesioned_edges=len(lesion),
        neuron_voltage_range_mV=[float(values['neuron_voltage_mV'].min()),float(values['neuron_voltage_mV'].max())],
        muscle_voltage_range_mV=[float(values['muscle_voltage_mV'].min()),float(values['muscle_voltage_mV'].max())],
        calcium_range_mM=[float(values['muscle_calcium_mM'].min()),float(values['muscle_calcium_mM'].max())],
        max_gap_pair_current_imbalance_nA=max(gap_currents),
        nonfinite=False,neural_training=False,mechanical_feedback_injected=False,
        wall_s=time.monotonic()-started,trace_sha256=sha(folder/'trace.npz'),complete=True)
    write(folder/'record.json',result);check(root);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','verify-pair','case'])
    p.add_argument('--root',type=Path,default=DEFAULT)
    p.add_argument('--name',choices=['background','AS2_pulse','AS2_nmj_lesion','AFD_pulse'])
    p.add_argument('--dt-ms',type=float,default=.05)
    a=p.parse_args()
    if a.stage=='prepare':prepare(a.root)
    elif a.stage=='verify-pair':check(a.root);verify_pair(a.root)
    else:run_case(a.root,a.name,dt_ms=a.dt_ms)

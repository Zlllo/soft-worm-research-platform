"""Audit all full302 dynamic runs, retain failures, and report actual effect sizes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_full302_c1 import check, sha, write, REPO, UPSTREAM
from thermotaxis.neuromechanics.full302_c1 import connection_plan


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x)**2)))


def main(root, output):
    design=check(root)
    plan=json.loads((root/'connection-plan.json').read_text())
    expected=connection_plan(json.loads((UPSTREAM/'c302/data/owmeta_cache.json').read_text()),unresolved_polarity='exc')
    assert plan==json.loads(json.dumps(expected))
    assert len(plan['neurons'])==302 and len(plan['muscles'])==95 and len(plan['chemical'])==4665
    assert len(plan['gap_pairs'])==1192
    pair=json.loads((root/'component-pair.json').read_text());assert pair['passed']
    records=[];traces={}
    paths=design['case_names']+['AS2_pulse-halfdt','AFD_pulse-halfdt','numerical-quarterstep/AFD_pulse-halfdt']
    for name in paths:
        folder=root/name;r=json.loads((folder/'record.json').read_text())
        assert r['complete'] and r['neurons']==302 and r['muscles']==95
        assert r['chemical_connections']==4665 and r['gap_pairs']==1192
        assert r['max_gap_pair_current_imbalance_nA']<1e-12
        assert sha(folder/'trace.npz')==r['trace_sha256']
        with np.load(folder/'trace.npz') as z:
            traces[name]={k:z[k] for k in z.files}
        z=traces[name]
        assert z['neuron_voltage_mV'].shape==(101,302) and z['muscle_voltage_mV'].shape==(101,95)
        assert z['muscle_calcium_mM'].shape==(101,95) and z['pose'].shape==(101,49,3)
        for value in z.values():
            assert np.isfinite(value).all()
        np.testing.assert_allclose(z['time_s'],np.arange(101)*.01,rtol=0,atol=1e-10)
        records.append(dict(path=name,**r))
    bg=traces['background'];responses=[]
    targets=[plan['muscles'].index(post) for pre,post,_ in plan['chemical'] if pre=='AS2' and post in plan['muscles']]
    assert len(targets)==6
    for name in design['case_names']:
        z=traces[name]
        delta_ca=z['muscle_calcium_mM']-bg['muscle_calcium_mM']
        responses.append(dict(name=name,
            max_neuron_voltage_change_mV=float(abs(z['neuron_voltage_mV']-bg['neuron_voltage_mV']).max()),
            max_muscle_voltage_change_mV=float(abs(z['muscle_voltage_mV']-bg['muscle_voltage_mV']).max()),
            max_muscle_calcium_change_mM=float(abs(delta_ca).max()),
            AS2_target_calcium_delta_rms_mM=rms(delta_ca[:,targets]),
            max_body_point_displacement_um=float(np.linalg.norm(z['pose'][:,:,:2]-bg['pose'][:,:,:2],axis=2).max()*1e6),
            max_rod_angle_change_rad=float(abs(z['pose'][:,:,2]-bg['pose'][:,:,2]).max()),
            max_command=float(z['commanded_drive'].max()),
            max_filtered_muscle=float(z['body_muscle'].max())))
    a=traces['AS2_pulse'];b=traces['AS2_nmj_lesion']
    as2=plan['neurons'].index('AS2')
    lesion=dict(edges_removed=6,
        AS2_voltage_max_difference_mV=float(abs(a['neuron_voltage_mV'][:,as2]-b['neuron_voltage_mV'][:,as2]).max()),
        target_calcium_delta_rms_ratio=float(rms((b['muscle_calcium_mM']-bg['muscle_calcium_mM'])[:,targets])/
            rms((a['muscle_calcium_mM']-bg['muscle_calcium_mM'])[:,targets])))
    numerical=[]
    for original,refined in [('AS2_pulse','AS2_pulse-halfdt'),('AFD_pulse','AFD_pulse-halfdt'),
                             ('AFD_pulse-halfdt','numerical-quarterstep/AFD_pulse-halfdt')]:
        a,b=traces[original],traces[refined]
        numerical.append(dict(reference=original,refined=refined,
            max_neuron_voltage_difference_mV=float(abs(a['neuron_voltage_mV']-b['neuron_voltage_mV']).max()),
            neuron_voltage_difference_rms_mV=rms(a['neuron_voltage_mV']-b['neuron_voltage_mV']),
            max_muscle_voltage_difference_mV=float(abs(a['muscle_voltage_mV']-b['muscle_voltage_mV']).max()),
            calcium_relative_rms=rms(a['muscle_calcium_mM']-b['muscle_calcium_mM'])/max(1e-30,rms(a['muscle_calcium_mM'])),
            max_body_point_difference_um=float(np.linalg.norm(a['pose'][:,:,:2]-b['pose'][:,:,:2],axis=2).max()*1e6)))
    assert '221 passed, 14 warnings' in (root/'regression.log').read_text()
    summary=dict(complete=True,full302_dynamics_run=True,neurons=302,muscles=95,
        chemical_connections=4665,chemical_neuron_entries=3709,chemical_nmj_entries=956,
        gap_pairs=1192,gap_point_processes=2384,diagonal_gap_entries_omitted=plan['gap_diagonal_entries_omitted'],
        component_pair=pair,records=records,responses=responses,AS2_lesion=lesion,numerical_checks=numerical,
        all_source_and_input_hashes_match=True,project_tests=221,neural_fits=0,
        physical_feedback_injected=False,local_temperature_input=False,
        scope='persistent c302 C1 components with all302 anatomy and feedforward mechanical coupling; not cellwise physiological calibration or thermotaxis',
        unresolved_sign_policy=design['chemistry'],efficacy_policy=design['efficacy'])
    output.mkdir(parents=True,exist_ok=True)
    write(root/'summary.json',summary);write(output/'summary.json',summary)
    time=bg['time_s'];fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name,label,color in [('AS2_pulse','intact','#2166ac'),('AS2_nmj_lesion','6 NMJs off','#b2182b')]:
        z=traces[name]
        axes[0,0].plot(time,z['neuron_voltage_mV'][:,as2],label=label,color=color)
        axes[0,1].plot(time,z['muscle_calcium_mM'][:,targets].mean(1),label=label,color=color)
    axes[0,0].set(ylabel='AS2 voltage (mV)',title='Same injected cell; direct NMJ lesion')
    axes[0,1].set(ylabel='Model calcium (mM)',title='Mean of six directly innervated muscles')
    axes[0,1].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for name in ['AFDL','AIYL','DB1']:
        axes[1,0].plot(time,traces['AFD_pulse']['neuron_voltage_mV'][:,plan['neurons'].index(name)],label=name)
    axes[1,0].set(ylabel='Voltage (mV)',title='Bilateral AFD current pulse in full302')
    for name in ['AFD_pulse','AFD_pulse-halfdt','numerical-quarterstep/AFD_pulse-halfdt']:
        z=traces[name];label={'AFD_pulse':'dt=0.05 ms','AFD_pulse-halfdt':'dt=0.025 ms','numerical-quarterstep/AFD_pulse-halfdt':'dt=0.0125 ms'}[name]
        axes[1,1].plot(time,np.linalg.norm(z['pose'][:,:,:2]-bg['pose'][:,:,:2],axis=2).max(1)*1e6,label=label)
    axes[1,1].set(ylabel='Max body-point displacement (um)',title='Small deformation relative to background')
    for ax in axes.flat:
        ax.axvspan(.2,.6,color='gray',alpha=.1);ax.set_xlabel('Time (s)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    # Isolated AS2 reference uses different timing; these panels show only full-network pulses.
    fig.suptitle('302 neurons + 95 muscles: persistent C1 dynamics, no fitted weights\nCurrent pulses and anatomical NMJ test; mechanical feedback / thermotaxis not connected')
    fig.tight_layout();fig.savefig(output/'full302-dynamics.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(responses=responses,lesion=lesion,numerical=numerical),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/full302-c1-dynamics-2026-11-05'))
    p.add_argument('--output',type=Path,default=Path('docs/reports/assets/full302-c1-dynamics-2026-11-05'))
    a=p.parse_args();main(a.root,a.output)

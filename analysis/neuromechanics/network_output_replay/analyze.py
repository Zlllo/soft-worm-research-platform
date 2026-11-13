"""Independent saved-trajectory analysis of the network-output replay assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run import ROOT, K, dump, sha, verify
from thermotaxis.neuromechanics.anatomical_body import PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_output import curvature


ASSETS=Path('docs/reports/assets/network-output-replay-2026-11-13')


def independent_interpolate(u,source):
    return np.column_stack([np.interp(u,source['time_s'],source['calcium_mM'][:,i]) for i in range(95)])


def continuous_moments(calcium,amplitude,order=32):
    """Ca moments analytic; activation moments by declared Gauss quadrature."""
    endpoints=calcium[400:801]*amplitude
    left,right=endpoints[:-1],endpoints[1:]
    x,w=np.polynomial.legendre.leggauss(order)
    q=(left[None,:,:]*(1-x[:,None,None])+right[None,:,:]*(1+x[:,None,None]))/2
    a=.3*q/(q+K)
    return np.stack([(left+right).mean(0)/2,
                     (left*left+left*right+right*right).mean(0)/3,
                     np.einsum('q,qij->j',w,a)/(2*len(left)),
                     np.einsum('q,qij->j',w,a*a)/(2*len(left))])


def metrics(folder,root):
    spec=json.loads((folder/'spec.json').read_text())
    record=json.loads((folder/'record.json').read_text())
    assert record['complete'] and record['trace_sha256']==sha(folder/'trace.npz')
    assert record['protocol_sha256']==sha(root/'protocol.json')
    z=np.load(folder/'trace.npz');source=np.load(root/'inputs'/f'{spec["mode"]}.npz')
    assert len(z['time_s'])==801
    np.testing.assert_allclose(z['source_time_s'],source['time_s'],atol=2e-12,rtol=0)
    for k in z.files:assert np.isfinite(z[k]).all(),k
    assert (z['internal_time_s']<=z['time_s']+1e-10).all()
    c=independent_interpolate(z['source_time_s'],source)*spec['amplitude']
    a=.3*c/(c+K)
    p=PlanarMuscleProjection.rank_bins(tuple(source['muscle_names']))
    np.testing.assert_allclose(z['calcium_mM'],c,atol=1e-20,rtol=0)
    np.testing.assert_allclose(z['activation'],a,atol=3e-15,rtol=0)
    np.testing.assert_allclose(z['command'],p.to_segments(a),atol=3e-15,rtol=0)
    # Reconstruct every delivered tick by a separate np.interp implementation.
    u=np.arange(round(4*spec['time_scale']/spec['dt_s']),
                round(8*spec['time_scale']/spec['dt_s']))*spec['dt_s']/spec['time_scale']
    dense=independent_interpolate(u,source)*spec['amplitude']
    dense_a=.3*dense/(dense+K)
    moment=np.stack([dense.mean(0),(dense*dense).mean(0),dense_a.mean(0),(dense_a*dense_a).mean(0)])
    np.testing.assert_allclose(z['delivered_tick_moments'],moment,atol=1e-25,rtol=3e-13)
    exact=continuous_moments(source['calcium_mM'],spec['amplitude'])
    low_order=continuous_moments(source['calcium_mM'],spec['amplitude'],16)
    quadrature_difference=float(np.max(abs(low_order-exact)/np.maximum(abs(exact),1e-30)))
    late=z['source_time_s']>=4-1e-12
    t=z['time_s'][late];physical_duration=4*spec['time_scale']
    k=curvature(z['pose'][late])[:,4:-4]*1e-3
    # Time-weighted temporal mean, then separate static and dynamic components.
    mean_k=np.trapezoid(k,t,axis=0)/physical_duration
    variance=np.trapezoid((k-mean_k)**2,t,axis=0)/physical_duration
    center=z['pose'][late,:,:2].mean(1)
    static=float(np.sqrt(np.mean(mean_k**2)))
    dynamic=float(np.sqrt(np.mean(variance)))
    return dict(name=spec['name'],spec=spec,trace_sha256=record['trace_sha256'],
                physical_duration_s=spec['duration_s'],physical_window_s=[float(t[0]),float(t[-1])],
                late_signed_speed_um_s=float(np.trapezoid(z['speed'][late],t)/physical_duration*1e6),
                late_net_displacement_um=float(np.linalg.norm(center[-1]-center[0])*1e6),
                late_path_um=float(np.linalg.norm(np.diff(center,axis=0),axis=1).sum()*1e6),
                dynamic_curvature_rms_per_mm=dynamic,static_curvature_rms_per_mm=static,
                total_curvature_rms_per_mm=float(np.sqrt(static*static+dynamic*dynamic)),
                input_mean_moments=exact.tolist(),delivered_tick_mean_moments=moment.tolist(),
                tick_quadrature_max_relative_error=float(np.max(abs(moment-exact)/np.maximum(abs(exact),1e-30))),
                activation_gauss_16_32_max_relative_difference=quadrature_difference,
                mean_calcium_over_K=float(exact[0].mean()/K),mean_activation=float(exact[2].mean()),
                mean_physical_command=float(z['command'][late].mean()),wall_s=record['wall_s'])


def analyze(root,output):
    verify(root)
    execution=json.loads((root/'execution.json').read_text());assert not execution['failures']
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[metrics(root/'cases'/s['name'],root) for s in protocol['specs']]
    by_name={r['name']:r for r in rows}
    checks=dict(all_ten_complete=len(rows)==10,source_and_program_sha_unchanged=True,
                all_inputs_independently_reconstructed=True,all_finite_and_causal=True,
                activation_quadrature_validated=max(r['activation_gauss_16_32_max_relative_difference'] for r in rows)<1e-10)
    pairs=[]
    for r in rows:
        if 'paired_with' not in r['spec']:continue
        coarse=by_name[r['spec']['paired_with']]
        speed_error=abs(coarse['late_signed_speed_um_s']-r['late_signed_speed_um_s'])
        speed_limit=max(.002,.02*abs(r['late_signed_speed_um_s']))
        curve_error=abs(coarse['dynamic_curvature_rms_per_mm']-r['dynamic_curvature_rms_per_mm'])
        curve_limit=max(.001,.03*r['dynamic_curvature_rms_per_mm'])
        pairs.append(dict(coarse=coarse['name'],fine=r['name'],speed_abs_error_um_s=speed_error,
                          speed_limit_um_s=speed_limit,dynamic_curvature_abs_error_per_mm=curve_error,
                          dynamic_curvature_limit_per_mm=curve_limit,
                          passed=bool(speed_error<=speed_limit and curve_error<=curve_limit)))
    checks['both_half_step_pairs_pass']=all(x['passed'] for x in pairs)
    temporal=[]
    for mode in ['off','live']:
        original=by_name[mode+'_original'];stretched=by_name[mode+'_time_x4']
        assert original['input_mean_moments']==stretched['input_mean_moments']
        temporal.append(dict(mode=mode,continuous_time_average_moments_identical=True,
                             total_input_integral_scale=4.,
                             delivered_original_max_relative_quadrature_error=original['tick_quadrature_max_relative_error'],
                             delivered_stretched_max_relative_quadrature_error=stretched['tick_quadrature_max_relative_error']))
    checks['time_dilation_preserves_continuous_mean_moments']=True
    summary=dict(protocol_sha256=sha(root/'protocol.json'),checks=checks,half_step_pairs=pairs,
                 temporal_input_matching=temporal,cases=rows,
                 interpretation='A linearly resampled open-loop counterfactual from frozen unresolved-timestep neural calcium. Not original closed-loop replication or restored neural locomotion.',
                 source_window_s=[4,8],normalization='All input moments are per unit physical time. Total stretched dose and total time are four times as large.',
                 sample_count='One deterministic trace per condition, not independent biological replicates.')
    output.mkdir(parents=True,exist_ok=True);dump(output/'summary.json',summary)
    print(json.dumps(dict(checks=checks,half_step_pairs=pairs),indent=2))
    return summary


def plot(root,output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'#fbfaf7',
        'axes.facecolor':'#fbfaf7','savefig.facecolor':'#fbfaf7','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    labels=['Original','Calcium ×2','Calcium ×4','Time ×4']
    conditions=['original','calcium_x2','calcium_x4','time_x4']
    for mode,color,offset in [('off','#879dac',-.17),('live','#268b8b',.17)]:
        selected=[next(r for r in summary['cases'] if r['name']==mode+'_'+c) for c in conditions]
        axes[0].bar(np.arange(4)+offset,[r['late_signed_speed_um_s'] for r in selected],.32,color=color,label=mode)
        axes[1].bar(np.arange(4)+offset,[r['dynamic_curvature_rms_per_mm'] for r in selected],.32,color=color,label=mode)
    for ax in axes:
        ax.set_xticks(np.arange(4),labels);ax.axhline(0,color='#aaa49d',lw=.8);ax.grid(axis='y',alpha=.15)
    axes[0].set(title='A  Actual neural-output replay',ylabel='Signed forward speed (µm/s)')
    axes[1].set(title='B  Dynamic body response',ylabel='Curvature RMS (mm$^{-1}$)')
    axes[0].legend(frameon=False)
    fig.suptitle('Increasing calcium and slowing output are different interventions',fontsize=15,weight='bold')
    fig.savefig(output/'replay-comparison.png',dpi=180);fig.savefig(output/'replay-comparison.svg');plt.close(fig)
    # All four live conditions displayed on the same source-time window.
    fig,axes=plt.subplots(2,4,figsize=(14,6.5),constrained_layout=True)
    names=['live_'+x for x in conditions]
    arrays=[np.load(root/'cases'/n/'trace.npz') for n in names]
    maximum=max(float(abs(curvature(z['pose'][400:])*1e-3).max()) for z in arrays)
    for j,(name,z) in enumerate(zip(names,arrays)):
        mean_ca=z['calcium_mM'][400:].mean(1)/K
        axes[0,j].plot(z['source_time_s'][400:],mean_ca,color='#268b8b')
        axes[0,j].set_title(labels[j]);axes[0,j].set_ylim(bottom=0)
        axes[1,j].imshow((curvature(z['pose'][400:])*1e-3).T,aspect='auto',origin='upper',
                         extent=[4,8,1,0],cmap='RdBu_r',vmin=-maximum,vmax=maximum)
        axes[1,j].set_xlabel('Source time (s)')
    axes[0,0].set_ylabel('Mean muscle calcium / K')
    axes[1,0].set_ylabel('Curvature: head → tail')
    fig.suptitle('Frozen live-source calcium replay; time ×4 means 16–32 physical seconds',fontsize=14,weight='bold')
    fig.savefig(output/'replay-calcium-curvature.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ASSETS);parser.add_argument('--plots',action='store_true')
    a=parser.parse_args();s=analyze(a.root,a.output)
    if a.plots:plot(a.root,a.output,s)

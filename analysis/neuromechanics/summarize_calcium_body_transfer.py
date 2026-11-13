"""Recompute calcium/body assay evidence and figures from saved trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from calcium_body_transfer import ROOT, CA_HALF_MM, sha, signal, verify_locks, write_json
from thermotaxis.neuromechanics.anatomical_body import PlanarMuscleProjection
from thermotaxis.neuromechanics.boyle_output import curvature


def harmonic(time_s, values, frequency):
    phase = 2*np.pi*frequency*np.asarray(time_s)
    design = np.column_stack([np.ones(len(phase)), np.cos(phase), np.sin(phase)])
    coefficients = np.linalg.lstsq(design, np.asarray(values).reshape(len(phase), -1), rcond=None)[0]
    return (coefficients[1]-1j*coefficients[2]).reshape(values.shape[1:])


def rms(values):
    return float(np.sqrt(np.mean(np.abs(values)**2)))


def case_metrics(folder, protocol):
    record = json.loads((folder/'record.json').read_text())
    spec = json.loads((folder/'spec.json').read_text())
    assert record['trace_sha256'] == sha(folder/'trace.npz')
    assert record['protocol_sha256'] == sha(folder.parents[1]/'protocol.json')
    z = np.load(folder/'trace.npz')
    t = z['time_s']
    assert len(t) == 2001 and np.allclose(t, np.arange(2001)*.01, atol=1e-12, rtol=0)
    assert all(np.isfinite(z[k]).all() for k in z.files if z[k].dtype.kind in 'fi')
    assert np.all(z['internal_time_s'] <= t+1e-10)
    late = (t >= 8-1e-10) & (t < 20-1e-10)
    inclusive = t >= 8-1e-10
    names = tuple(z['muscle_names'])
    projection = PlanarMuscleProjection.rank_bins(names)
    # Independent reconstruction at every saved sample, no reliance on runner moments.
    ca = np.array([signal(float(x), spec, names)[0] for x in t])
    activation = .3*ca/(ca+CA_HALF_MM)
    np.testing.assert_allclose(z['calcium_mM'], ca, rtol=0, atol=1e-20)
    np.testing.assert_allclose(z['activation'], activation, rtol=0, atol=2e-16)
    np.testing.assert_allclose(z['command'], projection.to_segments(activation), rtol=0, atol=2e-16)
    # Independent finer-than-save-grid input quadrature checks stored full-tick moments.
    dense_t = np.arange(round(8/record['dt_s']), round(20/record['dt_s']))*record['dt_s']
    phase = np.array([(int(n[3:])-.5)/24 for n in names])*2*np.pi*1.5
    if spec['pattern'] == 'synchronous':
        phase[:] = 0
    direction = -1 if spec['pattern'] == 'reverse' else 1
    side = np.array([1 if n[1] == 'D' else -1 for n in names])
    dense_ca = CA_HALF_MM*spec['ratio']*(1+side*np.sin(
        2*np.pi*spec['frequency']*dense_t[:, None]-direction*phase[None, :]))/2
    dense_a = .3*dense_ca/(dense_ca+CA_HALF_MM)
    moments = np.stack([dense_ca.sum(0), (dense_ca*dense_ca).sum(0),
                        dense_a.sum(0), (dense_a*dense_a).sum(0)])*record['dt_s']
    np.testing.assert_allclose(z['input_moments'], moments, rtol=1e-12, atol=1e-26)
    input_diff = z['command'][late, :, 0]-z['command'][late, :, 1]
    muscle_diff = z['muscle'][late, :, 0]-z['muscle'][late, :, 1]
    input_h = harmonic(t[late], input_diff, spec['frequency'])
    muscle_h = harmonic(t[late], muscle_diff, spec['frequency'])
    k = curvature(z['pose'])
    curvature_h = harmonic(t[late], k[late], spec['frequency'])*1e-3  # mm^-1
    local_muscle_h = (muscle_h[:-1]+muscle_h[1:])/2
    selection = slice(4, -4)  # report central body; retain full arrays in trace
    f = spec['frequency']
    meta = json.loads((folder.parents[1]/'build'/spec['variant']/'build.json').read_text())
    b = record['dt_s']/meta['muscle_tau_s']
    expected_h = b*np.exp(-2j*np.pi*f*record['dt_s'])/(1-(1-b)*np.exp(-2j*np.pi*f*record['dt_s']))
    filter_gain = rms(muscle_h)/rms(input_h) if rms(input_h)>1e-12 else None
    center = z['pose'][:, :, :2].mean(1)
    speed = float(np.trapezoid(z['speed'][inclusive], t[inclusive])/12*1e6)
    shape = z['pose'][:, :, :2]-center[:, None, :]
    return dict(name=spec['name'], spec=spec, dt_s=record['dt_s'],
                signed_speed_um_s=speed,
                net_displacement_um=float(np.linalg.norm(center[-1]-center[800])*1e6),
                curvature_harmonic_rms_per_mm=rms(curvature_h[selection]),
                command_harmonic_rms=rms(input_h), muscle_harmonic_rms=rms(muscle_h),
                curvature_per_muscle=rms(curvature_h[selection])/rms(local_muscle_h[selection])
                if rms(local_muscle_h[selection])>1e-12 else None,
                muscle_filter_gain=filter_gain, analytic_discrete_filter_gain=float(abs(expected_h)),
                analytic_continuous_filter_gain=float(1/np.sqrt(1+(2*np.pi*f*meta['muscle_tau_s'])**2)),
                input_moments=z['input_moments'].tolist(),
                max_center_motion_um=float(np.linalg.norm(center-center[0], axis=1).max()*1e6),
                max_shape_change_um=float(np.linalg.norm(shape-shape[0], axis=2).max()*1e6),
                wall_s=record['wall_s'], trace_sha256=record['trace_sha256'])


def summarize(root, output):
    verify_locks(root)
    protocol = json.loads((root/'protocol.json').read_text())
    execution = json.loads((root/'execution.json').read_text())
    assert not execution['failures']
    cases = [case_metrics(root/'cases'/spec['name'], protocol) for spec in protocol['specs']]
    by_name = {x['name']: x for x in cases}
    checks = dict(all_declared_cases_complete=len(cases)==22, inputs_reconstructed=True,
                  source_and_protocol_hashes=True, finite_and_causal=True)
    checks['zero_input_stationary'] = by_name['zero']['max_center_motion_um'] < 1e-6
    errors = {x['name']: abs(x['muscle_filter_gain']-x['analytic_discrete_filter_gain'])
              for x in cases if x['muscle_filter_gain'] is not None}
    checks['all_filters_match_discrete_equation'] = max(errors.values()) < .002
    matching = []
    for ratio in [.25, 1., 4.]:
        names = [f'wave_r{ratio:g}_f{f:g}' for f in [.5, 1., 2.]]
        if ratio == 1:
            names += [x['name'] for x in cases if x['spec']['ratio']==1 and x['spec']['variant']=='normal'
                      and x['spec']['pattern'] != 'travel']
        reference = np.array(by_name[names[0]]['input_moments'])
        error = max(float(np.max(abs(np.array(by_name[name]['input_moments'])-reference)
                                 /np.maximum(abs(reference), 1e-30))) for name in names)
        matching.append(dict(ratio=ratio, names=names, max_relative_moment_error=error))
    checks['matched_calcium_and_activation_moments'] = max(x['max_relative_moment_error'] for x in matching)<1e-10
    refinements = []
    for fine in cases:
        if 'paired_with' not in fine['spec']:
            continue
        coarse = by_name[fine['spec']['paired_with']]
        speed_error = abs(coarse['signed_speed_um_s']-fine['signed_speed_um_s'])
        speed_limit = max(.2, .02*abs(fine['signed_speed_um_s']))
        curvature_error = abs(coarse['curvature_harmonic_rms_per_mm']-fine['curvature_harmonic_rms_per_mm'])/max(
            fine['curvature_harmonic_rms_per_mm'], 1e-9)
        item = dict(coarse=coarse['name'], fine=fine['name'], speed_error_um_s=speed_error,
                    speed_limit_um_s=speed_limit, curvature_relative_error=curvature_error,
                    passed=bool(speed_error <= speed_limit and curvature_error <= .03))
        refinements.append(item)
    checks['all_four_half_step_pairs_pass'] = all(x['passed'] for x in refinements)
    diagnostic = []
    for mode in ['off', 'live']:
        path = Path(f'results/full302-joint-numerics-2026-11-12/{mode}_dt0.00625/trace.npz')
        z = np.load(path)
        late = z['time_s']>=4-1e-9
        r = z['muscle_calcium_mM'][late]/CA_HALF_MM
        diagnostic.append(dict(mode=mode, source=str(path), sha256=sha(path),
                               calcium_over_engineering_half_saturation_quantiles=np.quantile(r, [.05,.5,.95,.99,1]).tolist(),
                               mean_temporal_drive_sd=float(z['drive'][late].std(0).mean()),
                               mean_drive=float(z['drive'][late].mean()),
                               late_signed_speed_um_s=float(z['speed'][late].mean()*1e6),
                               scope='Existing unresolved-timestep full302 snapshot; descriptive reference only. No replay or new neuron run.'))
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(protocol_sha256=sha(root/'protocol.json'), checks=checks, cases=cases,
                   refinement=refinements, matched_input_checks=matching, filter_gain_errors=errors,
                   frozen_network_diagnostics=diagnostic,
                   metric_definition='8<=t<20 harmonic LS, central 39 curvature positions; speed trapezoid 8..20. Input moments on every solver exchange tick.',
                   scope='Open-loop, prescribed model calcium. Not neural autonomous locomotion, thermotaxis, or physiological dose-response.')
    write_json(output/'summary.json', summary)
    print(json.dumps(dict(checks=checks, refinements=refinements), indent=2))
    return summary


def plots(root, output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'figure.facecolor':'#fbfaf7', 'axes.facecolor':'#fbfaf7',
                         'savefig.facecolor':'#fbfaf7', 'axes.titleweight':'bold'})
    cases = {x['name']:x for x in summary['cases']}
    colors = ['#86a8b9', '#268b8b', '#dd944b']
    fig, axes = plt.subplots(1, 3, figsize=(15,4.7), constrained_layout=True)
    for ratio, color in zip([.25,1.,4.], colors):
        rows = [cases[f'wave_r{ratio:g}_f{f:g}'] for f in [.5,1.,2.]]
        axes[0].plot([.5,1,2], [x['signed_speed_um_s'] for x in rows], 'o-', color=color,
                     lw=2.3, label=f'Calcium peak / K = {ratio:g}')
        axes[1].plot([.5,1,2], [x['curvature_harmonic_rms_per_mm'] for x in rows], 'o-', color=color, lw=2.3)
    axes[0].set(xlabel='Drive frequency (Hz)', ylabel='Forward speed (µm/s)', title='A  Strength and rhythm')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].set(xlabel='Drive frequency (Hz)', ylabel='Curvature amplitude (mm$^{-1}$)', title='B  The body response')
    variants = [('normal','Standard muscle / drag','#268b8b'),
                ('fast_muscle','Faster muscle filter','#d37a50'), ('heavy_drag','4× environmental drag','#6c6ca4')]
    for variant,label,color in variants:
        names = [f'wave_r1_f{f:g}' if variant=='normal' else f'{variant}_r1_f{f:g}' for f in [.5,2.]]
        axes[2].plot([.5,2], [cases[n]['signed_speed_um_s'] for n in names], 'o-', color=color, lw=2.3, label=label)
    axes[2].set(xlabel='Drive frequency (Hz)', ylabel='Forward speed (µm/s)', title='C  Separate two physical effects')
    axes[2].legend(frameon=False,fontsize=9)
    for ax in axes:
        ax.axhline(0,color='#aaa49d',lw=.8)
        ax.grid(alpha=.15)
        ax.set_xticks([.5,1,2])
    fig.suptitle('The same calcium statistics can produce different movement', fontsize=16, weight='bold')
    fig.savefig(output/'transfer-summary.png',dpi=180)
    fig.savefig(output/'transfer-summary.svg')
    plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(14,7.5),constrained_layout=True)
    conditions = [('wave_r1_f0.5','0.5 Hz travelling wave'), ('wave_r1_f2','2 Hz travelling wave'),
                  ('synchronous_r1_f2','2 Hz synchronous contraction')]
    vmax = max(abs(curvature(np.load(root/'cases'/n/'trace.npz')['pose'])) .max()*1e-3 for n,_ in conditions)
    for j,(name,title) in enumerate(conditions):
        z=np.load(root/'cases'/name/'trace.npz'); t=z['time_s']; mask=(t>=16)&(t<=20)
        d=z['muscle'][mask,:,0]-z['muscle'][mask,:,1]
        axes[0,j].imshow(d.T,aspect='auto',origin='upper',extent=[16,20,1,0],cmap='RdBu_r',vmin=-.1,vmax=.1)
        im=axes[1,j].imshow((curvature(z['pose'][mask])*1e-3).T,aspect='auto',origin='upper',
                           extent=[16,20,1,0],cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes[0,j].set_title(title+'\n'+f"{cases[name]['signed_speed_um_s']:.2f} µm/s")
        axes[1,j].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Muscle activation\nHead → tail')
    axes[1,0].set_ylabel('Body curvature\nHead → tail')
    fig.colorbar(im,ax=axes[1,:],label='Curvature (mm$^{-1}$)',shrink=.7)
    fig.suptitle('Muscle activity is transmitted through body mechanics',fontsize=16,weight='bold')
    fig.savefig(output/'muscle-curvature.png',dpi=160);plt.close(fig)
    # Measured/analytic filter response separates a built-in filter from mechanics.
    fig, axes=plt.subplots(1,2,figsize=(10,4.2),constrained_layout=True)
    for tau,label,color,variant in [(.1,'100 ms muscle filter','#268b8b','normal'),(.025,'25 ms muscle filter','#d37a50','fast_muscle')]:
        freq=np.linspace(.25,2.3,200)
        axes[0].plot(freq,1/np.sqrt(1+(2*np.pi*freq*tau)**2),color=color,label=label)
        fs=[.5,1,2] if variant=='normal' else [.5,2]
        names=[f'wave_r1_f{f:g}' if variant=='normal' else f'{variant}_r1_f{f:g}' for f in fs]
        axes[0].scatter(fs,[cases[n]['muscle_filter_gain'] for n in names],color=color,s=40)
        axes[1].plot(fs,[cases[n]['curvature_per_muscle'] for n in names],'o-',color=color,label=label)
    axes[0].set(xlabel='Frequency (Hz)',ylabel='Muscle / command amplitude',title='A  Known filter equation')
    axes[1].set(xlabel='Frequency (Hz)',ylabel='Curvature / muscle amplitude',title='B  Remaining mechanical response')
    axes[0].legend(frameon=False,fontsize=9)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(output/'filter-and-mechanics.png',dpi=180);plt.close(fig)
    # A fixed matched-input comparison, not the best-looking case selected after results.
    animated=['wave_r1_f0.5','wave_r1_f2','heavy_drag_r1_f2']
    traces=[np.load(root/'cases'/name/'trace.npz') for name in animated]
    fig,axes=plt.subplots(3,1,figsize=(8,6),constrained_layout=True)
    lines=[]
    for ax,name,z in zip(axes,animated,traces):
        xy=z['pose'][:,:,:2]*1e3
        ax.set_xlim(float(xy[...,0].min())-.1,float(xy[...,0].max())+.1)
        ax.set_ylim(float(xy[...,1].min())-.08,float(xy[...,1].max())+.08)
        ax.set_aspect('equal',adjustable='box');ax.set_ylabel('y (mm)')
        ax.set_title(name.replace('_',' ')+f"  |  {cases[name]['signed_speed_um_s']:.2f} µm/s",fontsize=10)
        line,=ax.plot([],[],lw=3,color='#268b8b');head,=ax.plot([],[],'o',color='#d37a50',ms=5)
        lines.append((line,head))
    axes[-1].set_xlabel('x (mm)'); clock=fig.suptitle('Prescribed calcium → muscle filter → physical body')
    def update(index):
        for z,(line,head) in zip(traces,lines):
            xy=z['pose'][index,:,:2]*1e3;line.set_data(xy[:,0],xy[:,1]);head.set_data(xy[:1,0],xy[:1,1])
        clock.set_text(f'Prescribed calcium → physical movement   t={index*.01:.2f} s')
    animation=FuncAnimation(fig,update,frames=range(800,2001,10),interval=80)
    animation.save(output/'matched-input-motion.gif',writer=PillowWriter(fps=12),dpi=90)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=Path('docs/reports/assets/calcium-body-transfer-2026-11-13'))
    parser.add_argument('--plots',action='store_true')
    args=parser.parse_args();result=summarize(args.root,args.output)
    if args.plots:plots(args.root,args.output,result)

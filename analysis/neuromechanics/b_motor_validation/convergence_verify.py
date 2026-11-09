"""Independent raw-data audit: NumPy only, does not import the analysis code.

Prominence is computed by searching for the nearest strictly higher sample on
each side and taking the higher of the two interval minima. This independently
checks the SciPy peak frequency estimates used by the primary analysis.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]/'results/b-motor-convergence-2026-11-09'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def near(a,b): np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
def energy(x): return float(np.sqrt(np.sum(x*x)/x.size))


def peaks(x):
    candidates=np.where((x[1:-1]>x[:-2])&(x[1:-1]>x[2:]))[0]+1
    accepted=[]
    for i in candidates:
        higher_left=np.where(x[:i]>x[i])[0]
        higher_right=np.where(x[i+1:]>x[i])[0]
        left=int(higher_left[-1]) if higher_left.size else 0
        right=int(i+1+higher_right[0]) if higher_right.size else len(x)-1
        prominence=x[i]-max(x[left:i+1].min(),x[i:right+1].min())
        if prominence>=3:accepted.append(int(i))
    return np.array(accepted,dtype=int)


def main():
    s=read(ROOT/'summary.json');p=read(ROOT/'protocol.json')['acceptance'];raw={};own={}
    muscles=read(ROOT/'connection-plan.json')['muscles']
    assert s['protocol_sha256']==digest(ROOT/'protocol.json')==read(ROOT/'protocol-lock.json')['protocol_sha256']
    for name,d in read(ROOT/'inputs.json').items(): assert digest(ROOT/name)==d['sha256']
    for name,m in s['metrics'].items():
        z=dict(np.load(ROOT/name/'trace.npz',allow_pickle=False));raw[name]=z
        assert digest(ROOT/name/'trace.npz')==m['trace_sha256']
        late=np.arange(400,801);v=z['neuron_voltage_mV'][late]
        ix=[z['neuron_names'].tolist().index(n) for n in z['B_names']];v=v[:,ix]
        pp=[peaks(v[:,j]) for j in range(18)]
        freq=[1/np.mean(np.diff(z['time_s'][late][a])) for a in pp]
        near(freq,m['B_frequency_Hz']);assert [len(a) for a in pp]==m['B_peak_counts']
        near(np.std(v,axis=0),m['B_std_mV']);near(v.mean(0),m['B_mean_mV'])
        amp=np.quantile(v,.95,axis=0)-np.quantile(v,.05,axis=0);near(amp,m['B_amplitude_mV'])
        ca=z['muscle_calcium_mM'][late];near(ca.mean(0),m['calcium_mean_mM']);near(ca.std(0),m['calcium_std_mM'])
        calcium=np.maximum(z['muscle_calcium_mM'],0)
        muscle_drive=.3*calcium/(calcium+2.5e-7)
        expected_drive=np.zeros((801,48,2))
        for j,muscle in enumerate(muscles):
            side=0 if muscle[1]=='D' else 1
            first=2*(int(muscle[3:])-1)
            expected_drive[:,first:first+2,side]+=.5*muscle_drive[:,j,None]
        near(expected_drive,z['drive'])
        near(z['drive'][late].std(0).mean(),m['drive_mean_channel_std'])
        near(z['speed'][late].mean()*1e6,m['late_speed_um_s'])
        centroid=np.mean(z['pose'][:,:,:2],axis=1)
        displacement=np.linalg.norm(centroid[-1]-centroid[400])*1e6;near(displacement,m['late_displacement_um'])
        # Independent intrinsic shape via head-tail angle and trigonometric rotation.
        angle=np.arctan2(z['pose'][:,0,1]-z['pose'][:,-1,1],z['pose'][:,0,0]-z['pose'][:,-1,0])
        centered=z['pose'][:,:,:2]-centroid[:,None,:]
        yy=(-centered[:,:,0]*np.sin(angle[:,None])+centered[:,:,1]*np.cos(angle[:,None]))*1e6
        amplitude=(yy[late].max(0)-yy[late].min(0)).max();near(amplitude,m['body_frame_lateral_amplitude_um'])
        strain=z['side_length']/z['side_length'][0]-1
        own[name]=dict(v=v,frequency=np.array(freq),amplitude=amp,yy=yy,strain=strain,displacement=displacement)
    checked=0
    for key,c in s['comparisons'].items():
        na=f"{c['feedback']}_dt{c['coarse_dt_ms']:g}";nb=f"{c['feedback']}_dt{c['reference_dt_ms']:g}"
        a=raw[na];b=raw[nb];x=own[na];y=own[nb];calc={}
        calc['B_frequency']=max(abs(x['frequency']/y['frequency']-1))
        calc['B_amplitude']=max(abs(x['amplitude']/y['amplitude']-1))
        calc['B_mean']=max(abs(x['v'].mean(0)-y['v'].mean(0)))
        calc['B_trajectory']=max(np.sqrt(np.mean((x['v']-y['v'])**2,axis=0)))
        ix=[a['neuron_names'].tolist().index(n) for n in a['B_names']]
        max_offsets=[]
        nominal_time=np.arange(350,801)/100
        for j in ix:
            ta=nominal_time[peaks(a['neuron_voltage_mV'][350:,j])]
            tb=nominal_time[peaks(b['neuron_voltage_mV'][350:,j])]
            distances=[min(abs(q-tb)) for q in ta if q>=4-1e-8]+[min(abs(q-ta)) for q in tb if q>=4-1e-8]
            max_offsets.append(max(distances))
        calc['B_peak_phase']=max(max_offsets)
        for field,label,floor in [('muscle_calcium_mM','calcium',p['calcium_scale_floor_mM']),('drive','drive',p['drive_scale_floor'])]:
            aa=a[field][400:].reshape(401,-1);bb=b[field][400:].reshape(401,-1)
            denominator=np.maximum(np.quantile(bb,.95,axis=0)-np.quantile(bb,.05,axis=0),floor)
            stats=[abs(aa.mean(0)-bb.mean(0)),abs(aa.std(0)-bb.std(0)),abs(np.quantile(aa,.05,axis=0)-np.quantile(bb,.05,axis=0)),abs(np.quantile(aa,.95,axis=0)-np.quantile(bb,.95,axis=0))]
            calc[label+'_statistics']=float(np.max(np.array(stats)/denominator))
            calc[label+'_trajectory']=float(max(np.sqrt(np.mean((aa-bb)**2,axis=0))/denominator))
        calc['body_shape_trajectory']=energy(x['yy'][400:]-y['yy'][400:])
        calc['body_shape_amplitude']=abs(s['metrics'][na]['body_frame_lateral_amplitude_um']-s['metrics'][nb]['body_frame_lateral_amplitude_um'])
        calc['side_strain_trajectory']=energy(x['strain'][400:]-y['strain'][400:])
        calc['mean_speed']=abs(a['speed'][400:].mean()-b['speed'][400:].mean())*1e6
        calc['speed_trajectory']=energy(a['speed'][400:]-b['speed'][400:])*1e6
        calc['centroid_displacement']=abs(x['displacement']-y['displacement'])
        calc['feedback_trajectory']=energy(a['feedback_pA'][4000:]-b['feedback_pA'][4000:])
        limits=dict(B_frequency=p['B_frequency_max_relative'],B_amplitude=p['B_q95_q05_amplitude_max_relative'],
            B_mean=p['B_mean_max_abs_mV'],B_trajectory=p['B_unshifted_RMS_max_mV'],B_peak_phase=p['B_peak_time_max_abs_s'],
            calcium_statistics=p['calcium_stats_and_trace_scaled_error_max'],calcium_trajectory=p['calcium_stats_and_trace_scaled_error_max'],
            drive_statistics=p['drive_stats_and_trace_scaled_error_max'],drive_trajectory=p['drive_stats_and_trace_scaled_error_max'],
            body_shape_trajectory=p['body_shape_RMS_max_um'],
            body_shape_amplitude=max(p['body_lateral_amplitude_abs_um'],p['body_lateral_amplitude_relative']*s['metrics'][nb]['body_frame_lateral_amplitude_um']),
            side_strain_trajectory=max(p['side_strain_RMS_abs'],p['side_strain_RMS_relative']*energy(y['strain'][400:])),
            mean_speed=p['speed_mean_abs_difference_um_s'],speed_trajectory=p['speed_trace_RMS_um_s'],
            centroid_displacement=p['centroid_displacement_abs_difference_um'],
            feedback_trajectory=max(p['feedback_RMS_abs_pA'],p['feedback_RMS_relative']*energy(b['feedback_pA'][4000:])))
        for gate,value in calc.items():
            near(value,c['errors'][gate]['value'])
            near(limits[gate],c['errors'][gate]['limit'])
            assert bool(value<=c['errors'][gate]['limit']+1e-12)==c['gates'][gate],(key,gate)
            checked+=1
        assert c['passed']==all(c['gates'].values())
    assert len(raw)<=6
    if len(raw)>=4:
        required=any(not s['comparisons'][f'{mode}_0.025_vs_0.0125']['passed'] for mode in ['off','live'])
        assert required==s['fine_required']
        if len(raw)==6: assert required, 'Refinement was not authorized by the declared rule'
        step=s['latest_pair_ms']
        passed=all(s['comparisons'][f'{mode}_{step[0]:g}_vs_{step[1]:g}']['passed'] for mode in ['off','live'])
        assert s['neural_step_convergence_accepted']==bool(passed and (len(raw)==6 or not required))
    result=dict(independent_numpy_audit_passed=True,runs_checked=len(raw),pair_gate_values_checked=checked,
        independently_implemented=['Prominent peaks without SciPy','Frequencies and amplitudes','Calcium/drive channel errors',
            'Calcium to 95-muscle / 48x2 planar drive mapping','Body-frame geometry by rotation angle','Speed/displacement/strain/feedback error magnitudes'],
        summary_sha256=digest(ROOT/'summary.json'),analyzer_imported=False)
    (ROOT/'independent-validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()

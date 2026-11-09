"""Static-only audit of frozen generated state updates; no NEURON or simulation."""
import hashlib
import json
import math
from pathlib import Path
import re

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/b-motor-convergence-2026-11-09'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Neither regenerate the formal report/archive nor write any existing input.
    manifest=json.loads((ROOT/'inputs.json').read_text())
    sources={}
    def evidence(relative,first,last):
        path=ROOT/relative
        assert sha(path)==manifest[relative]['sha256'],relative
        lines=path.read_text().splitlines()
        key=f'{relative}:{first}-{last}'
        sources[key]=dict(sha256=sha(path),lines=[f'{i+1}: {lines[i]}' for i in range(first-1,last)])
    for relative,first,last in [
        ('kernels/ca_boyle.mod',81,100),('kernels/ca_boyle.mod',187,243),
        ('kernels/k_fast.mod',71,85),('kernels/k_fast.mod',163,211),
        ('kernels/k_slow.mod',59,69),('kernels/k_slow.mod',135,160),
        ('kernels/CaPool.mod',45,53),('kernels/CaPool.mod',85,109),
        ('kernels/arm64/ca_boyle.cpp',524,562),('kernels/arm64/ca_boyle.cpp',696,709),
        ('kernels/arm64/k_fast.cpp',453,484),('kernels/arm64/k_fast.cpp',612,625),
        ('kernels/arm64/k_slow.cpp',377,398),('kernels/arm64/k_slow.cpp',520,533),
        ('kernels/arm64/CaPool.cpp',361,380),('kernels/arm64/CaPool.cpp',558,573),
        ('kernels/neuron_to_neuron_exc_syn.mod',90,108),
        ('kernels/arm64/neuron_to_neuron_exc_syn.cpp',362,385),
        ('kernels/arm64/neuron_to_neuron_inh_syn.cpp',362,385),
        ('kernels/GenericMuscleCell.hoc',152,178),('kernels/GenericNeuronCell.hoc',153,179),
        ('runtime/driver.py',29,38),('runtime/driver.py',99,120),
        ('runtime/src/thermotaxis/neuromechanics/full302_c1.py',97,100)]:
        evidence(relative,first,last)
    updates={}
    for mechanism,states in [('ca_boyle',['e_q','f_q']),('k_fast',['p_q','q_q']),('k_slow',['n_q']),('CaPool',['concentration'])]:
        text=(ROOT/f'kernels/arm64/{mechanism}.cpp').read_text()
        start=text.index('static int states (')
        block=text[start:text.index('static int  rates',start)]
        assert 'rates (' in block and 'exp(' not in block and 'exp (' not in block
        for state in states:
            rate='rate_concentration' if state=='concentration' else 'rate_'+state
            expected=f'{state} = {state} - dt*(- ( {rate} ) ) ;'
            assert expected in block,(mechanism,state)
            updates[f'{mechanism}.{state}']=dict(classification='explicit Euler at fixed-step state update',generated_statement=expected)
    for mechanism in ['neuron_to_neuron_exc_syn','neuron_to_neuron_inh_syn']:
        text=(ROOT/f'kernels/arm64/{mechanism}.cpp').read_text()
        start=text.index('static int states (');block=text[start:text.index('static int  rates',start)]
        assert 's = s + (1. - exp(' in block and '( 1.0 - inf ) > 1e-4' in block
    constants={}
    for mechanism,parameters in [('ca_boyle',['e_timeCourse_tau','f_timeCourse_tau']),
        ('k_fast',['p_timeCourse_tau','q_timeCourse_tau']),('k_slow',['n_timeCourse_tau']),('CaPool',['decayConstant'])]:
        text=(ROOT/f'kernels/{mechanism}.mod').read_text()
        for parameter in parameters:
            tau=float(re.search(r'\b'+parameter+r'\s*=\s*([0-9.Ee+-]+)\s*\(ms\)',text).group(1))
            constants[f'{mechanism}.{parameter}']=dict(tau_ms=tau,
                dt_over_tau={str(dt):dt/tau for dt in [.025,.0125,.00625]})
    tau=constants['ca_boyle.e_timeCourse_tau']['tau_ms']
    algebra=[]
    for dt in [.025,.0125,.00625]:
        r=dt/tau;exact=-math.expm1(-r)
        algebra.append(dict(dt_ms=dt,dt_over_tau=r,Euler_increment_fraction=r,
            fixed_coefficient_exponential_increment_fraction=exact,
            increment_relative_difference=(r-exact)/exact))
    archive=PROJECT/'docs/reports/assets/b-motor-convergence-2026-11-09/reproduction.zip'
    record=json.loads((ROOT/'archive-validation.json').read_text())
    assert sha(archive)==record['archive_sha256']
    result=dict(scope='Read-only static inspection and closed-form scalar arithmetic; no simulation, compilation or kernel edits',
        generator_runtime_version='NEURON 9.0.2, pinned convergence provenance',
        evidence=sources,state_updates=updates,time_constants=constants,fast_Ca_gate_single_step_algebra=algebra,
        chemistry_exponential_branch_confirmed=['exc','inh'],
        original_archive_sha256=sha(archive),original_report_sha256=record['report_sha256'],
        original_summary_sha256=record['summary_sha256'],
        cause_of_full_system_nonconvergence_proven=False,new_full_runs=0,
        audit_script_sha256=sha(Path(__file__)))
    target=ROOT/'numerical-diagnostic-evidence.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(static_audit_passed=True,Euler_state_variables=len(updates),evidence_blocks=len(sources),
        archive_unchanged=True,new_runs=0,fast_gate_algebra=algebra),ensure_ascii=False))


if __name__=='__main__':main()

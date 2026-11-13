"""Read-only review of A's locked 2x2 proposal; no build/run authorization."""
import json
from pathlib import Path
import shutil
import numpy as np
from timing_review_static import ROOT, REPO, OLD, A, REPORT, sha, write, block

REPLACEMENTS={
    'ca_boyle.mod':('fopen0 = e_fcond * f_fcond * h_fcond','fopen0 = (e_q ^ e_instances) * (f_q ^ f_instances) * h_fcond'),
    'k_fast.mod':('fopen0 = p_fcond * q_fcond','fopen0 = (p_q ^ p_instances) * (q_q ^ q_instances)'),
    'k_slow.mod':('fopen0 = n_fcond','fopen0 = n_q ^ n_instances')}

def read(p):return json.loads(p.read_text())
def main():
    protocol_bytes=(A/'protocol.json').read_bytes();p=json.loads(protocol_bytes);digest=sha(A/'protocol.json')
    assert digest==read(A/'protocol-lock.json')['protocol_sha256']
    assert sha(A/'sources.json')==p['source_manifest_sha256']
    assert sha(A/'reuse-audit.json')==p['reuse']['audit_sha256']
    assert sha(REPO/'analysis/neuromechanics/b_motor_validation/timing_prepare.py')==p['prepare_script_sha256']
    sources=read(A/'sources.json')
    for rel,d in sources.items():assert sha(A/rel)==d['sha256'],rel
    proposals=read(A/'proposal-source-hashes.json')
    for rel,h in proposals.items():assert sha(A/rel)==h,rel
    inherited=read(OLD/'protocol.json');assert sha(OLD/'protocol.json')==p['main']['inherited_protocol_sha256']
    assert p['acceptance']==inherited['acceptance']
    assert p['main']['input_convention']==inherited['main']['input_convention']
    assert p['main']['new_conditions']==6 and p['main']['reused_conditions']==6
    assert p['main']['dt_ms']==inherited['dt_ms'] and p['main']['duration_ms']==8000
    assert p['main']['window_ms']==[4000,8000] and p['main']['record_ms']==.025
    patches={}
    for parent,variant in [('original','euler_direct'),('both','exp_direct')]:
        changes=[]
        for original in sorted((A/'frozen'/parent).iterdir()):
            if original.suffix not in ['.mod','.hoc']:continue
            old=original.read_text();actual=(A/'proposed-variants'/variant/original.name).read_text();expected=old
            if original.name in REPLACEMENTS:
                before,after=REPLACEMENTS[original.name];assert old.count(before)==1
                expected=old.replace(before,after)
            assert actual==expected,(variant,original.name)
            if old!=actual:changes.append(original.name)
            if original.name in REPLACEMENTS:
                for label in ['NEURON','UNITS','PARAMETER','STATE','INITIAL','DERIVATIVE states','PROCEDURE rates()']:
                    assert block(actual,label.replace('(',r'\(').replace(')',r'\)')+r'\s*\{')==block(old,label.replace('(',r'\(').replace(')',r'\)')+r'\s*\{')
                if original.name=='ca_boyle.mod':
                    assert '* h_fcond' in actual and '(f_q ^ f_instances) * h_q' not in actual
        assert changes==['ca_boyle.mod','k_fast.mod','k_slow.mod']
        patches[variant]=dict(source_changes_exactly_three_declared_fopen_replacements=True,
            parameters_states_rates_initial_sync_unchanged=True,h_fcond_retained=True,changed_files=changes)
    assert sha(A/'presynaptic-input.npz')==p['main']['input_sha256']
    with np.load(A/'presynaptic-input.npz') as z:source=z['voltage_mV'].copy()
    audit=read(A/'reuse-audit.json');reused={}
    for name,d in audit.items():
        folder=REPO/d['source'];record=read(folder/'record.json')
        assert sha(folder/'trace.npz')==d['trace_sha256']==record['trace_sha256']
        assert sha(folder/'spec.json')==d['spec_sha256'] and sha(folder/'record.json')==d['record_sha256']
        assert record['complete'] and record['versions']['neuron']=='9.0.2'
        assert record['spec']['runner_sha256']==sha(A/'legacy-code/integration_run.py')
        dt=record['spec']['dt_ms'];assert dt==record['actual_dt_ms']
        with np.load(folder/'trace.npz') as z:
            cols=z['columns'].tolist();data=z['data'];assert data.shape==(320001,32) and np.isfinite(data).all()
            c=lambda key:data[:,cols.index(key)]
            assert np.array_equal(c('time_ms'),np.arange(320001)*.025)
            assert np.array_equal(c('peer_at_record_mV'),source[::4])
            ix=np.maximum(np.arange(320001)*4-round(dt/.00625),0)
            assert np.array_equal(c('last_applied_peer_mV'),source[ix])
        reused[name]=dict(trace_sha256=d['trace_sha256'],source_record_sha256=d['record_sha256'])
    assert len(reused)==6
    runtime=read(A/'current-runtime-defaults.json')
    assert runtime['neuron_version']=='9.0.2' and runtime['secondorder']==0
    assert runtime['cells_constructed']==runtime['finitialize_calls']==runtime['fadvance_calls']==0
    budget=p['budgets']['small_controls'];allowed=16;issues=[]
    if budget>allowed:
        issues.append(dict(id='P1',severity='requires_protocol_resolution',field='controls.conditions / budgets.small_controls',
            proposed=budget,mainline_current_cap=allowed,
            message='Locked A proposal exceeds the current mainline short-control cap16; resolve before execution.',
            possible_within_cap='8 zero-step tests (4variants x2 voltages) plus4 K-fast short tests (one integrator x2 readouts x observed/plain arms); verify other-integrator current functions are identical and keep all four STATE algorithms checked statically. An alternative8 short tests at one dt plus4 zero-step tests also totals12 if explicitly re-locked.'))
    pending=['compiled direct/observer executable C++ bodies not yet available',
        'passive-observer physical noninterference and callback placement not yet measured',
        'actual same-dt callback hypothesis discrimination and fixed-voltage control data not yet available',
        'six new main outcomes and factorial interaction not yet available']
    notes=[
        'The all-current voltage-clamp/metadata-pair design can identify instantaneous readout dependence using measured callback V/h, if noninterference and formula-separation gates pass; it is a valid alternative to the minimal K-only diagnostic in principle.',
        'h_q is ASSIGNED and h_fcond remains unchanged; Ca current may still retain h(cai) and pool scheduling differences.',
        'Main recordings do not promise before-last-substep V/cache/h snapshots. Outcome reuse is valid, but main after-step values alone cannot reconstruct earlier current calls; classify main current columns as endpoint observations unless the missing passive pre-snapshots are specified.',
        'Neither initial-phase windows nor startup exclusion identifies what fraction of late readout effects came from startup; report total factor effects and limit attribution.',
        'Old missing secondorder is accepted for reuse as explicitly disclosed same-version fresh-process/default inference, not a directly measured historical field.',
        'Main extra fcurrent/rates/hooks forbidden; terminal fcurrent is isolated after the last control step and never followed by continuation. Native finitialize internal current setup is retained, not an extra observer call.',
        'Generated observer code has to preserve all physical arithmetic and state updates. Last-callback metadata is not a complete call trace or time-average.',
        'No full system or biological-delay conclusion is authorized by this component design.']
    notes.append('Mainline explicitly accepts16 short5ms controls replacing, not adding to, the earlier tentative12-control plan; two dt values each receive observed/plain noninterference pairs for all4 kernels.')
    out=read(ROOT/'static-review.json')
    result=dict(status='reviewed_with_findings' if issues else 'complete',reviewed_protocol_sha256=digest,
        protocol_status=p['status'],scientific_execution_authorized_by_this_review=False,
        source_only_proposal_checks_passed=True,proposed_patch_review=patches,
        historical_reuse=dict(decision='accepted_with_documented_default_inference',historical_secondorder_directly_measured=False,
            current_same_version_secondorder=0,verified_traces=reused),
        budget=dict(new_main=6,reused_main=6,full302=0,body=0,small_controls_proposed=budget,current_mainline_cap=allowed,within_current_cap=budget<=allowed,
            decision='explicit mainline acceptance of16 replacing earlier tentative12; not an added control set'),
        causal_design_adequate_in_principle=True,ready_for_scientific_execution=False,
        protocol_acceptable_for_implementation=not issues,main_execution_ready=False,
        next_gate='Inspect generated physical/observer functions before paired short controls; require measured observer noninterference before main. This review itself launches or authorizes no run.',
        decision='acceptable_for_implementation_and_gated_controls' if not issues else 'requires_protocol_resolution',
        pending_generated_code_and_data=pending,notes=notes,issues=issues)
    out.update(status='protocol_review_complete',A_protocol_available=True,protocol_review=result,issues=issues)
    write(ROOT/'review.json',out);write(ROOT/'protocol-review.json',result);write(ROOT/'protocol_review.json',result)
    snap=ROOT/'A-protocol-snapshot';snap.mkdir(exist_ok=True)
    for name in ['protocol.json','protocol-lock.json','reuse-audit.json','current-runtime-defaults.json','proposal-source-hashes.json']:
        shutil.copy2(A/name,snap/name)
    shutil.copytree(A/'patches',snap/'patches',dirs_exist_ok=True)
    report=REPORT.read_text().split('\n## A 锁定协议复核')[0]
    appendix=['', '## A 锁定协议复核', '',f'已审阅协议 SHA：`{digest}`。本节是提议源码与协议复核，没有编译、导入 NEURON 或运行实验。', '',
        '两种直接版本确实只替换三处 fopen0 表达式，恰好对应五个真 STATE 的幂；h_fcond 保留。全部参数、初始化、导数、rates、CaPool 和 NMJ 未改。六条旧轨迹的 SHA、记录形状、时钟、完整输入索引均独立核对，建议按已披露的默认值推断接受复用；旧 secondorder 仍不是实测历史字段。', '',
        'A 的回调元数据/无观察器配对方案在原理上足以作为另一种时序诊断：必须使用同次回调 V/STATE/缓存/h，并先证明观察器不改变物理轨迹。不能把其最后一次回调快照称为完整调用序列。实际生成观察器与新结果尚未保存，本次没有提前验收。', '',
        f'**主线已明确接受 {budget} 个5ms短控制的上限。** 四版本×两个dt×观察器有/无配对，取代此前暂定12个控制的设计，不是叠加。这个增加用于在两个dt分别证明观察器不改变物理轨迹，理由充分。提议源码与协议层面可以进入实现和分阶段检查；尚不能据此宣布未生成的代码或控制已通过，也不授权跳过这些检查直接运行主条件。', '',
        '主条件保持六条新运行、零完整网络/身体。主数据新增被动前一步 V/cache/h 快照仍是建议项；若不增加，逐时电流因果解释应依赖短控制的回调证据，主端点电流只能按端点观察列报告。主结果必须保留初始化差异，并报告同dt两因子对照和交互，不从晚期窗口推断启动效应已经消失。', '',
        '机器可读结论在 `review.json.protocol_review`、`protocol_review.json`（及同内容 protocol-review.json）：协议及 MOD 提议审查通过，复用可按明确记录的默认值推断接受；生成代码、观察器无扰动、控制判别及主结果均未验证。本静态阶段到此结束，无后台轮询或运行进程。']
    REPORT.write_text(report+'\n'.join(appendix)+'\n')
    print(json.dumps(dict(protocol_review=result['status'],protocol_sha256=digest,issues=issues,new_simulations=0),ensure_ascii=False))

if __name__=='__main__':main()

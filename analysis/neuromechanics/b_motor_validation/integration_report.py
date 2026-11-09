"""Write the isolation handoff and archive; no scientific simulation is run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

PROJECT=Path(__file__).resolve().parents[3]
ROOT=PROJECT/'results/c1-muscle-integration-isolation-2026-11-09'
REPORT=PROJECT/'docs/reports/c1-muscle-integration-isolation-2026-11-09.md'
ASSETS=REPORT.parent/'assets/c1-muscle-integration-isolation-2026-11-09'
SCRIPTS=['integration_prepare.py','integration_run.py','integration_batch.py','integration_analyze.py','integration_verify.py','integration_report.py']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def f(x):return f'{x:.6g}'


def main():
    s=read(ROOT/'summary.json');audit=read(ROOT/'independent-validation.json');p=read(ROOT/'protocol.json')
    assert audit['independent_audit_passed'] and audit['summary_sha256']==sha(ROOT/'summary.json')
    lines=['# C1肌肉积分隔离：局部解析正确，但未显示总体步长差改善', '', '日期：2026-11-09；仅孤立组件，完整302/身体新增运行数为0。', '',
        '**结论：直接导数补丁生成了预期的指数更新，修复版的门控及CaPool STATE通过解析对照；同输入肌肉响应没有满足预先声明的改善条件，因此本阶段不建议直接启动下一轮完整系统预算。**', '',
        '四版本在最细两档间均通过本阶段孤立肌肉容差，原版也通过；合并版的电位、钙及驱动逐时差反而约为原版的2.1倍。前一阶段完整系统的未收敛结果仍封存，不能由本组件测试改记为通过。', '',
        '## 冻结协议与最小补丁', '',
        f"协议在编译和科学运行前锁定，SHA-256：`{s['protocol_sha256']}`。没有改验收标准或追加主条件。四版本为original原版、gates仅5门控、pool仅浓度池、both合并；原几何、参数、初始化、NMJ、rate观测变量、缓存、电流和离子同步/截断顺序全部保留。", '',
        '| 版本 | Ca e/f、快K p/q、慢K n | CaPool浓度 |', '|---|---|---|',
        '| original | 原生成Euler | 原生成Euler |', '| gates | 直接导数，生成指数更新 | 原生成Euler |',
        '| pool | 原生成Euler | 直接导数，生成指数更新 |', '| both | 直接导数，生成指数更新 | 直接导数，生成指数更新 |', '',
        '检查的是实际生成C++的`states()`函数，不是METHOD标签或文件内嵌MOD文本。原版重建的4种状态函数与旧冻结C++去空白后一致；gates/both的门控状态函数、pool/both的浓度状态函数分别完全一致。每个变体的`patch.diff`和`build-evidence.json`保存改动、实际生成函数和逐文件SHA。独立审计重建允许的完整源文件差异，确认没有夹带缓存或离子同步修改。', '',
        '## 解析控制：12个小条件', '',
        '只测original与both两个代表版本，各用0.025、0.0125、0.00625 ms；gates/pool中的相应状态函数与both相同。固定电压门控为600 ms、−10 mV，5个实测初值为0.8/0.2/0.7/0.1/0.6，均与稳态相距至少0.05。所有密度电导置零，仅用于解析控制，SEClamp实测电压误差须≤1e−6 mV。参考为`xinf+(x0−xinf)exp(−t/τ)`，最大状态误差门槛1e−9。', '',
        '浓度控制使用105 ms、初始STATE/cai均2e−7 mM，实际钙电流密度依次为−0.002（0–50 ms）、0（50–100 ms）、+0.002 mA/cm²（100–105 ms）。前100 ms比较解析解，后5 ms只检查截断和暴露值同步。用实际密度、面积及CaPool读取的总电流交叉核对，不能将名义点电流当作钙电流。', '',
        '单位换算沿用冻结模型：面积为`area_um2×1e−8 cm²`，向内为正的池输入总电流为`−ica×area_cm2×1e6 nA`；浓度源项为`−1e6×rho×ica mM/ms`，rho=2.38919e−12，衰减常数11.5943 ms。实际面积314.159 μm²，非零总电流为±0.00628319 nA。STATE解析误差门槛1e−15 mM；zero-input段另以实测50 ms状态核对衰减。', '',
        '| 版本 | dt ms | 5门控最大误差 | 门控解析 | STATE浓度最大误差 mM（0–100 ms） | STATE解析 | 压力段最大STATE/cai差 mM |',
        '|---|---:|---:|---|---:|---|---:|']
    for v in ['original','both']:
        for dt in p['dt_ms']:
            g=s['analytic_controls'][f'gates_{v}_{dt:g}'];c=s['analytic_controls'][f'pool_{v}_{dt:g}']
            lines.append(f"| {v} | {dt:g} | {f(max(g['max_abs_state_error'].values()))} | {'通过' if g['passed'] else '未通过'} | {f(c['STATE_max_abs_error_mM'])} | {'通过' if c['STATE_analytic_passed'] else '未通过'} | {f(c['outward_stress_max_sync_error_mM'])} |")
    lines+=['', '原版解析误差随步长减半近似缩小，修复版达到浮点精度附近。向内/零输入段的修复版STATE和cai均通过解析误差要求。所有版本STATE非负；外向压力段却出现负cai，状态截断后的同步要求未通过。独立审计使用每步实际电流重建Euler/指数状态更新、截断后的STATE及截断前cai，均与保存数组相符。这个边界缺陷被保留，没有混入修复。', '',
        '## 完全相同输入的12个孤立肌肉主条件', '',
        '四版本×三步长，各8秒，后4秒验收。使用原C1肌肉（长20 μm、直径5 μm、cm=1 μF/cm²；初始−45 mV，cai=0、cao=2）、原NMJ机制、0.9 nS与释放阈值−30 mV。输入为本阶段声明的规定波形，仅是组件测试，不是神经自主活动：250 ms前−60 mV，经250–350 ms三次平滑过渡，进入`−32+6sin(2π×1.9t/1000)` mV。', '',
        '输入先保存为0.00625 ms高密度文件，各积分区间使用其左端值；没有使用10 ms全网迹线恢复输入。所有条件在每个0.025 ms共同记录点的peer电位逐元素一致，12条输入数组SHA相同；同一步长下四版本的实际突触s状态也逐元素一致。不同步长的区间保持宽度仍不同，属于明确保留的输入离散与总体积分误差，不将其冒称完全连续求解。每条迹线还分别保存前一区间实际使用的peer值，防止只比较记录端值而遗漏执行输入。', '',
        '状态每0.025 ms记录，含电位、5门控、5个对应幂缓存、h观测量、前一神经步5门控、实际ica、浓度STATE、cai、两种钙到驱动读出、突触s/电流及两类输入。解析控制则每个神经步记录。保存每条运行的spec、record、版本、输入哈希和实际时钟；主条件没有重跑或扩大。', '',
        '预定误差：电位逐时RMS≤0.5 mV、均值差≤0.2 mV、q95−q05幅度相对差≤5%（参考幅度下限1 mV）；STATE/cai/驱动的均值、标准差、q05/q95差与逐时RMS均除以细步长q95−q05后≤5%，钙尺度下限1e−9 mM、驱动0.001；5门控最大逐时RMS≤0.01。没有相位平移，也未按结果选择窗口。', '',
        '| 版本 | 比较 dt ms | 电位RMS差 mV | STATE钙标准化RMS | cai标准化RMS | 驱动标准化RMS | 5门控最大RMS | 全部容差 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for v in p['variants']:
        for a,b in [(.025,.0125),(.0125,.00625),(.025,.00625)]:
            c=s['paired_step_comparisons'][f'{v}_{a:g}_vs_{b:g}'];e=c['errors']
            vals=[e[k]['value'] for k in ['voltage_RMS_mV','STATE_calcium_scaled_RMS','cai_scaled_RMS','drive_scaled_RMS','gate_state_RMS']]
            lines.append(f"| {v} | {a:g}→{b:g} | "+' | '.join(f(x) for x in vals)+f" | {'通过' if c['passed'] else '未通过'} |")
    lines+=['', '表中标准化差为比例，例如0.0045表示参考动态范围的0.45%，不是45%。全部统计误差、直接0.025→0.00625比较及逐条失败项保存在summary.json，没有只用峰值验收。', '',
        '## 改善判据与后续决定', '',
        '协议在运行前规定：合并版相对原版，最细两档的电位、STATE钙、cai、驱动RMS误差须各缩小至少一半；原版误差已≤对应容差1/10的指标可豁免改善要求。豁免不表示实测误差变小。', '',
        '| 指标 | 合并/原版误差比 | 预定条件 |', '|---|---:|---|']
    for name,d in s['combined_over_original_improvement'].items():
        exempt=d['original_error']<=.1*s['paired_step_comparisons']['original_0.0125_vs_0.00625']['errors'][name]['limit']
        lines.append(f"| {name} | {f(d['ratio'])} | {'原版已小，按协议豁免；并未改善' if exempt else ('满足缩小要求' if d['strong_improvement'] else '未满足缩小要求')} |")
    lines+=['', '**本阶段没有证明总体响应改善。** 电位误差由0.079889 mV变为0.168627 mV，未满足缩小条件；钙/驱动误差也约增至2.1倍，只是仍很小且属于事先规定的豁免项。局部线性状态的解析积分变得正确，与复合非线性肌肉的逐时步长差变小，是两项不同的检验。可能存在误差抵消、膜电压积分、缓存和输入保持误差等作用，本轮没有单独确定各自贡献。', '',
        '**建议先做独立的缓存/调用时序隔离，不直接启动新的完整系统预算。** 本轮未测试共享机制对284个C1神经元的影响，也未测试神经—身体交换。若后续有新预算，必须独立记录并重新做完整系统收敛，不能把原六次失败改记为通过。', '',
        '## 缓存与同步的独立发现', '',
        '所有主条件保存的幂缓存，与“紧邻前一神经步”的门控幂一致至约1.11e−16；与当前更新后状态并不一致。以both为例，0.025/0.0125/0.00625 ms的更新后最大差分别为'+ '、'.join(f(s['main_metrics'][f'both_{dt:g}']['cache']['max_abs_to_current_state_after_t0']) for dt in p['dt_ms'])+'。这支持在已测隔离模型中存在一神经更新的缓存时序差；初始缓存与已初始化门控也有近1的最大差。尚未通过独立补丁比较量化它对总体步长误差的贡献。', '',
        '普通主条件的cai与STATE在所有保存点完全一致且非负；不能把外向压力控制的负cai描述为普通主条件已发生的失稳。压力段最大不同步约1.19e−10 mM（0.025 ms），随步长减半缩小，both也保留此问题。修改缓存、初始化后刷新或负浓度同步，均应作为下一项明确隔离的变体，不能回写本阶段。', '',
        '## 验证、原始文件与复算', '',
        f"独立程序不导入主分析或NEURON，检查{audit['protected_full_source_files_checked']}个完整源文件、12个解析控制、12个主条件及{audit['paired_gate_values_checked']}个误差/阈值判定；还按实际每步电流独立重建浓度更新与cai暴露顺序。协议及冻结输入哈希核对通过。", '',
        '准备阶段两次观测设置失败发生在第一次fadvance之前，原日志/spec保存在setup-attempts，不计为科学条件；随后只修正GLOBAL变量iCa的读取方式。另修正生成代码检查器对负系数的字符串识别，保留运行前准备脚本副本；协议、补丁和验收数值未改。', '',
        '- 原始数据、协议、输入与四版本：`results/c1-muscle-integration-isolation-2026-11-09/`。',
        '- `summary.json`为可重算汇总，`independent-validation.json`为独立审计，`sources.json`/`build-evidence.json`保存来源和变体SHA；`variants/*/patch.diff`保存最小补丁。',
        '- 正式归档：`docs/reports/assets/c1-muscle-integration-isolation-2026-11-09/reproduction.zip`及SHA文件。归档保留项目相对路径；重算只需NumPy，运行肌肉才需要固定NEURON与本机编译机制。其他平台需要另行重建并记录新来源。',
        '- 前一阶段kernel、六条原始结果、汇总和归档未修改。本阶段无完整302/身体运行。', '',
        '在解压后的项目根目录，使用具备NumPy 2.3.5的Python执行：', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/integration_analyze.py',
        'OPENBLAS_NUM_THREADS=1 python analysis/neuromechanics/b_motor_validation/integration_verify.py',
        '```', '']
    REPORT.write_text('\n'.join(lines));ASSETS.mkdir(parents=True,exist_ok=True)
    payload={}
    for path in ROOT.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path not in [ROOT/'archive-validation.json',ROOT/'task-control.json']:
            payload[str(path.relative_to(PROJECT))]=path
    for name in SCRIPTS:
        path=Path(__file__).resolve().with_name(name);payload[str(path.relative_to(PROJECT))]=path
    payload[str(REPORT.relative_to(PROJECT))]=REPORT
    manifest={name:sha(path) for name,path in sorted(payload.items())}
    archive=ASSETS/'reproduction.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(payload.items()):z.write(path,name,compress_type=zipfile.ZIP_STORED if path.suffix=='.npz' else zipfile.ZIP_DEFLATED)
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with tempfile.TemporaryDirectory(prefix='worm-integration-archive-') as temp:
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(temp)
        for name,digest in manifest.items():assert sha(Path(temp)/name)==digest
        for name in ['integration_analyze.py','integration_verify.py']:
            subprocess.run([sys.executable,str(Path(temp)/'analysis/neuromechanics/b_motor_validation'/name)],cwd=temp,
                check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'},stdout=subprocess.DEVNULL)
        assert read(Path(temp)/'results/c1-muscle-integration-isolation-2026-11-09/summary.json')==s
    digest=sha(archive);archive.with_suffix('.zip.sha256').write_text(digest+'  reproduction.zip\n')
    result=dict(archive_sha256=digest,archive_bytes=archive.stat().st_size,payload_files=len(payload),
        CRC_passed=True,all_extracted_SHA_passed=True,isolated_extracted_analysis_and_independent_audit_passed=True,
        summary_sha256=sha(ROOT/'summary.json'),report_sha256=sha(REPORT))
    (ROOT/'archive-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()

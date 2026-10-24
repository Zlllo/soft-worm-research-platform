# Cook2019 成年雌雄同体连接数据

原始研究：Cook et al., 2019, *Whole-animal connectomes of both Caenorhabditis elegans sexes*，DOI: https://doi.org/10.1038/s41586-019-1352-7 。使用 WormWiring 于 2020 年 7 月修正后的版本，通过 OpenWorm ConnectomeToolbox 的固定版本导入。

- 数据仓库：https://github.com/openworm/ConnectomeToolbox
- 固定 commit：`ffe570268b7cffb9329cd009a1bb37d60db2d465`
- 上游缓存：`cect/cache/Cook2019HermReader.json`
- 缓存 Git blob SHA-1：`2c44e7abaa1efa71c4d814a945692f5775692abc`
- 对应修正版 workbook Git blob：`4d53c16e9a07c1ecfd068a02b39ccffd6b8a3eaa`
- 抽取日期：2026-10-24。

`cook2019_herm.json` 无损保留缓存的节点顺序和全部非零连接；每条边是 `[source_index,target_index,type,weight]`。矩阵方向是行=源、列=目标。`herm_neurons.json` 的302个神经元名称来自同一 commit 的 `cect/Cells.py`，使用显式名单过滤，不把肌肉和其他细胞算作神经元。

全数据有473个细胞。化学连接4879个非零有向条目，总权重28113；gap矩阵2883个非零条目，总权重23313。gap矩阵包含双向存储与17个对角条目，不能将条目数直接当作独立gap数量。神经元子网为302节点，3709个化学条目，2196个gap条目。

权重是表示连接数量与大小的 EM serial sections 总数，不是训练权重、功能强度或兴奋/抑制符号。这里不推断未知连接的符号。gap自环仍保存在原始数据中；简化扩散模型中不建立自耦合，因为它对差分项贡献为零。

`upstream_validation.yaml` 原样保存上游人工核查条目与总数；本项目测试复核19个条目、方向和全矩阵统计。这能检查导入是否一致，不能独立验证原始电镜重建的生物准确性。

重建紧凑数据（需要网络）：

```bash
python3 analysis/import_connectome.py --download --output /tmp/cook2019_herm.json
```

也可以使用 `--cache 原始缓存路径` 离线重建。导入程序校验原始 Git blob SHA-1，不执行上游代码。

上游项目 MIT 软件许可证保存在 `OPENWORM_LICENSE.txt`。原始实验数据的来源与引用应始终保留；本项目不为原始研究数据另外声明授权范围。

早期 `herm_full_edgelist.csv` 被上游标为 legacy / do not use，不作为本研究正式数据入口。

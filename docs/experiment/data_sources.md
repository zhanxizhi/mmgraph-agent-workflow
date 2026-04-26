# 电力实验数据源核查与使用说明

核查日期：2026-04-24

本文实验数据源分为“已进入当前主实验的数据源”和“已核查、适合作为扩展实验或图谱 schema 参考的数据源”。当前已新增 `data/benchmarks/power_gold_workflows_extended.json`，包含 30 条人工金标工作流；原始 6 条小样本仍保留在 `data/benchmarks/power_gold_workflows.json` 中用于快速回归。

| 数据源 | 核查入口 | 当前用途 | 本地/代码状态 |
|---|---|---|---|
| PowerGridQA | https://github.com/Hannaancode/PowerGridQA；论文 DOI: https://doi.org/10.1109/ACCESS.2026.3652625 | KQA 主数据源；NERC 子集也用于 OPS 故障排查任务改写 | 已存在于 `data/raw/kqa/`；`PowerGridQALoader` 可读取 |
| Open Power System Data (OPSD) | https://data.open-power-system-data.org/；论文 DOI: https://doi.org/10.1016/j.apenergy.2018.11.097 | DA 主数据源，构造负荷预测、峰谷差、新能源出力分析任务 | 已存在 `data/raw/opsd/time_series_60min_singleindex.csv`；`OPSDLoader` 可读取 |
| Electricity Knowledge Graph | https://github.com/sensorlab/energy-knowledge-graph；论文 DOI: https://doi.org/10.1038/s41597-024-04310-z | 电力消费知识图谱、RDF/元数据组织方式参考 | 已存在于 `data/raw/graph_base/`；当前未被主评估脚本直接读入 |
| ENTSO-E Transparency Platform | https://transparency.entsoe.eu/；综述 DOI: https://doi.org/10.1016/j.apenergy.2018.04.048 | DA/OPS 扩展数据源，适合补充欧洲实时与历史运行数据 | 需手动导出；当前未被 loader 直接解析 |
| Texas A&M Electric Grid Test Cases | https://electricgrids.engr.tamu.edu/electric-grid-test-cases/；验证论文 DOI: https://doi.org/10.3390/en10081233 | OPS/KQA 扩展数据源，适合补充合成电网拓扑、潮流、恢复场景 | 需按案例手动下载；当前未被 loader 直接解析 |
| CIM-Graph / CIMantic Graphs | https://github.com/PNNL-CIM-Tools/CIM-Graph；软件 DOI: https://doi.org/10.11578/dc.20240507.3 | CIM 模型图结构、实体类型和关系 schema 参考 | 已存在于 `data/raw/cim_graph/`；当前未被主评估脚本直接读入 |
| OpenEI/OEDI Event-correlated Outage Dataset in America | https://data.openei.org/submissions/6458 | OPS 扩展数据源，适合补充美国停电事件与异常报告 | 需手动下载；当前未被 loader 直接解析 |

## 当前样本规模

- PowerGridQA 本地版本：58,713 条 JSONL 记录，其中 `Questions Power System Theory.jsonl` 53,138 条、`Nerc questions.jsonl` 4,787 条、`Reasoning questions.jsonl` 788 条。
- OPSD 本地时间序列文件：50,401 条小时级记录，300 个字段。
- 当前工作流评估金标：`data/benchmarks/power_gold_workflows_extended.json` 中 30 条任务，OPS/KQA/DA 各 10 条；原始 `power_gold_workflows.json` 中 6 条任务可用 `--legacy-gold` 参数调用。

## 写作边界

论文可以写：本研究以 PowerGridQA 与 OPSD 作为当前可复现实验的主要数据来源，并核查 Electricity Knowledge Graph、CIM-Graph、ENTSO-E、Texas A&M 测试网和 OpenEI/OEDI 作为领域图谱扩展与扩展基准任务来源。

论文不应写：所有列出的数据源都已经被完整解析为训练/评估样本，或当前 30 条任务能代表这些数据源的完整统计分布。

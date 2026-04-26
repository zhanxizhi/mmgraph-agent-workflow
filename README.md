# 基于多模态图检索增强的智能体工作流生成研究

> 华中科技大学本科毕业设计（2026）  
> 作者：詹熙至 | 学号：U202215549 | 指导教师：莫益军

## 项目简介

本仓库对应本科毕业设计《基于多模态图检索增强的智能体工作流自主生成方法研究》。项目面向电力系统运维、知识问答和数据分析场景，探索如何利用知识图谱、GraphRAG 检索和智能体特征图，自动生成可验证的多智能体工作流。

当前实现的核心链路为：

1. 从电力领域文本资料中抽取实体与关系，构建 Neo4j 知识图谱并建立向量索引；
2. 通过语义向量检索和 BFS 子图扩展形成 GraphRAG 上下文；
3. 使用 LLM 将任务分解为子任务，并基于能力相似度、工具覆盖率和输入输出兼容性进行子任务-智能体匹配；
4. 将匹配结果编排为 DAG 工作流，并进行无环性、接口兼容性和任务完整性验证；
5. 在 OPS、KQA、DA 三类电力任务上进行 30 条基准任务评测和消融实验。

说明：论文中讨论了多模态图谱构建框架，但当前端到端实验主要验证文本图谱链路；图像、流程图和跨模态对齐链路保留为后续扩展方向。

## 主要结果

实验结果位于 `data/benchmarks/`，论文图表数据位于 `thesis/figures/data/`。当前 30 条扩展基准任务的全场景结果摘要如下：

| 方法 | Exec | SC | SC_lex | RAA | Logic |
|---|---:|---:|---:|---:|---:|
| ours_full | 1.000 | 0.768 | 0.196 | 0.296 | 0.711 |
| pure_llm | 1.000 | 0.696 | 0.091 | 0.000 | 0.947 |
| vector_rag | 1.000 | 0.784 | 0.181 | 0.425 | 0.912 |
| graphrag_nomatch | 1.000 | 0.784 | 0.180 | 0.056 | 0.868 |

主要结论：

- 三维量化匹配算法是角色分配准确率（RAA）的主要增益来源；
- 固定 `h=2` 的 BFS 子图扩展在当前图谱规模下未带来稳定收益，反而会引入噪声节点；
- BGE-M3 语义版步骤完整性指标比词汇版更适合中文电力工作流评估；
- LLM-as-Judge 的 Logic 指标存在偏向流畅文本的系统性偏差，论文中将 RAA 和 SC 作为更可靠的核心指标。

## 目录结构

```text
mmgraph-agent-workflow/
├── data/
│   ├── benchmarks/       # 金标工作流、评测结果、权重扫描缓存
│   ├── raw/              # 原始数据目录占位；大文件未纳入 Git
│   ├── processed/        # 预处理数据目录占位
│   └── graphs/           # 图谱导出目录占位
├── docs/
│   ├── algorithm/        # 算法设计文档
│   └── experiment/       # 实验方案与数据来源说明
├── experiments/
│   └── configs/          # 实验配置
├── scripts/
│   ├── run_eval_ops_workflow.py  # 30 任务端到端评测脚本
│   ├── run_weight_sweep.py       # 算法 C 权重敏感性扫描
│   └── start_neo4j.sh            # Neo4j Docker 启动脚本
├── src/
│   ├── agent_workflow/   # 智能体特征图、匹配与 DAG 生成
│   ├── common/           # LLM 客户端与通用类型
│   ├── data_loader/      # 电力数据加载
│   ├── evaluation/       # 评估指标与运行器
│   ├── graph_builder/    # 知识图谱构建
│   └── graphrag/         # GraphRAG 检索
├── tests/                # 轻量回归测试
└── thesis/               # 毕业论文 LaTeX 源码、图表和 main.pdf
```

## 环境准备

建议使用 Python 3.11。

```bash
conda create -n mmgraph python=3.11 -y
conda activate mmgraph
pip install -r requirements.txt
```

复制环境变量模板，并填入 LLM 和 Neo4j 配置：

```bash
cp .env.example .env
```

`.env` 不会被提交到 Git。仓库中不包含 API Key。

## 运行方式

启动 Neo4j：

```bash
bash scripts/start_neo4j.sh
```

运行一个工作流生成示例：

```bash
python -m src.agent_workflow.main \
  --task "220kV线路距离保护I段动作跳闸，重合闸失败，判断故障性质并制定处置方案" \
  --config experiments/configs/default.yaml
```

重新生成论文图表数据（不调用 LLM，不连接 Neo4j）：

```bash
python scripts/run_eval_ops_workflow.py --plots-only
```

运行完整 30 任务评测（需要 Neo4j 和 LLM API）：

```bash
python scripts/run_eval_ops_workflow.py
```

运行算法 C 权重扫描：

```bash
python scripts/run_weight_sweep.py --use-cache
```

## 测试

仓库包含两个不依赖外部服务的轻量回归测试：

```bash
python tests/test_step_completeness.py
python tests/test_workflow_validator_repair.py
```

如果安装了 `pytest`，也可以运行：

```bash
python -m pytest -q
```

## 论文

论文 LaTeX 源码位于 `thesis/`，当前生成版 PDF 为：

```text
thesis/main.pdf
```

参考文献核查记录：

```text
thesis/reference_verification.md
```

## 数据与隐私说明

- `.env`、本地编辑器配置、缓存文件、日志和原始大数据未纳入 Git；
- `data/raw/` 仅保留 `.gitkeep`，需要按 `docs/experiment/data_sources.md` 自行准备原始数据；
- `data/benchmarks/` 中保留了论文实验所需的金标工作流和评测结果；
- 论文 PDF、LaTeX 源码和图表数据已纳入仓库，便于复现论文图表。

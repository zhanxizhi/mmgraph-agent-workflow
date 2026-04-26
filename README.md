# 基于多模态图检索增强的智能体工作流生成

> 华中科技大学 本科毕业设计 (2026)
> 作者：詹熙至 | 指导教师：莫益军

## 项目简介

本项目构建一个以多模态图谱为知识底座、以 GraphRAG 为检索推理中枢、以智能体特征图为编排依据的工作流自主生成系统，对应任务书四项要求：

1. 基于图的智能体 RAG 分类综述
2. 基于 GraphRAG 的智能体工作流自主生成框架设计
3. GraphRAG 与智能体特征融合方法研究
4. 基于 GraphRAG 的智能体工作流编排规划智能体实现

## 技术栈

| 层级 | 组件 |
|------|------|
| LLM | OpenAI GPT-4o / Qwen2.5 / DeepSeek (通过统一接口切换) |
| 多模态编码 | CLIP / SigLIP (视觉), BGE-M3 (文本) |
| 流程图解析 | PaddleOCR + PP-StructureV2 |
| 图数据库 | **Neo4j 5.x** |
| GraphRAG | 微软 graphrag 库 + 自研扩展 |
| 智能体编排 | **LangGraph** (StateGraph 与本课题图建模天然同构) |
| 工作流输出 | DAG (JSON) / BPMN 2.0 / Mermaid |
| 可视化 | Streamlit + BPMN.js |

## 目录结构

```
mmgraph-agent-workflow/
├── data/
│   ├── raw/              # 原始多模态数据 (文本/图片/流程图)
│   ├── processed/        # 预处理后的数据
│   ├── graphs/           # 构建好的图谱导出 (Neo4j dump / GraphML)
│   └── benchmarks/       # 评估用数据集与标注
├── src/
│   ├── graph_builder/    # 多模态图谱构建模块
│   ├── graphrag/         # GraphRAG 检索推理模块
│   ├── agent_workflow/   # 智能体特征图与工作流生成模块
│   ├── common/           # LLM 接口、配置、日志、工具函数
│   └── evaluation/       # 评估指标与实验运行器
├── experiments/
│   ├── configs/          # YAML 实验配置
│   ├── scripts/          # 实验启动脚本
│   ├── results/          # 实验结果 (CSV/JSON)
│   └── logs/             # 运行日志
├── notebooks/            # 数据探查与可视化 Jupyter
├── docs/
│   ├── algorithm/        # 算法设计文档 (第四章素材)
│   ├── experiment/       # 实验方案文档 (第五章素材)
│   └── paper/            # 论文章节草稿
├── tests/                # 单元测试
└── scripts/              # 一次性脚本 (数据下载、Neo4j 初始化等)
```

## 快速开始

```bash
# 1. 创建环境
conda create -n mmgraph python=3.11 -y
conda activate mmgraph
pip install -r requirements.txt

# 2. 启动 Neo4j (Docker)
bash scripts/start_neo4j.sh

# 3. 配置 API Key
cp .env.example .env  # 编辑填入 OPENAI_API_KEY 等

# 4. 运行最小流程
python -m src.agent_workflow.main --task "示例任务描述" --config experiments/configs/default.yaml
```

## 进度

- [x] 开题报告
- [x] 项目结构搭建
- [ ] 多模态图谱构建模块
- [ ] GraphRAG 检索模块
- [ ] 智能体特征图构建
- [ ] 工作流生成模块
- [ ] 系统集成与实验
- [ ] 论文撰写

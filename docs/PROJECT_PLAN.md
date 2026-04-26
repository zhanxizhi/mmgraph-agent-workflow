# 项目总体规划

> 基于多模态图检索增强的智能体工作流生成
> 华中科技大学 2026 本科毕业设计 | 詹熙至 | 指导教师：莫益军
> 规划日期：2026-04-06

---

## 1. 现状盘点

### 1.1 已完成

| 内容 | 文件 | 状态 |
|------|------|------|
| 项目目录结构 | 全部 `src/`、`data/`、`experiments/` 目录 | ✅ |
| 数据类型定义 | `src/common/types.py` | ✅ |
| 算法设计文档 | `docs/algorithm/algorithm_design.md` | ✅ |
| 实验方案文档 | `docs/experiment/experiment_plan.md` | ✅ |
| 实验配置 | `experiments/configs/default.yaml` | ✅ |
| 依赖清单 | `requirements.txt` | ✅ |
| 所有模块骨架 | `src/*/*.py`（均为 `raise NotImplementedError`） | ✅ |

### 1.2 待实现（全部核心代码）

| 模块 | 文件 | 优先级 |
|------|------|--------|
| LLM 统一接口 | `src/common/llm_client.py` | P0（其他一切的前提） |
| 多模态图谱构建 | `src/graph_builder/builder.py` | P0 |
| GraphRAG 检索 | `src/graphrag/retriever.py` | P0 |
| 智能体特征图 | `src/agent_workflow/feature_graph.py` | P1 |
| 工作流生成器 | `src/agent_workflow/generator.py` | P1 |
| 评估指标 | `src/evaluation/metrics.py` | P1 |
| Neo4j 初始化脚本 | `scripts/start_neo4j.sh` | P2 |
| 主入口 | `src/agent_workflow/main.py` | P2 |
| 数据集准备 | `data/benchmarks/` | P1（与实现并行） |

---

## 2. 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                   用户任务描述 (query)                    │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                 算法 B：GraphRAG 检索推理                       │
│  1. BGE-M3 向量搜索种子节点 (top-k=10)                        │
│  2. 加权游走采样扩展子图 (max_hops=2)                         │
│  3. 社区摘要补充                                               │
│  4. LLM 重排序剪枝                                            │
│  5. 结构化上下文格式化                                        │
└───────────────┬──────────────────────────────┬────────────────┘
                │ graph_context                │
                ▼                              ▼
┌──────────────────────┐          ┌─────────────────────────────┐
│  算法 C1：任务分解    │          │   算法 C2：子任务-智能体匹配  │
│  (LLM + 能力词表约束)│ subtasks │  (能力嵌入 + 工具覆盖 + IO)  │
└──────────────────────┘─────────▶└─────────────────────────────┘
                                              │ assignments
                                              ▼
                              ┌──────────────────────────────┐
                              │  算法 D：工作流生成与验证      │
                              │  1. 显式/隐式依赖推导         │
                              │  2. 拓扑排序                  │
                              │  3. 三层验证                  │
                              │  4. 导出 JSON/BPMN/Mermaid   │
                              └──────────────────────────────┘

              ↑ 上层依赖底层的图谱 ↑

┌─────────────────────────────────────────────────────────────┐
│               算法 A：多模态图谱构建（离线）                    │
│                                                             │
│  文本文档 ──LLM抽取──▶  文本节点 / 实体节点                  │
│  图像     ──CLIP编码──▶  图像节点                            │
│  流程图   ──OCR+检测──▶  流程图节点（顺序边）【创新点】        │
│                    ↓                                        │
│            跨模态对齐（余弦相似度 > 0.75）                   │
│                    ↓                                        │
│          社区发现（Neo4j GDS Leiden）+ 社区摘要              │
│                    ↓                                        │
│                   Neo4j 5.x                                 │
└─────────────────────────────────────────────────────────────┘

              ↑ 知识底座 ↑

┌──────────────────────────────────────────────────────────┐
│               智能体特征图（注册表，离线维护）               │
│  Agent 节点 ──HAS_CAPABILITY──▶ Capability 节点           │
│             ──USES_TOOL──────▶ Tool 节点                  │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 实现路线（四个阶段）

### 第一阶段：基础设施（4月第1周，当前优先）

**目标**：把所有 `raise NotImplementedError` 的 P0 模块跑通，能走完一个最小端到端流程。

| 任务 | 文件 | 关键点 |
|------|------|--------|
| **实现 LLMClient** | `src/common/llm_client.py` | 支持 OpenAI 兼容接口（GPT-4o-mini / DeepSeek / Qwen 通过 base_url 切换）；实现 `chat`、`embed`、`structured_output` |
| **实现文本节点抽取** | `src/graph_builder/builder.py::extract_text_nodes` | chunk + LLM 结构化输出（实体+摘要一次调用），存入 GraphNode 列表 |
| **实现 Neo4j 写入** | `src/graph_builder/builder.py::build` | Cypher MERGE 节点/边，带嵌入向量（Neo4j vector index） |
| **实现向量检索** | `src/graphrag/retriever.py::semantic_search` | Neo4j 向量索引 `db.index.vector.queryNodes` |
| **实现子图扩展** | `src/graphrag/retriever.py::expand_subgraph` | BFS N 跳，先不做加权游走采样（简单版先跑通） |
| **实现上下文格式化** | `src/graphrag/retriever.py::format_context` | 按算法设计文档第2.3节的 Markdown 格式 |
| **搭建 Neo4j** | `scripts/start_neo4j.sh` | Docker 一键启动，开启 GDS + APOC 插件 |
| **OPS 场景数据准备** | `data/raw/ops/` | Ansible/K8s 故障文档 + 流程图，50个任务，人工/GPT-4o辅助标注 |

**里程碑**：能对一个 OPS 场景任务执行 `retrieve(query)` 并返回结构化上下文。

---

### 第二阶段：核心算法（4月第2周）

**目标**：完成图像/流程图处理（创新点）、智能体匹配、工作流生成。

| 任务 | 文件 | 关键点 |
|------|------|--------|
| **实现图像节点** | `builder.py::extract_image_nodes` | CLIP encode_image + VLM caption（用 Qwen-VL 或 BLIP-2） |
| **实现流程图解析** | `builder.py::extract_flowchart_nodes` | PaddleOCR 文字框 → 箭头检测 → 构建 SEQUENTIAL 边 → LLM 后处理 |
| **实现跨模态对齐** | `builder.py::align_cross_modal` | 投影到统一空间 → 余弦相似度 > 阈值（0.75）→ SEMANTIC 边 |
| **实现社区发现** | `builder.py::detect_communities` | 调用 Neo4j GDS Leiden + LLM 生成社区摘要 |
| **实现 AgentFeatureGraph** | `feature_graph.py::to_neo4j` | 写入 Agent/Capability/Tool 节点和边 |
| **实现任务分解** | `feature_graph.py::decompose_task` | LLM structured_output + 能力词表约束 |
| **实现智能体匹配** | `feature_graph.py::match_subtask_to_agent` | 三维度打分（cap_sim, tool_coverage, io_compat），工具覆盖率硬约束 |
| **实现工作流生成** | `generator.py::generate` | 端到端串联：retrieve → decompose → match → buildDAG → sort |
| **实现 LangGraph 编译** | `generator.py::to_langgraph` | 将 DAG 编译为 LangGraph StateGraph |
| **实现三层验证** | `generator.py::WorkflowValidator` | DAG 无环检查、IO 兼容、LLM-as-judge 完整性 |
| **KQA + DA 数据准备** | `data/raw/kqa/`, `data/raw/da/` | 50+50 任务用例，GPT-4o 辅助预标注 |

**里程碑**：给定任务描述，能输出完整 Workflow DAG（JSON + Mermaid 可视化）。

---

### 第三阶段：评估与实验（4月第3-4周）

**目标**：跑通实验一、二、三，产出论文所需的数字。

| 任务 | 文件 | 关键点 |
|------|------|--------|
| **实现评估指标** | `src/evaluation/metrics.py` | 实现全部 P/R/F1、Recall@k、MRR、可执行率等 |
| **实现实验运行器** | `src/evaluation/runner.py`（新建） | 读取 YAML 配置 → 批量跑 → 自动写 JSON/CSV 结果 |
| **实验一：图谱构建质量** | `experiments/results/exp1_graph_quality/` | 三对比组（LLM-only / +CLIP / +Flowchart） |
| **实验二：GraphRAG 检索** | `experiments/results/exp2_retrieval/` | 五对比组，报告 Recall@10、MRR、相关性 |
| **实验三：工作流生成** | `experiments/results/exp3_workflow/` | B1-B5 全跑，消融 α/β/γ 权重网格搜索 |
| **LLM-as-judge 实现** | `src/evaluation/llm_judge.py`（新建） | 同一样本评 3 次取均值，关键指标人工复核 |
| **加权游走采样优化** | `graphrag/retriever.py::expand_subgraph` | 替换朴素 BFS，实现算法设计文档第2.2节的采样策略 |
| **LLM 重排序** | `graphrag/retriever.py::rerank` | LLM 打分 → 保留 top 60% 节点 |

**里程碑**：三张主要结果表格（实验一/二/三）数据填满，论文第五章可以动笔。

---

### 第四阶段：系统集成与论文（5月）

**目标**：完整系统跑通 + 案例研究 + 论文写作。

| 任务 | 关键点 |
|------|--------|
| 实验四：案例研究 | 每场景 1 个典型任务，可视化子图 + 生成过程 + Mermaid 工作流 |
| Streamlit 演示界面 | 上传文档 → 建图 → 输入任务 → 展示 DAG，答辩演示用 |
| Mermaid / BPMN 导出 | `generator.py::to_bpmn` / `to_mermaid` 完整实现 |
| 论文第四章 | 对应算法设计文档，补充理论推导和复杂度分析 |
| 论文第五章 | 对应实验方案文档，填入实验数据，写分析和消融结论 |
| 答辩 PPT | 重点：系统架构图、算法 C 打分机制（可解释性）、实验三主表 |

---

## 4. 模块依赖关系

```
LLMClient
    ↑
    ├── MultimodalGraphBuilder (文本/图像/流程图 → Neo4j)
    │       ↑
    │   VisionEncoder (CLIP/SigLIP)
    │   OCREngine (PaddleOCR)
    │   Neo4jDriver
    │
    ├── GraphRAGRetriever (Neo4j → 结构化上下文)
    │       ↑
    │   Neo4jDriver
    │
    ├── AgentFeatureGraph + AgentMatcher (智能体匹配)
    │
    └── WorkflowGenerator (串联以上所有)
            ↓
        WorkflowValidator
            ↓
        Workflow (JSON / BPMN / Mermaid / LangGraph)
```

**开发顺序**（严格遵循依赖关系）：
`LLMClient` → `Neo4j 环境` → `GraphBuilder (文本)` → `GraphRAGRetriever` → `AgentFeatureGraph` → `WorkflowGenerator` → `GraphBuilder (图像+流程图)` → `评估指标` → `实验运行器`

---

## 5. 关键技术决策

### 5.1 LLM 接口统一化（首要任务）
- 统一用 OpenAI 兼容接口，`base_url` 切换 provider
- GPT-4o-mini 用于生产实验（成本低），GPT-4o 用于 LLM-as-judge
- DeepSeek-V3 作为替代（如果 OpenAI 访问有问题）

### 5.2 Neo4j 向量索引
- 嵌入维度：BGE-M3 = 1024，CLIP = 512
- 图像节点：双塔嵌入 = concat(CLIP_visual_emb, BGE_text_emb) → 1536 维
- 需要建立分离的向量索引（按 NodeType 分开，或建统一索引用过滤）

### 5.3 流程图解析（最大技术风险）
- 用 PaddleOCR 检测文本框，PP-StructureV2 检测表格/区域
- 箭头检测用 OpenCV 霍夫变换（直线检测）+ 轮廓分析
- LLM 后处理是兜底：OCR 结果差时，让 LLM 直接看图理解结构
- **风险预案**：解析失败时降级为"图像节点"（作为消融实验 B3 的一部分）

### 5.4 智能体特征图内容
- 先定义 OPS 场景的 10-15 个智能体（日志分析、网络检测、容器管理等）
- 能力/工具词表要与 GraphRAG 上下文中出现的词汇对齐（避免零召回）

### 5.5 评估数据标注策略
- GPT-4o 做初步预标注（半自动）
- 人工只做校验和修正（降低标注工作量）
- 关键指标（逻辑正确性、角色分配准确率）必须人工抽检 20% 样本

---

## 6. 时间表（对齐实验方案）

| 时间 | 阶段 | 主要产出 |
|------|------|---------|
| **4月第1周**（当前）| 第一阶段 | LLMClient ✓ + 文本图谱建图 ✓ + 向量检索 ✓ + OPS数据准备 ✓ |
| **4月第2周** | 第二阶段 | 图像/流程图处理 ✓ + 智能体匹配 ✓ + 工作流生成 ✓ + KQA/DA数据 ✓ |
| **4月第3周** | 第三阶段前半 | 评估指标 ✓ + 实验一 ✓ + 实验二 ✓ |
| **4月第4周** | 第三阶段后半 | 实验三主表 ✓ + 消融实验 ✓ |
| **5月第1周** | 第四阶段前半 | 案例研究 ✓ + Streamlit 演示 ✓ |
| **5月第2周** | 论文 | 第五章初稿 |
| **5月第3-4周** | 收尾 | 全文修改 + 答辩 PPT |

---

## 7. 风险与预案

| 风险 | 概率 | 预案 |
|------|------|------|
| PaddleOCR 流程图解析质量差 | 中 | 降级到"纯图像节点"，仍可作为消融实验 B3 的对比 |
| Neo4j GDS Leiden 在 Docker 环境跑不起来 | 低 | 用 networkx 的 Louvain 算法替代（精度稍低，但够用） |
| LLM API 访问不稳定/成本超支 | 中 | 批量实验用本地 Qwen2.5-72B（vLLM 部署），judge 用 GPT-4o |
| 数据标注工作量超出预期 | 中 | 严格限制标注范围：每场景只标注 40 个测试样本 |
| 三个场景实验时间不够 | 低 | 优先保 OPS（任务书直接相关），KQA/DA 各减至 30 样本 |
| 智能体匹配后 IO 不兼容导致工作流验证失败率高 | 中 | 放宽 IO 兼容检查（从硬约束改为软约束），并插入适配器步骤 |

---

## 8. 下一步（本周立即执行）

1. **实现 `LLMClient`**（`src/common/llm_client.py`）：OpenAI 兼容接口，chat + embed + structured_output
2. **实现 `extract_text_nodes`**（`src/graph_builder/builder.py`）：chunk → LLM 抽取 → GraphNode 列表
3. **实现 Neo4j 写入与向量索引**（`builder.py::build`）：MERGE 节点、创建向量索引
4. **实现 `semantic_search` + `format_context`**（`src/graphrag/retriever.py`）：最小可用检索链路
5. **准备 OPS 场景数据**（`data/raw/ops/`）：下载 K8s/Ansible 故障文档 + 10-20 张流程图
6. **搭建 Neo4j**（`scripts/start_neo4j.sh`）：Docker Compose，开 GDS + APOC 插件

**验收标准**：运行 `python -m src.agent_workflow.main --task "Pod CrashLoopBackOff排查" --config experiments/configs/default.yaml`，能打印出结构化的 GraphRAG 上下文（即使工作流生成部分还是 stub）。

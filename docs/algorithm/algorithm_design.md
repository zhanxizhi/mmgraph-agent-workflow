# 算法设计文档

> 对应论文第四章：GraphRAG 与智能体特征融合方法研究
> 本文档目标：把第三章的框架细化到可实现的算法级别

## 0. 符号约定

| 符号 | 含义 |
|------|------|
| $\mathcal{G} = (V, E)$ | 多模态知识图谱 |
| $V = V_T \cup V_I \cup V_F$ | 节点集 (文本/图像/流程图) |
| $E$ | 边集，含语义边、顺序边、引用边 |
| $\mathbf{e}_v \in \mathbb{R}^d$ | 节点 $v$ 的嵌入向量 |
| $\mathcal{A} = \{a_1, \ldots, a_n\}$ | 智能体特征图 |
| $q$ | 用户任务描述 |
| $\mathcal{S}_q$ | 任务子图 (检索得到的相关子图) |
| $\mathcal{W} = (N_W, E_W)$ | 输出工作流 DAG |

---

## 1. 算法 A：多模态图谱构建

### 1.1 总体流程

```
输入: sources = [{type, content}]
输出: 多模态图谱 G

Step 1. 模态分流
Step 2. 各模态独立抽取节点
Step 3. 跨模态对齐生成语义边
Step 4. 社区发现生成社区摘要
Step 5. 写入 Neo4j
```

### 1.2 文本节点抽取（A1）

```
function ExtractTextNodes(doc):
    chunks ← chunk_with_overlap(doc, size=512, overlap=64)
    nodes ← []
    for c in chunks:
        # LLM 同时输出实体和摘要 (一次调用降本)
        result ← LLM.structured_output(
            prompt = ENTITY_EXTRACTION_PROMPT(c),
            schema = {entities: [...], summary: str}
        )
        chunk_node ← GraphNode(type=TEXT, content=result.summary)
        nodes.append(chunk_node)
        for ent in result.entities:
            ent_node ← GraphNode(type=ENTITY, content=ent.name)
            edges.append(Edge(chunk_node, ent_node, REFERENCES))
    return nodes, edges
```

**关键设计**：分块时保留重叠，避免实体被切断；用结构化输出强约束 LLM，避免后续解析失败。

### 1.3 图像节点抽取（A2）

```
function ExtractImageNodes(image):
    visual_emb ← CLIP.encode_image(image)
    caption ← VLM.generate_caption(image)   # BLIP-2 / Qwen-VL
    text_emb ← BGE.encode(caption)
    return GraphNode(
        type=IMAGE,
        content=caption,
        embedding=concat(visual_emb, text_emb)  # 双塔拼接
    )
```

### 1.4 流程图解析（A3）—— 课题创新点

```
function ParseFlowchart(image):
    # 步骤1: OCR 提取文本框 + 坐标
    text_boxes ← PaddleOCR(image)

    # 步骤2: 检测连接线 (霍夫变换 / PP-StructureV2)
    arrows ← detect_arrows(image)

    # 步骤3: 构建初始图 (文本框=节点, 箭头=有向边)
    nodes ← [GraphNode(type=FLOWCHART_STEP, content=tb.text) for tb in text_boxes]
    edges ← []
    for arrow in arrows:
        src ← nearest_box(arrow.start, text_boxes)
        dst ← nearest_box(arrow.end, text_boxes)
        edges.append(Edge(src, dst, SEQUENTIAL))

    # 步骤4: LLM 后处理 (修正 OCR 错误 + 推断缺失边)
    nodes, edges ← LLM.refine_flowchart(nodes, edges, image_caption)
    return nodes, edges
```

**创新性**：把流程图从"图像"降维成"结构化子图"直接并入主图谱，让 GraphRAG 能直接检索流程节点，这是本课题区别于纯文本 GraphRAG 的关键。

### 1.5 跨模态对齐（A4）

```
function AlignCrossModal(nodes):
    # 把所有节点嵌入投影到统一空间
    projected ← {v: project(v.embedding) for v in nodes}

    edges ← []
    for v1, v2 in pairs(nodes):
        if v1.modality != v2.modality:    # 仅跨模态
            sim ← cosine(projected[v1], projected[v2])
            if sim > τ:                    # 阈值, 默认 0.75
                edges.append(Edge(v1, v2, SEMANTIC, weight=sim))
    return edges
```

### 1.6 社区发现（A5）

调用 Neo4j GDS 的 Leiden 算法，对每个社区用 LLM 生成 200 字摘要，存储为社区节点的 `summary` 属性。

---

## 2. 算法 B：GraphRAG 检索推理

### 2.1 总体流程

```
function GraphRAGRetrieve(query q):
    # 阶段1: 种子节点检索 (向量召回)
    q_emb ← BGE.encode(q)
    seeds ← Neo4j.vector_search(q_emb, top_k=10)

    # 阶段2: 子图扩展 (图结构推理)
    subgraph ← BFS(seeds, max_hops=2, edge_filter=高权重)

    # 阶段3: 社区摘要补充
    communities ← {v.community_id for v in subgraph.nodes}
    summaries ← [c.summary for c in communities]

    # 阶段4: LLM 重排序 (剪枝)
    relevance_scores ← LLM.score_relevance(q, subgraph.nodes)
    subgraph ← keep_top(subgraph, relevance_scores, ratio=0.6)

    # 阶段5: 结构化上下文格式化
    context ← format_as_prompt(subgraph, summaries)
    return context
```

### 2.2 子图扩展的剪枝策略

朴素的 N 跳 BFS 会爆炸。采用**加权游走采样**：

$$
P(v_j \mid v_i) = \frac{w_{ij} \cdot \text{sim}(q, v_j)}{\sum_k w_{ik} \cdot \text{sim}(q, v_k)}
$$

每个种子节点采样 $b$ 条游走路径（默认 $b=5$，长度 $L=2$），合并去重得到子图。这样既利用了图结构，又避免了组合爆炸。

### 2.3 结构化上下文格式

不同于普通 RAG 拼接文本片段，本方法把子图序列化为：

```
[Subgraph Context]
## Communities (high-level)
- Community 1: <summary>
- Community 2: <summary>

## Entities
- E1: <name> | <description>
- E2: ...

## Relations
- E1 --[references]--> E2 (weight=0.83)
- E3 --[sequential]--> E4

## Source Snippets
- S1: <text>
- S2: <text>
```

实证上 LLM 对这种 Markdown 结构化上下文的推理稳定性更高（这是你实验中要验证的点之一）。

---

## 3. 算法 C：智能体特征图与匹配 ⭐ 核心创新

### 3.1 智能体特征图建模

把每个智能体表示为多属性节点：

```python
AgentSpec(
    id="data_analyst_v1",
    name="数据分析智能体",
    capabilities=["sql_query", "data_visualization", "statistical_test"],
    tools=["pandas", "matplotlib", "scipy"],
    input_schema={"data_source": "str", "question": "str"},
    output_schema={"chart": "image", "summary": "str"},
)
```

在 Neo4j 中建模为：

```
(:Agent {id, name, embedding})
  -[:HAS_CAPABILITY]-> (:Capability {name, embedding})
  -[:USES_TOOL]-> (:Tool {name, signature})
```

### 3.2 任务分解算法（C1）

```
function DecomposeTask(q, graph_context):
    prompt ← TASK_DECOMPOSITION_PROMPT(q, graph_context)
    subtasks ← LLM.structured_output(prompt, schema=SubtaskList)
    # 每个 subtask = {id, description, required_capabilities,
    #                 required_tools, input_type, output_type, depends_on}
    return subtasks
```

**关键**：分解 prompt 必须显式告知 LLM 可用的能力词表（来自智能体特征图），避免生成无法匹配的虚构能力。

### 3.3 子任务-智能体匹配算法（C2）⭐⭐ 最核心

混合匹配，三个维度加权打分：

```
function MatchAgent(subtask, agent):
    # 维度1: 能力嵌入相似度
    cap_sim ← max_cosine(
        BGE.encode(subtask.required_capabilities),
        agent.capability_embeddings
    )

    # 维度2: 工具覆盖率 (硬约束)
    tool_coverage ← |subtask.required_tools ∩ agent.tools| / |subtask.required_tools|
    if tool_coverage < 1.0:    # 硬约束: 工具必须全覆盖
        return -inf

    # 维度3: IO schema 兼容性
    io_compat ← schema_match(subtask.input_type, agent.input_schema) *
                schema_match(subtask.output_type, agent.output_schema)

    score ← α·cap_sim + β·tool_coverage + γ·io_compat
    return score


function MatchAll(subtasks):
    assignments ← []
    for st in subtasks:
        candidates ← [(a, MatchAgent(st, a)) for a in AgentFG.agents]
        best ← argmax(candidates)
        if best.score < θ:    # 找不到合适的，触发回退
            best ← FALLBACK_AGENT
        assignments.append((st, best))
    return assignments
```

权重默认 $\alpha=0.5, \beta=0.3, \gamma=0.2$，作为消融实验变量。

### 3.4 为什么这是创新点？

现有工作流生成的角色分配大多让 LLM 在 prompt 里"自由想象"。本算法把它转化为**图匹配优化问题**：子任务图节点 ↔ 智能体特征图节点。优势有三：

1. **可解释**：每个分配都有打分依据，能在论文里画出对照表
2. **可控**：通过硬约束（工具覆盖率）防止幻觉
3. **可消融**：三个维度的权重可以做消融实验，论文有亮点

---

## 4. 算法 D：工作流生成与验证

### 4.1 依赖关系推导

```
function BuildDAG(assignments):
    nodes ← [WorkflowStep(agent=a, action=st.description, ...)
             for (st, a) in assignments]

    # 显式依赖: subtask.depends_on (来自分解)
    # 隐式依赖: 数据流分析 (前一步的 output_type ⊇ 后一步的 input_type)
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes[i+1:], start=i+1):
            if data_flow_compatible(ni, nj):
                edges.append((ni, nj))

    # 拓扑排序
    sorted_nodes ← topological_sort(nodes, edges)
    return Workflow(steps=sorted_nodes)
```

### 4.2 三层验证

| 层级 | 检查内容 | 失败处理 |
|------|---------|---------|
| 结构 | DAG 无环、无孤立节点 | 直接报错重生成 |
| 语义 | 相邻步骤 IO schema 兼容 | 插入适配步骤 |
| 完整性 | LLM-as-judge 评估覆盖度 | 反馈给生成器迭代 |

---

## 5. 算法复杂度分析

| 算法 | 时间复杂度 | 主要成本 |
|------|-----------|---------|
| A 图谱构建 | $O(\|V\| \cdot c_{LLM})$ | LLM 抽取 |
| B GraphRAG 检索 | $O(k + b \cdot L)$ | 向量检索 + 游走 |
| C 任务分解 | $O(c_{LLM})$ | 单次 LLM 调用 |
| C 智能体匹配 | $O(\|subtasks\| \cdot \|agents\|)$ | 嵌入比对 |
| D 工作流生成 | $O(\|N_W\|^2)$ | 数据流分析 |

整体瓶颈在 LLM 调用次数，因此算法设计上**尽量合并 LLM 调用**（如 1.2 中实体+摘要一次出）。

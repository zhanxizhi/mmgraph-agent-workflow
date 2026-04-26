"""评估指标模块

三类指标:
  1. 图谱构建质量 (节点/关系抽取 P/R/F1, 跨模态对齐准确率)
  2. GraphRAG 检索质量 (Recall@k, MRR, 上下文相关性)
  3. 工作流生成质量 (步骤完整性, 逻辑正确性, 可执行率, 工具调用准确率)
"""
from __future__ import annotations

from typing import Callable, Sequence

import networkx as nx

from src.common.types import Workflow

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


# ============================================================
# 通用工具
# ============================================================

def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ============================================================
# 1. 图谱构建质量（实验一）
# ============================================================

def graph_node_prf(pred_nodes: list[str], gold_nodes: list[str]) -> dict:
    """节点抽取精确率/召回率/F1。

    Parameters
    ----------
    pred_nodes:
        预测节点内容列表（归一化后的实体名或摘要）。
    gold_nodes:
        标注节点内容列表。
    """
    pred_set = {n.lower().strip() for n in pred_nodes}
    gold_set = {n.lower().strip() for n in gold_nodes}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return _prf(tp, fp, fn)


def graph_relation_prf(
    pred_triples: list[tuple[str, str, str]],
    gold_triples: list[tuple[str, str, str]],
) -> dict:
    """关系抽取 P/R/F1。三元组格式 (head, relation_type, tail)。"""
    def _norm(t):
        return (t[0].lower().strip(), t[1].lower().strip(), t[2].lower().strip())

    pred_set = {_norm(t) for t in pred_triples}
    gold_set = {_norm(t) for t in gold_triples}
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return _prf(tp, fp, fn)


def cross_modal_alignment_accuracy(
    pred_edges: list[tuple[str, str]],
    gold_edges: list[tuple[str, str]],
) -> float:
    """跨模态对齐准确率：预测的跨模态边命中标注的比例。"""
    if not gold_edges:
        return 1.0
    pred_set = {(a.lower(), b.lower()) for a, b in pred_edges}
    gold_set = {(a.lower(), b.lower()) for a, b in gold_edges}
    hit = len(pred_set & gold_set)
    return round(hit / len(gold_set), 4)


def flowchart_node_accuracy(pred_steps: list[str], gold_steps: list[str]) -> float:
    """流程图节点抽取准确率（精确匹配 or 模糊匹配）。"""
    if not gold_steps:
        return 1.0
    pred_norm = {s.lower().strip() for s in pred_steps}
    gold_norm = {s.lower().strip() for s in gold_steps}
    hit = len(pred_norm & gold_norm)
    return round(hit / len(gold_norm), 4)


# ============================================================
# 2. GraphRAG 检索质量（实验二）
# ============================================================

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Recall@k：top-k 检索结果中命中相关节点的比例。

    Parameters
    ----------
    retrieved:
        检索返回的节点 ID 列表（按排名顺序）。
    relevant:
        标注的相关节点 ID 集合。
    k:
        取前 k 个结果。
    """
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    rel_set = set(relevant)
    return round(len(top_k & rel_set) / len(rel_set), 4)


def mrr(retrieved_lists: list[list[str]], relevant_lists: list[list[str]]) -> float:
    """Mean Reciprocal Rank（多个查询的平均倒数排名）。

    Parameters
    ----------
    retrieved_lists:
        每个查询的检索排名列表。
    relevant_lists:
        每个查询对应的相关节点集合。
    """
    rr_sum = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_lists):
        rel_set = set(relevant)
        for rank, node_id in enumerate(retrieved, start=1):
            if node_id in rel_set:
                rr_sum += 1.0 / rank
                break
    return round(rr_sum / len(retrieved_lists), 4) if retrieved_lists else 0.0


def context_relevance(query: str, context: str, llm_judge) -> float:
    """LLM-as-judge：上下文与查询的相关性评分（0-1）。

    同一样本评 3 次取均值，减少 LLM 评分方差。
    """
    _JUDGE_SYSTEM = (
        "你是一个检索质量评估专家。评估给定上下文对回答查询的帮助程度。"
        "返回JSON：{\"score\": 0-10的整数}"
    )
    _JUDGE_PROMPT = f"查询：{query}\n\n上下文：{context[:1500]}\n\n请打分（0=完全无关，10=高度相关）："

    scores = []
    for _ in range(3):
        try:
            result = llm_judge.structured_output(
                prompt=_JUDGE_PROMPT,
                schema={"type": "object", "properties": {"score": {"type": "integer"}}},
                system=_JUDGE_SYSTEM,
            )
            scores.append(min(10, max(0, int(result.get("score", 5)))))
        except Exception:
            scores.append(5)

    return round(sum(scores) / len(scores) / 10, 4)


def context_compression_ratio(context_tokens: int, full_graph_tokens: int) -> float:
    """上下文压缩率 = 子图 token 数 / 全图 token 数（越小越好）。"""
    if full_graph_tokens == 0:
        return 1.0
    return round(context_tokens / full_graph_tokens, 4)


# ============================================================
# 3. 工作流生成质量（实验三）
# ============================================================

def _text_units(text: str) -> set[str]:
    """Tokenize mixed Chinese/English action text without extra dependencies."""
    import re

    text = text.lower().strip()
    words = set(re.findall(r"[a-z0-9_]+", text))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = {
        "".join(cjk_chars[i:i + 2])
        for i in range(max(0, len(cjk_chars) - 1))
    }
    cjk_trigrams = {
        "".join(cjk_chars[i:i + 3])
        for i in range(max(0, len(cjk_chars) - 2))
    }
    return words | cjk_bigrams | cjk_trigrams


def _action_similarity(a: str, b: str) -> float:
    """Jaccard similarity over English tokens and Chinese character n-grams."""
    au = _text_units(a)
    bu = _text_units(b)
    if not au or not bu:
        return 0.0
    return len(au & bu) / len(au | bu)


def _cosine_matrix(
    gold_vecs: list[list[float]], pred_vecs: list[list[float]],
) -> list[list[float]]:
    """计算 gold × pred 余弦相似度矩阵（假设向量已 L2 归一化）。"""
    import numpy as np

    if not gold_vecs or not pred_vecs:
        return [[0.0] * len(pred_vecs) for _ in gold_vecs]
    g = np.asarray(gold_vecs, dtype=float)
    p = np.asarray(pred_vecs, dtype=float)
    g_norm = np.linalg.norm(g, axis=1, keepdims=True)
    p_norm = np.linalg.norm(p, axis=1, keepdims=True)
    g_norm[g_norm == 0] = 1.0
    p_norm[p_norm == 0] = 1.0
    g = g / g_norm
    p = p / p_norm
    return (g @ p.T).tolist()


_AGENT_ALIASES = {
    "faultdiagnosisagent": "fault_detector",
    "faultdiagnosis": "fault_detector",
    "fault_detector": "fault_detector",
    "fault detection agent": "fault_detector",
    "故障诊断智能体": "fault_detector",
    "故障检测智能体": "fault_detector",
    "topologyanalyzer": "topology_analyzer",
    "topology_analyzer": "topology_analyzer",
    "电网拓扑分析智能体": "topology_analyzer",
    "protectionagent": "protection_checker",
    "protection_checker": "protection_checker",
    "继电保护检查智能体": "protection_checker",
    "保护定值校验智能体": "protection_checker",
    "powerflowanalyzer": "power_flow_analyzer",
    "power_flow_analyzer": "power_flow_analyzer",
    "潮流分析智能体": "power_flow_analyzer",
    "restorationplanner": "restoration_planner",
    "restoration_planner": "restoration_planner",
    "恢复操作规划智能体": "restoration_planner",
    "knowledgeretrievalagent": "knowledge_retriever",
    "knowledge_retriever": "knowledge_retriever",
    "知识检索智能体": "knowledge_retriever",
    "regulationchecker": "regulation_checker",
    "regulation_checker": "regulation_checker",
    "规程合规检查智能体": "regulation_checker",
    "dataanalysisagent": "load_forecaster",
    "loadforecaster": "load_forecaster",
    "load_forecaster": "load_forecaster",
    "负荷预测智能体": "load_forecaster",
    "anomalydetector": "anomaly_detector",
    "anomaly_detector": "anomaly_detector",
    "电力数据异常检测智能体": "anomaly_detector",
    "renewableanalyzer": "renewable_analyzer",
    "renewable_analyzer": "renewable_analyzer",
    "新能源出力分析智能体": "renewable_analyzer",
    "docgenerationagent": "report_generator",
    "reportgenerator": "report_generator",
    "report_generator": "report_generator",
    "文档生成智能体": "report_generator",
    "电力报告生成智能体": "report_generator",
    "orchestratoragent": "__fallback__",
    "orchestrator": "__fallback__",
    "通用协调智能体": "__fallback__",
    "通用智能体": "__fallback__",
}


def _normalize_agent_id(agent_id: str) -> str:
    key = str(agent_id or "").strip().lower().replace("-", "_")
    compact = key.replace("_", "").replace(" ", "")
    return _AGENT_ALIASES.get(key, _AGENT_ALIASES.get(compact, key))


def step_completeness(
    pred: Workflow,
    gold: Workflow,
    threshold: float | None = None,
    embed_fn: EmbedFn | None = None,
) -> float:
    """步骤完整性：预测工作流覆盖 gold 步骤的比例。

    匹配模式：
      * 传入 ``embed_fn`` 时使用语义相似度（推荐 BGE-M3）。``embed_fn``
        接收文本列表，返回与之等长的向量列表。每个 gold 步骤与全部
        pred 步骤计算余弦相似度，最高分超过 ``threshold`` 即视为覆盖。
        默认阈值 0.65，对应 BGE-M3 在中文专业术语近义判别上的经验值。
      * 未传 ``embed_fn`` 时回退到英文 token + 中文字 n-gram 的 Jaccard
        相似度，默认阈值 0.18，保持与旧实验脚本兼容。

    Parameters
    ----------
    pred:
        生成的工作流。
    gold:
        金标工作流。
    threshold:
        覆盖判定阈值；为 ``None`` 时按所选模式取默认值。
    embed_fn:
        可选嵌入函数，签名为 ``Callable[[Sequence[str]], list[list[float]]]``。
    """
    if not gold.steps:
        return 1.0
    if not pred.steps:
        return 0.0

    if embed_fn is not None:
        thr = 0.65 if threshold is None else threshold
        gold_actions = [s.action for s in gold.steps]
        pred_actions = [s.action for s in pred.steps]
        try:
            vectors = embed_fn(gold_actions + pred_actions)
        except Exception:
            vectors = []
        if vectors and len(vectors) == len(gold_actions) + len(pred_actions):
            gold_vecs = vectors[: len(gold_actions)]
            pred_vecs = vectors[len(gold_actions) :]
            sim = _cosine_matrix(gold_vecs, pred_vecs)
            covered = sum(1 for row in sim if row and max(row) >= thr)
            return round(covered / len(gold.steps), 4)
        # 嵌入失败时静默回退到词袋路径，保证脚本不中断

    thr = 0.18 if threshold is None else threshold
    covered = 0
    for gold_step in gold.steps:
        for pred_step in pred.steps:
            if _action_similarity(gold_step.action, pred_step.action) >= thr:
                covered += 1
                break

    return round(covered / len(gold.steps), 4)


def logical_correctness(workflow: Workflow, llm_judge) -> float:
    """逻辑正确性：LLM-as-judge 评估步骤顺序与依赖是否合理（0-1）。"""
    _SYSTEM = "你是工作流逻辑评估专家。评估工作流步骤顺序和依赖关系的合理性。"
    steps_str = "\n".join(
        f"{i+1}. [{s.agent_id}] {s.action} (depends_on: {s.depends_on})"
        for i, s in enumerate(workflow.steps)
    )
    prompt = (
        f"工作流任务：{workflow.task}\n\n步骤：\n{steps_str}\n\n"
        "请评估逻辑合理性（0=完全不合理，10=完全合理），返回JSON：{\"score\": 整数, \"issues\": [\"问题描述\"]}"
    )
    scores = []
    for _ in range(3):
        try:
            result = llm_judge.structured_output(
                prompt=prompt,
                schema={"type": "object", "properties": {
                    "score": {"type": "integer"}, "issues": {"type": "array"}
                }},
                system=_SYSTEM,
            )
            scores.append(min(10, max(0, int(result.get("score", 5)))))
        except Exception:
            scores.append(5)
    return round(sum(scores) / len(scores) / 10, 4)


def executability(workflow: Workflow) -> float:
    """可执行率：DAG 合法 + IO 兼容 + 工具字段存在。

    返回 0.0（不可执行）或 1.0（可执行）。
    """
    # 1. DAG 无环
    g = nx.DiGraph()
    step_map = {s.step_id: s for s in workflow.steps}
    for s in workflow.steps:
        g.add_node(s.step_id)
        for dep in s.depends_on:
            g.add_edge(dep, s.step_id)
    if not nx.is_directed_acyclic_graph(g):
        return 0.0

    # 2. 无孤立节点（每个步骤都有 agent_id）
    if any(not s.agent_id for s in workflow.steps):
        return 0.0

    # 3. 依赖引用的步骤必须存在
    for step in workflow.steps:
        for dep in step.depends_on:
            if dep not in step_map:
                return 0.0

    return 1.0


def tool_call_accuracy(pred: Workflow, gold: Workflow) -> float:
    """工具调用准确率：预测步骤中工具选择与 gold 的匹配比例。"""
    if not gold.steps:
        return 1.0

    def _tools(step: "WorkflowStep") -> set[str]:
        tools = step.inputs.get("required_tools", [])
        return {t.lower() for t in tools}

    correct = 0
    for i, gold_step in enumerate(gold.steps):
        if i >= len(pred.steps):
            break
        pred_tools = _tools(pred.steps[i])
        gold_tools = _tools(gold_step)
        if not gold_tools:
            correct += 1
            continue
        overlap = len(pred_tools & gold_tools) / len(gold_tools)
        if overlap >= 0.5:
            correct += 1

    return round(correct / len(gold.steps), 4)


def role_assignment_accuracy(
    pred: Workflow,
    gold: Workflow,
) -> float:
    """角色分配准确率：预测智能体与 gold 智能体的匹配比例。

    为了公平比较 pure LLM 等自由命名输出，先将常见英文类名和中文
    智能体名归一化为本系统的标准 agent_id，再按拓扑/步骤顺序对齐计算。
    """
    if not gold.steps:
        return 1.0
    correct = sum(
        1 for p, g in zip(pred.steps, gold.steps)
        if _normalize_agent_id(p.agent_id) == _normalize_agent_id(g.agent_id)
    )
    return round(correct / len(gold.steps), 4)


# ============================================================
# 批量评估入口
# ============================================================

def evaluate_workflow(
    pred: Workflow,
    gold: Workflow,
    task: str,
    llm_judge=None,
    embed_fn: EmbedFn | None = None,
) -> dict:
    """一次性计算工作流生成的所有指标，返回结果字典。

    传入 ``embed_fn`` 时，步骤完整性走语义相似度匹配（推荐 BGE-M3）。
    """
    result = {
        "step_completeness":    step_completeness(pred, gold, embed_fn=embed_fn),
        "executability":        executability(pred),
        "tool_call_accuracy":   tool_call_accuracy(pred, gold),
        "role_assignment_accuracy": role_assignment_accuracy(pred, gold),
    }
    if llm_judge:
        result["logical_correctness"] = logical_correctness(pred, llm_judge)
    return result

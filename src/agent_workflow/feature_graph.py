"""智能体特征图与匹配模块（算法 C）

核心创新：把子任务-智能体分配转化为图匹配优化问题
匹配维度：能力嵌入相似度(α) + 工具覆盖率(β) + IO schema 兼容性(γ)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel

from src.common.types import AgentSpec, EdgeType, NodeType


# ============================================================
# 任务分解 LLM Schema
# ============================================================

class _Subtask(BaseModel):
    id: str
    description: str
    required_capabilities: list[str]
    required_tools: list[str]
    input_type: str | list = ""   # LLM 有时返回 list，宽松接收
    output_type: str | list = ""
    depends_on: list[str] = []

    def model_post_init(self, __context):
        # 若 LLM 返回 list，拼成逗号字符串
        if isinstance(self.input_type, list):
            self.input_type = ",".join(str(x) for x in self.input_type)
        if isinstance(self.output_type, list):
            self.output_type = ",".join(str(x) for x in self.output_type)


class _SubtaskList(BaseModel):
    subtasks: list[_Subtask]


_DECOMPOSE_SYSTEM = """\
你是一个智能体工作流编排专家。根据任务描述和知识图谱上下文，把任务分解为若干个子任务。
每个子任务必须：
1. 从给定的能力词表中选择所需能力（required_capabilities）
2. 从给定的工具列表中选择所需工具（required_tools）
3. 声明输入输出数据类型（input_type / output_type）
4. 声明依赖关系（depends_on 列表，引用其他子任务的 id）
"""

_DECOMPOSE_PROMPT = """\
任务描述：{task}

知识图谱上下文：
{context}

可用能力词表（required_capabilities 必须从此列表中选择）：
{capability_vocab}

可用工具列表（required_tools 必须从此列表中选择）：
{tool_vocab}

请将任务分解为 3-6 个有依赖关系的子任务，返回 JSON。
"""

_FALLBACK_AGENT_ID = "__fallback__"


# ============================================================
# 匹配打分辅助
# ============================================================

def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _tool_coverage(required: list[str], available: list[str]) -> float:
    if not required:
        return 1.0
    available_set = {t.lower() for t in available}
    covered = sum(1 for t in required if t.lower() in available_set)
    return covered / len(required)


def _io_compat(
    subtask_in: str, subtask_out: str,
    agent_in: dict[str, Any], agent_out: dict[str, Any],
) -> float:
    """简单 IO 兼容性：子任务输入/输出类型是否出现在 agent schema key 中。"""
    def _match(dtype: str, schema: dict) -> float:
        dtype = dtype.lower()
        for key in schema:
            if dtype in key.lower() or key.lower() in dtype:
                return 1.0
        return 0.5  # 未知类型给中等分，不作为硬拒绝

    in_score  = _match(subtask_in,  agent_in)
    out_score = _match(subtask_out, agent_out)
    return (in_score + out_score) / 2


# ============================================================
# AgentFeatureGraph
# ============================================================

class AgentFeatureGraph:
    """智能体特征图：维护所有可用智能体及其能力/工具信息。"""

    def __init__(self):
        self.agents: dict[str, AgentSpec] = {}
        # 注册一个兜底智能体，匹配失败时使用
        self._register_fallback()

    def _register_fallback(self) -> None:
        self.agents[_FALLBACK_AGENT_ID] = AgentSpec(
            id=_FALLBACK_AGENT_ID,
            name="通用智能体",
            description="无专用智能体时的兜底，通过 LLM 直接执行子任务",
            capabilities=["general_purpose"],
            tools=[],
            input_schema={"input": "any"},
            output_schema={"output": "any"},
        )

    def register(self, agent: AgentSpec) -> None:
        self.agents[agent.id] = agent

    def get_capability_vocab(self) -> list[str]:
        """返回所有注册智能体能力的去重词表，用于约束任务分解 prompt。"""
        vocab: set[str] = set()
        for agent in self.agents.values():
            vocab.update(agent.capabilities)
        return sorted(vocab)

    def get_tool_vocab(self) -> list[str]:
        """返回所有注册智能体工具的去重词表。"""
        vocab: set[str] = set()
        for agent in self.agents.values():
            vocab.update(agent.tools)
        return sorted(vocab)

    def to_neo4j(self, driver) -> None:
        """持久化智能体特征图到 Neo4j。"""
        with driver.session() as session:
            for agent in self.agents.values():
                if agent.id == _FALLBACK_AGENT_ID:
                    continue
                # Agent 节点
                session.run(
                    """
                    MERGE (a:AgentNode {node_id: $id})
                    SET a.content = $name,
                        a.node_type = 'agent',
                        a.description = $desc,
                        a.tools = $tools,
                        a.input_schema = $in_schema,
                        a.output_schema = $out_schema
                    """,
                    id=agent.id, name=agent.name, desc=agent.description,
                    tools=agent.tools,
                    in_schema=json.dumps(agent.input_schema),
                    out_schema=json.dumps(agent.output_schema),
                )
                # Capability 节点 + HAS_CAPABILITY 边
                for cap in agent.capabilities:
                    session.run(
                        """
                        MERGE (c:CapabilityNode {node_id: $cap_id})
                        SET c.content = $cap, c.node_type = 'capability'
                        WITH c
                        MATCH (a:AgentNode {node_id: $agent_id})
                        MERGE (a)-[:HAS_CAPABILITY]->(c)
                        """,
                        cap_id=f"cap_{cap}", cap=cap, agent_id=agent.id,
                    )
                # Tool 节点 + USES_TOOL 边
                for tool in agent.tools:
                    session.run(
                        """
                        MERGE (t:ToolNode {node_id: $tool_id})
                        SET t.content = $tool, t.node_type = 'tool'
                        WITH t
                        MATCH (a:AgentNode {node_id: $agent_id})
                        MERGE (a)-[:USES_TOOL]->(t)
                        """,
                        tool_id=f"tool_{tool}", tool=tool, agent_id=agent.id,
                    )


# ============================================================
# AgentMatcher（算法 C1 + C2）
# ============================================================

class AgentMatcher:
    """子任务-智能体混合匹配器。

    匹配得分 = α·cap_sim + β·tool_coverage + γ·io_compat
    工具覆盖率 < 1.0 时拒绝该智能体（硬约束）。
    """

    def __init__(
        self,
        feature_graph: AgentFeatureGraph,
        llm_client,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        tool_coverage_threshold: float = 0.0,
    ):
        """
        Parameters
        ----------
        alpha, beta, gamma:
            三维度权重（论文消融实验变量），默认 0.5/0.3/0.2。
        tool_coverage_threshold:
            工具覆盖率硬约束阈值，低于此值直接拒绝。
            默认 0.0（纯软约束），消融实验时可设为 0.5/1.0。
        """
        self.fg = feature_graph
        self.llm = llm_client
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tool_threshold = tool_coverage_threshold

    # ------------------------------------------------------------------
    # 算法 C1：任务分解
    # ------------------------------------------------------------------

    def decompose_task(self, task: str, graph_context: str) -> list[dict]:
        """LLM 基于 GraphRAG 上下文分解任务为子任务列表。

        Returns
        -------
        list of subtask dicts, each with keys:
            id, description, required_capabilities, required_tools,
            input_type, output_type, depends_on
        """
        cap_vocab = "\n".join(f"- {c}" for c in self.fg.get_capability_vocab())
        tool_vocab = "\n".join(f"- {t}" for t in self.fg.get_tool_vocab())

        prompt = _DECOMPOSE_PROMPT.format(
            task=task,
            context=graph_context[:2000],  # 截断避免超长
            capability_vocab=cap_vocab,
            tool_vocab=tool_vocab,
        )

        result = self.llm.structured_output(
            prompt=prompt,
            schema=_SubtaskList,
            system=_DECOMPOSE_SYSTEM,
        )

        return result.get("subtasks", [])

    # ------------------------------------------------------------------
    # 算法 C2：子任务-智能体匹配
    # ------------------------------------------------------------------

    def _score(self, subtask: dict, agent: AgentSpec) -> float:
        """对单个 (subtask, agent) 对计算匹配得分。不满足硬约束返回 -inf。"""
        # 工具覆盖率硬约束
        required_tools = subtask.get("required_tools", [])
        tc = _tool_coverage(required_tools, agent.tools)
        if tc < self.tool_threshold and required_tools:
            return float("-inf")

        # 能力嵌入相似度
        if agent.embedding and subtask.get("_cap_emb"):
            cap_sim = _cosine(subtask["_cap_emb"], agent.embedding)
        else:
            # 无嵌入时用能力关键词匹配的 Jaccard 相似度
            req_caps  = set(c.lower() for c in subtask.get("required_capabilities", []))
            agent_caps = set(c.lower() for c in agent.capabilities)
            cap_sim = len(req_caps & agent_caps) / len(req_caps | agent_caps) if (req_caps | agent_caps) else 0.0

        # IO 兼容性
        io_s = _io_compat(
            subtask.get("input_type", ""),
            subtask.get("output_type", ""),
            agent.input_schema,
            agent.output_schema,
        )

        return self.alpha * cap_sim + self.beta * tc + self.gamma * io_s

    def match_subtask_to_agent(self, subtask: dict) -> tuple[AgentSpec, float]:
        """为单个子任务找最优智能体，返回 (agent, score)。"""
        best_agent: AgentSpec | None = None
        best_score = float("-inf")

        for agent in self.fg.agents.values():
            if agent.id == _FALLBACK_AGENT_ID:
                continue
            s = self._score(subtask, agent)
            if s > best_score:
                best_score = s
                best_agent = agent

        # 无合适智能体时启用兜底
        if best_agent is None or best_score == float("-inf"):
            return self.fg.agents[_FALLBACK_AGENT_ID], 0.0

        return best_agent, best_score

    def match_all(self, subtasks: list[dict]) -> list[tuple[dict, AgentSpec, float]]:
        """批量匹配，返回 [(subtask, agent, score), ...]。"""
        results = []
        for st in subtasks:
            agent, score = self.match_subtask_to_agent(st)
            results.append((st, agent, score))
        return results

    def embed_subtask_capabilities(self, subtasks: list[dict]) -> list[dict]:
        """为子任务的 required_capabilities 生成嵌入，加速后续余弦计算。"""
        texts = [" ".join(st.get("required_capabilities", [])) for st in subtasks]
        embeddings = self.llm.embed(texts)
        for st, emb in zip(subtasks, embeddings):
            st["_cap_emb"] = emb
        return subtasks

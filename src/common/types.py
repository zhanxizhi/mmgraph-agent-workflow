"""通用数据类型定义：节点、边、图、智能体、工作流"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    """图谱节点类型"""
    TEXT = "text"           # 文本片段节点
    ENTITY = "entity"       # 实体节点 (从文本抽取)
    IMAGE = "image"         # 图像节点
    FLOWCHART_STEP = "flowchart_step"   # 流程图节点
    TOOL = "tool"           # 工具节点
    AGENT = "agent"         # 智能体节点
    CAPABILITY = "capability"           # 能力节点


class EdgeType(str, Enum):
    """图谱边类型"""
    SEMANTIC = "semantic"       # 语义相关
    SEQUENTIAL = "sequential"   # 流程顺序 (流程图)
    REFERENCES = "references"   # 引用 (图像引用文本等)
    HAS_CAPABILITY = "has_capability"   # 智能体->能力
    USES_TOOL = "uses_tool"     # 智能体->工具
    BELONGS_TO = "belongs_to"   # 节点->社区


@dataclass
class GraphNode:
    id: str
    type: NodeType
    content: str                        # 文本/描述
    embedding: list[float] | None = None
    modality: str = "text"              # text | image | mixed
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSpec:
    """智能体特征图节点"""
    id: str
    name: str
    description: str
    capabilities: list[str]             # 能力标签
    tools: list[str]                    # 可用工具
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    embedding: list[float] | None = None


@dataclass
class WorkflowStep:
    step_id: str
    agent_id: str
    action: str
    inputs: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Workflow:
    """最终输出的工作流 (DAG)"""
    task: str
    steps: list[WorkflowStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dag_json(self) -> dict:
        return {
            "task": self.task,
            "nodes": [s.__dict__ for s in self.steps],
            "metadata": self.metadata,
        }

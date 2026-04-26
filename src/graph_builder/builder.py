"""多模态图谱构建模块

输入: 文本文档 / 图像 / 流程图截图
输出: 多模态知识图谱 (写入 Neo4j)

流水线:
  1. 模态分流 (text/image/flowchart)
  2. 节点抽取 (LLM/CLIP/OCR)
  3. 跨模态对齐
  4. 关系抽取
  5. 社区发现 (Leiden)
  6. 写入 Neo4j
"""
from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from src.common.types import EdgeType, GraphEdge, GraphNode, NodeType


_NODE_TYPE_TO_LABEL: dict[NodeType, str] = {
    NodeType.TEXT: "TextNode",
    NodeType.ENTITY: "EntityNode",
    NodeType.IMAGE: "ImageNode",
    NodeType.FLOWCHART_STEP: "FlowchartNode",
    NodeType.AGENT: "AgentNode",
    NodeType.CAPABILITY: "CapabilityNode",
    NodeType.TOOL: "ToolNode",
}

# ============================================================
# LLM 抽取用的 Pydantic Schema
# ============================================================

class _Entity(BaseModel):
    name: str
    type: str           # 实体类型，如 "tool", "concept", "component"
    description: str    # 一句话描述


class _ChunkExtraction(BaseModel):
    summary: str             # 该文本块的 50 字以内摘要
    entities: list[_Entity]  # 抽取到的实体列表


# ============================================================
# 分块工具
# ============================================================

def _chunk_text(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    """按字符数滑动窗口分块，保留 overlap 个字符的重叠。"""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


def _node_id(content: str, prefix: str = "") -> str:
    """基于内容生成稳定 ID（相同内容不重复写入 Neo4j）。"""
    digest = hashlib.md5(content.encode()).hexdigest()[:12]
    return f"{prefix}{digest}"


# ============================================================
# LLM 抽取 Prompt
# ============================================================

_ENTITY_EXTRACTION_SYSTEM = (
    "你是一个知识图谱构建专家。从给定文本中抽取关键实体并生成摘要。\n"
    "实体类型包括：tool（工具/软件）、concept（概念）、component（系统组件）、"
    "process（流程步骤）、metric（指标）、other（其他）。\n"
    "摘要不超过50字，实体名称使用文中原文。"
)

_ENTITY_EXTRACTION_PROMPT = """\
请分析以下文本，抽取实体并生成摘要：

<text>
{text}
</text>

要求：
1. summary：50字以内的核心内容摘要
2. entities：文中出现的关键实体（3-8个），每个实体给出 name、type、description
"""


# ============================================================
# MultimodalGraphBuilder
# ============================================================

class MultimodalGraphBuilder:
    def __init__(self, llm_client, neo4j_driver=None, vision_encoder=None, ocr_engine=None):
        """
        Parameters
        ----------
        llm_client:
            LLMClient 实例，用于文本抽取。
        neo4j_driver:
            neo4j.GraphDatabase.driver 实例，为 None 时跳过写库（纯内存模式）。
        vision_encoder:
            视觉编码器（第二阶段实现），当前可传 None。
        ocr_engine:
            OCR 引擎（第二阶段实现），当前可传 None。
        """
        self.llm = llm_client
        self.db = neo4j_driver
        self.vision = vision_encoder
        self.ocr = ocr_engine

    # ------------------------------------------------------------------
    # 1. 文本节点抽取（算法 A1）
    # ------------------------------------------------------------------

    def extract_text_nodes(
        self,
        doc: str,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        source_meta: dict | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """从文本文档抽取节点和边。

        流程：分块 → LLM 同时输出摘要+实体 → 构建 TEXT 节点 + ENTITY 节点 + REFERENCES 边

        Parameters
        ----------
        doc:
            原始文本内容。
        chunk_size / chunk_overlap:
            分块参数（字符数）。
        source_meta:
            可选元数据，写入节点 metadata 字段（如文件名、场景）。

        Returns
        -------
        (nodes, edges)
        """
        chunks = _chunk_text(doc, size=chunk_size, overlap=chunk_overlap)
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        meta = source_meta or {}

        for i, chunk in enumerate(chunks):
            # --- LLM 抽取 ---
            prompt = _ENTITY_EXTRACTION_PROMPT.format(text=chunk)
            result: dict = self.llm.structured_output(
                prompt=prompt,
                schema=_ChunkExtraction,
                system=_ENTITY_EXTRACTION_SYSTEM,
            )

            summary = result.get("summary", chunk[:50])
            entities = result.get("entities", [])

            # --- TEXT（chunk）节点 ---
            chunk_node = GraphNode(
                id=_node_id(chunk, prefix="text_"),
                type=NodeType.TEXT,
                content=summary,
                modality="text",
                metadata={**meta, "chunk_index": i, "raw_length": len(chunk)},
            )
            nodes.append(chunk_node)

            # --- ENTITY 节点 + REFERENCES 边 ---
            for ent in entities:
                name = ent.get("name", "").strip()
                if not name:
                    continue
                ent_node = GraphNode(
                    id=_node_id(name, prefix="ent_"),
                    type=NodeType.ENTITY,
                    content=name,
                    modality="text",
                    metadata={
                        "entity_type": ent.get("type", "other"),
                        "description": ent.get("description", ""),
                    },
                )
                nodes.append(ent_node)
                edges.append(GraphEdge(
                    source=chunk_node.id,
                    target=ent_node.id,
                    type=EdgeType.REFERENCES,
                    weight=1.0,
                ))

        return nodes, edges

    # ------------------------------------------------------------------
    # 2. 图像节点抽取（算法 A2）—— 第二阶段实现
    # ------------------------------------------------------------------

    def extract_image_nodes(self, image_path: str) -> list[GraphNode]:
        raise NotImplementedError("图像节点抽取在第二阶段实现")

    # ------------------------------------------------------------------
    # 3. 流程图解析（算法 A3）—— 第二阶段实现
    # ------------------------------------------------------------------

    def extract_flowchart_nodes(self, image_path: str) -> tuple[list[GraphNode], list[GraphEdge]]:
        raise NotImplementedError("流程图解析在第二阶段实现")

    # ------------------------------------------------------------------
    # 4. 跨模态对齐（算法 A4）—— 第二阶段实现（图像接入后再做）
    # ------------------------------------------------------------------

    def align_cross_modal(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        raise NotImplementedError("跨模态对齐在第二阶段实现")

    # ------------------------------------------------------------------
    # 5. 社区发现（算法 A5）
    # ------------------------------------------------------------------

    def detect_communities(self) -> dict[str, int]:
        """调用 Neo4j GDS Leiden 算法，返回 {node_id: community_id}。"""
        if self.db is None:
            raise RuntimeError("detect_communities 需要 Neo4j 连接")
        with self.db.session() as session:
            # 投影图（仅文本和实体节点之间的关系）
            session.run("""
                CALL gds.graph.project(
                    'mmgraph',
                    ['TextNode', 'EntityNode'],
                    { REFERENCES: { orientation: 'UNDIRECTED' } }
                )
            """)
            result = session.run("""
                CALL gds.leiden.write('mmgraph', {
                    writeProperty: 'community_id',
                    randomSeed: 42
                })
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """)
            record = result.single()
            print(f"[communities] count={record['communityCount']}, modularity={record['modularity']:.4f}")
        # 读回 community_id 映射
        with self.db.session() as session:
            rows = session.run("MATCH (n) WHERE n.community_id IS NOT NULL RETURN n.node_id AS id, n.community_id AS cid")
            return {r["id"]: r["cid"] for r in rows}

    def generate_community_summaries(self, community_map: dict[str, int]) -> dict[int, str]:
        """为每个社区用 LLM 生成 200 字摘要，存为社区节点 summary 属性。"""
        if self.db is None:
            raise RuntimeError("需要 Neo4j 连接")

        # 按社区 ID 收集节点内容
        from collections import defaultdict
        community_contents: dict[int, list[str]] = defaultdict(list)
        with self.db.session() as session:
            rows = session.run("""
                MATCH (n) WHERE n.community_id IS NOT NULL
                RETURN n.community_id AS cid, n.content AS content
            """)
            for r in rows:
                community_contents[r["cid"]].append(r["content"])

        summaries: dict[int, str] = {}
        for cid, contents in community_contents.items():
            joined = "\n".join(contents[:20])
            summaries[cid] = self.llm.chat_simple(
                f"请用不超过200字总结以下知识图谱社区的主题和内容：\n\n{joined}",
            )

        # 批量写回 Neo4j（单次 session）
        with self.db.session() as session:
            for cid, summary in summaries.items():
                session.run(
                    "MATCH (n) WHERE n.community_id = $cid SET n.community_summary = $summary",
                    cid=cid, summary=summary,
                )

        return summaries

    # ------------------------------------------------------------------
    # 6. Neo4j 写入
    # ------------------------------------------------------------------

    def _write_nodes(self, session, nodes: list[GraphNode]) -> None:
        """MERGE 节点（幂等写入，相同 node_id 不重复创建）。每节点一次查询。"""
        for node in nodes:
            label = _NODE_TYPE_TO_LABEL.get(node.type, "GraphNode")
            session.run(
                f"""
                MERGE (n:{label} {{node_id: $node_id}})
                SET n.content = $content,
                    n.modality = $modality,
                    n.node_type = $node_type,
                    n.metadata = $metadata,
                    n.embedding = CASE WHEN $embedding IS NOT NULL THEN $embedding ELSE n.embedding END
                """,
                node_id=node.id,
                content=node.content,
                modality=node.modality,
                node_type=node.type.value,
                metadata=json.dumps(node.metadata, ensure_ascii=False),
                embedding=node.embedding,
            )

    def _write_edges(self, session, edges: list[GraphEdge]) -> None:
        """MERGE 边（幂等写入）。按边类型分组，每种类型一次 UNWIND 批量写入。"""
        _REL_MAP = {
            EdgeType.SEMANTIC: "SEMANTIC",
            EdgeType.SEQUENTIAL: "SEQUENTIAL",
            EdgeType.REFERENCES: "REFERENCES",
            EdgeType.HAS_CAPABILITY: "HAS_CAPABILITY",
            EdgeType.USES_TOOL: "USES_TOOL",
            EdgeType.BELONGS_TO: "BELONGS_TO",
        }
        # 按关系类型分桶
        buckets: dict[str, list[dict]] = {}
        for edge in edges:
            rel_type = _REL_MAP.get(edge.type, "RELATED")
            buckets.setdefault(rel_type, []).append(
                {"src": edge.source, "dst": edge.target, "weight": edge.weight}
            )
        for rel_type, rows in buckets.items():
            session.run(
                f"""
                UNWIND $rows AS row
                MATCH (a {{node_id: row.src}}), (b {{node_id: row.dst}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.weight = row.weight
                """,
                rows=rows,
            )

    def _ensure_vector_index(self, session, dim: int = 1024) -> None:
        """创建向量索引（已存在则跳过）。"""
        session.run(f"""
            CREATE VECTOR INDEX mmgraph_embedding IF NOT EXISTS
            FOR (n:TextNode) ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {dim},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

    def write_to_neo4j(self, nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
        """把节点和边写入 Neo4j（幂等，可重复调用）。"""
        if self.db is None:
            print("[builder] Neo4j 未连接，跳过写库")
            return
        with self.db.session() as session:
            self._ensure_vector_index(session)
            self._write_nodes(session, nodes)
            self._write_edges(session, edges)

    # ------------------------------------------------------------------
    # 7. 嵌入生成
    # ------------------------------------------------------------------

    def embed_nodes(self, nodes: list[GraphNode]) -> list[GraphNode]:
        """批量为节点生成嵌入向量（原地修改并返回）。"""
        texts = [n.content for n in nodes]
        embeddings = self.llm.embed(texts)
        for node, emb in zip(nodes, embeddings):
            node.embedding = emb
        return nodes

    # ------------------------------------------------------------------
    # 8. 主入口（算法 A）
    # ------------------------------------------------------------------

    def build(
        self,
        sources: list[dict],
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        embed: bool = True,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        """从多模态源构建图谱并写入 Neo4j。

        Parameters
        ----------
        sources:
            数据源列表，每项格式：
            - {"type": "text", "content": "...", "meta": {...}}
            - {"type": "text", "path": "/path/to/file.txt", "meta": {...}}
            - {"type": "image", "path": "..."}        （第二阶段）
            - {"type": "flowchart", "path": "..."}    （第二阶段）
        embed:
            是否为节点生成嵌入向量（写入向量索引时需要为 True）。
        """
        all_nodes: list[GraphNode] = []
        all_edges: list[GraphEdge] = []

        for source in sources:
            src_type = source.get("type", "text")
            meta = source.get("meta", {})

            if src_type == "text":
                content = source.get("content") or _read_file(source.get("path", ""))
                nodes, edges = self.extract_text_nodes(
                    content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    source_meta=meta,
                )
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            elif src_type == "image":
                nodes = self.extract_image_nodes(source["path"])
                all_nodes.extend(nodes)

            elif src_type == "flowchart":
                nodes, edges = self.extract_flowchart_nodes(source["path"])
                all_nodes.extend(nodes)
                all_edges.extend(edges)

            else:
                print(f"[builder] 未知 source type: {src_type}，跳过")

        # 去重（同一个实体可能从多个 chunk 里抽出来）
        all_nodes = _dedup_nodes(all_nodes)

        # 生成嵌入
        if embed:
            print(f"[builder] 正在为 {len(all_nodes)} 个节点生成嵌入...")
            self.embed_nodes(all_nodes)

        # 写入 Neo4j
        self.write_to_neo4j(all_nodes, all_edges)

        print(f"[builder] 完成：{len(all_nodes)} 节点，{len(all_edges)} 边")
        return all_nodes, all_edges


# ============================================================
# 工具函数
# ============================================================

def _read_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _dedup_nodes(nodes: list[GraphNode]) -> list[GraphNode]:
    """按 node.id 去重，保留第一次出现的节点。"""
    seen: set[str] = set()
    result: list[GraphNode] = []
    for n in nodes:
        if n.id not in seen:
            seen.add(n.id)
            result.append(n)
    return result

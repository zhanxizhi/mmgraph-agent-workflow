"""GraphRAG 检索推理模块

输入: 自然语言任务描述
输出: 结构化上下文 (子图 + 社区摘要)

策略: 混合检索
  1. 语义检索: 任务嵌入 vs 节点嵌入 (top-k 种子节点)
  2. 图结构扩展: 加权游走采样 N 跳邻居
  3. 社区摘要: 命中社区的预生成摘要
  4. LLM 重排序: 过滤无关节点
"""
from __future__ import annotations

from neo4j.exceptions import Neo4jError

from src.common.types import GraphNode, NodeType


# ============================================================
# 结构化上下文格式（算法设计文档 2.3 节）
# ============================================================

_CONTEXT_TEMPLATE = """\
[Subgraph Context]
## Communities (high-level)
{communities}

## Entities
{entities}

## Relations
{relations}

## Source Snippets
{snippets}
"""

_RERANK_SYSTEM = (
    "你是一个知识图谱检索质量评估专家。"
    "对给定的图节点与查询的相关性打分（0-10整数），只输出JSON。"
)

_RERANK_PROMPT = """\
查询: {query}

以下是图谱节点，请为每个节点的相关性打分（0=完全无关，10=高度相关）：

{nodes_text}

请按照以下格式返回JSON（key为节点ID，value为0-10的整数）：
{{"node_id_1": 8, "node_id_2": 3, ...}}
"""


# ============================================================
# GraphRAGRetriever
# ============================================================

class GraphRAGRetriever:
    def __init__(
        self,
        neo4j_driver,
        llm_client,
        top_k: int = 10,
        max_hops: int = 2,
        rerank_ratio: float = 0.6,
        use_community_summary: bool = True,
    ):
        """
        Parameters
        ----------
        neo4j_driver:
            neo4j.GraphDatabase.driver 实例。
        llm_client:
            LLMClient 实例。
        top_k:
            语义搜索返回的种子节点数。
        max_hops:
            子图扩展跳数。
        rerank_ratio:
            LLM 重排序后保留节点的比例。
        use_community_summary:
            是否在上下文中加入社区摘要。
        """
        self.db = neo4j_driver
        self.llm = llm_client
        self.top_k = top_k
        self.max_hops = max_hops
        self.rerank_ratio = rerank_ratio
        self.use_community_summary = use_community_summary

    # ------------------------------------------------------------------
    # 阶段 1：语义搜索种子节点
    # ------------------------------------------------------------------

    def semantic_search(self, query: str, k: int | None = None) -> list[GraphNode]:
        """向量检索：返回与 query 最相似的 top-k 节点。"""
        k = k or self.top_k
        query_emb = self.llm.embed_one(query)

        with self.db.session() as session:
            result = session.run(
                """
                CALL db.index.vector.queryNodes('mmgraph_embedding', $k, $embedding)
                YIELD node, score
                RETURN node.node_id AS node_id,
                       node.content AS content,
                       node.node_type AS node_type,
                       node.modality AS modality,
                       node.community_id AS community_id,
                       score
                ORDER BY score DESC
                """,
                k=k,
                embedding=query_emb,
            )
            nodes = []
            for r in result:
                try:
                    node_type = NodeType(r["node_type"])
                except (ValueError, KeyError):
                    node_type = NodeType.TEXT
                node = GraphNode(
                    id=r["node_id"],
                    type=node_type,
                    content=r["content"] or "",
                    modality=r["modality"] or "text",
                    metadata={
                        "community_id": r["community_id"],
                        "score": r["score"],
                    },
                )
                nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    # 阶段 2：子图扩展
    # ------------------------------------------------------------------

    def expand_subgraph(self, seeds: list[GraphNode], hops: int | None = None) -> dict:
        """从种子节点出发做 N 跳 BFS 扩展，优先用 APOC，降级用纯 Cypher。"""
        hops = hops or self.max_hops
        seed_ids = [n.id for n in seeds]
        if not seed_ids:
            return {"nodes": [], "edges": []}

        try:
            return self._expand_apoc(seed_ids, hops)
        except Neo4jError:
            print("[retriever] APOC 不可用，降级为简单 BFS")
            return self._expand_simple(seed_ids, seeds, hops)

    def _expand_apoc(self, seed_ids: list[str], hops: int) -> dict:
        """使用 APOC subgraphAll 扩展（含边数据）。"""
        with self.db.session() as session:
            result = session.run(
                """
                MATCH (seed) WHERE seed.node_id IN $seed_ids
                CALL apoc.path.subgraphAll(seed, {
                    maxLevel: $hops,
                    relationshipFilter: 'REFERENCES|SEMANTIC|SEQUENTIAL'
                })
                YIELD nodes, relationships
                RETURN nodes, relationships
                """,
                seed_ids=seed_ids,
                hops=hops,
            )
            all_nodes: dict[str, dict] = {}
            all_edges: list[dict] = []
            for record in result:
                for node in record["nodes"]:
                    nid = node.get("node_id")
                    if nid:
                        all_nodes[nid] = {
                            "id": nid,
                            "content": node.get("content", ""),
                            "node_type": node.get("node_type", ""),
                            "community_id": node.get("community_id"),
                            "community_summary": node.get("community_summary"),
                        }
                for rel in record["relationships"]:
                    all_edges.append({
                        "source": rel.start_node.get("node_id"),
                        "target": rel.end_node.get("node_id"),
                        "type": rel.type,
                        "weight": rel.get("weight", 1.0),
                    })
        return {"nodes": list(all_nodes.values()), "edges": all_edges}

    def _expand_simple(self, seed_ids: list[str], seeds: list[GraphNode], hops: int) -> dict:
        """纯 Cypher BFS 降级方案。"""
        with self.db.session() as session:
            result = session.run(
                """
                MATCH (seed)-[r*1..$hops]-(neighbor)
                WHERE seed.node_id IN $seed_ids
                  AND neighbor.node_id IS NOT NULL
                RETURN DISTINCT neighbor.node_id AS nid,
                               neighbor.content AS content,
                               neighbor.node_type AS node_type,
                               neighbor.community_id AS community_id,
                               neighbor.community_summary AS community_summary
                LIMIT 100
                """,
                seed_ids=seed_ids,
                hops=hops,
            )
            nodes = [{
                "id": r["nid"],
                "content": r["content"] or "",
                "node_type": r["node_type"] or "",
                "community_id": r["community_id"],
                "community_summary": r["community_summary"],
            } for r in result if r["nid"]]

        # 补充种子节点本身
        existing_ids = {n["id"] for n in nodes}
        for seed in seeds:
            if seed.id not in existing_ids:
                nodes.append({
                    "id": seed.id,
                    "content": seed.content,
                    "node_type": str(seed.type),
                    "community_id": seed.metadata.get("community_id"),
                    "community_summary": None,
                })
        return {"nodes": nodes, "edges": []}

    # ------------------------------------------------------------------
    # 阶段 3：社区摘要
    # ------------------------------------------------------------------

    def fetch_community_summaries(self, subgraph: dict) -> list[str]:
        """从子图节点中收集已有的社区摘要，去重后返回。"""
        seen: set[int] = set()
        summaries: list[str] = []
        for node in subgraph["nodes"]:
            summary = node.get("community_summary")
            cid = node.get("community_id")
            if summary and cid not in seen:
                seen.add(cid)
                summaries.append(f"Community {cid}: {summary}")
        return summaries

    # ------------------------------------------------------------------
    # 阶段 4：LLM 重排序
    # ------------------------------------------------------------------

    def rerank(self, query: str, subgraph: dict) -> dict:
        """LLM 打分，保留 top rerank_ratio 比例的节点。最多对 30 个节点打分。"""
        nodes = subgraph["nodes"]
        if not nodes:
            return subgraph

        # 限制送入 LLM 的节点数，避免 token 爆炸
        candidates = nodes[:30]
        nodes_text = "\n".join(
            f"- ID={n['id']} | 类型={n.get('node_type','')} | 内容={n.get('content','')[:100]}"
            for n in candidates
        )
        prompt = _RERANK_PROMPT.format(query=query, nodes_text=nodes_text)

        try:
            scores: dict = self.llm.structured_output(
                prompt=prompt,
                schema={"type": "object", "additionalProperties": {"type": "integer"}},
                system=_RERANK_SYSTEM,
            )
        except Exception as e:
            print(f"[rerank] LLM 打分失败，跳过重排序: {e}")
            return subgraph

        keep_count = max(1, int(len(candidates) * self.rerank_ratio))
        sorted_nodes = sorted(candidates, key=lambda n: scores.get(n["id"], 0), reverse=True)
        kept_ids = {n["id"] for n in sorted_nodes[:keep_count]}

        return {
            "nodes": [n for n in nodes if n["id"] in kept_ids],
            "edges": [e for e in subgraph["edges"]
                      if e.get("source") in kept_ids and e.get("target") in kept_ids],
        }

    # ------------------------------------------------------------------
    # 阶段 5：格式化上下文（算法设计文档 2.3 节）
    # ------------------------------------------------------------------

    def format_context(self, subgraph: dict, summaries: list[str]) -> str:
        """把子图序列化为 LLM 可消费的结构化 Markdown 上下文。"""
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]

        communities_str = "\n".join(f"- {s}" for s in summaries) if summaries else "- (暂无社区摘要)"

        entity_lines, snippet_lines = [], []
        for n in nodes[:30]:
            content = (n.get("content") or "")[:200]
            if n.get("node_type") == "text":
                snippet_lines.append(f"- {n['id']}: {content}")
            else:
                entity_lines.append(f"- {n['id']} [{n.get('node_type','')}]: {content}")

        rel_lines = [
            f"- {e.get('source','?')} --[{e.get('type','?')}]--> {e.get('target','?')}"
            + (f" (weight={e['weight']:.2f})" if e.get("weight") else "")
            for e in edges[:30]
        ]

        return _CONTEXT_TEMPLATE.format(
            communities=communities_str,
            entities="\n".join(entity_lines) or "- (无)",
            relations="\n".join(rel_lines) or "- (无)",
            snippets="\n".join(snippet_lines) or "- (无)",
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> dict:
        """端到端检索：query → {subgraph, summaries, context_text}。

        Returns
        -------
        dict with keys:
            subgraph     : {"nodes": [...], "edges": [...]}
            summaries    : list[str]
            context_text : str  ← 直接塞入 LLM prompt
        """
        # 1. 种子节点
        seeds = self.semantic_search(query)
        print(f"[retriever] 种子节点: {len(seeds)} 个")

        # 2. 子图扩展
        subgraph = self.expand_subgraph(seeds)
        print(f"[retriever] 扩展后子图: {len(subgraph['nodes'])} 节点, {len(subgraph['edges'])} 边")

        # 3. 社区摘要
        summaries = self.fetch_community_summaries(subgraph) if self.use_community_summary else []

        # 4. LLM 重排序
        subgraph = self.rerank(query, subgraph)
        print(f"[retriever] 重排序后: {len(subgraph['nodes'])} 节点")

        # 5. 格式化
        context_text = self.format_context(subgraph, summaries)

        return {
            "subgraph": subgraph,
            "summaries": summaries,
            "context_text": context_text,
        }

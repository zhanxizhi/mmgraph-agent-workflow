"""验证 MultimodalGraphBuilder 文本抽取功能（不需要 Neo4j）。

用法:
    PYTHONUTF8=1 python scripts/test_graph_builder.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.common.llm_client import LLMClient
from src.graph_builder.builder import MultimodalGraphBuilder
from src.common.types import NodeType, EdgeType

SAMPLE_TEXT = """
Kubernetes 中 Pod 出现 CrashLoopBackOff 通常由以下原因导致：

1. 容器启动命令失败：检查 command 和 args 配置是否正确。
2. 资源限制不足：查看 resources.limits 中 CPU 和内存设置。
3. 镜像拉取失败：确认 imagePullPolicy 和私有仓库认证。
4. 配置文件错误：ConfigMap 或 Secret 挂载路径不正确会导致程序启动报错。
5. 健康检查过严：livenessProbe 超时时间太短，容器还未初始化完成就被杀死。

排查步骤：
- kubectl describe pod <pod-name> 查看 Events
- kubectl logs <pod-name> --previous 查看上次崩溃日志
- kubectl exec -it <pod-name> -- /bin/sh 进入容器调试
"""


def test_extract_text_nodes():
    print("\n=== test_extract_text_nodes ===")
    llm = LLMClient()
    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=None)

    nodes, edges = builder.extract_text_nodes(
        SAMPLE_TEXT,
        chunk_size=300,
        chunk_overlap=50,
        source_meta={"scene": "ops", "doc": "k8s_troubleshoot"},
    )

    print(f"节点数: {len(nodes)}")
    print(f"边数:   {len(edges)}")

    text_nodes = [n for n in nodes if n.type == NodeType.TEXT]
    entity_nodes = [n for n in nodes if n.type == NodeType.ENTITY]
    ref_edges = [e for e in edges if e.type == EdgeType.REFERENCES]

    print(f"  TEXT 节点:   {len(text_nodes)}")
    print(f"  ENTITY 节点: {len(entity_nodes)}")
    print(f"  REFERENCES 边: {len(ref_edges)}")

    print("\n--- TEXT 节点摘要 ---")
    for n in text_nodes:
        print(f"  [{n.id}] {n.content}")

    print("\n--- 抽取到的实体 ---")
    for n in entity_nodes:
        etype = n.metadata.get("entity_type", "")
        desc  = n.metadata.get("description", "")
        print(f"  [{etype}] {n.content}: {desc}")

    assert len(text_nodes) >= 1
    assert len(entity_nodes) >= 1
    assert len(ref_edges) >= 1
    print("\nPASS")


def test_embed_nodes():
    print("\n=== test_embed_nodes (BGE-M3) ===")
    llm = LLMClient()
    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=None)

    nodes, _ = builder.extract_text_nodes(SAMPLE_TEXT, chunk_size=512)
    # 只对前3个节点做嵌入（节省时间）
    sample = nodes[:3]
    builder.embed_nodes(sample)

    for n in sample:
        assert n.embedding is not None
        assert len(n.embedding) == 1024   # BGE-M3 维度
        print(f"  [{n.type.value}] {n.content[:30]}... dim={len(n.embedding)}")

    print("PASS")


def test_build_no_neo4j():
    print("\n=== test_build (no Neo4j) ===")
    llm = LLMClient()
    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=None)

    sources = [{"type": "text", "content": SAMPLE_TEXT, "meta": {"scene": "ops"}}]
    nodes, edges = builder.build(sources, embed=False)   # embed=False 跳过嵌入，加快测试

    assert len(nodes) > 0
    assert len(edges) > 0
    print(f"build 完成: {len(nodes)} 节点, {len(edges)} 边")
    print("PASS")


if __name__ == "__main__":
    test_extract_text_nodes()
    test_embed_nodes()
    test_build_no_neo4j()
    print("\n所有测试通过 ✓")

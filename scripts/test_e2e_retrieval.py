"""端到端测试：文本建图 → Neo4j 写入 → GraphRAG 检索

用法:
    PYTHONUTF8=1 python scripts/test_e2e_retrieval.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import neo4j
from src.common.llm_client import LLMClient
from src.graph_builder.builder import MultimodalGraphBuilder
from src.graphrag.retriever import GraphRAGRetriever

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

OPS_DOCS = [
    {
        "type": "text",
        "content": """
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
""",
        "meta": {"scene": "ops", "doc": "k8s_crashloop"},
    },
    {
        "type": "text",
        "content": """
Kubernetes Service 无法访问后端 Pod 的常见原因：
1. Label Selector 不匹配：Service 的 selector 必须与 Pod 的 labels 完全一致。
2. 端口配置错误：targetPort 必须与容器实际监听端口一致。
3. NetworkPolicy 限制：检查是否有 NetworkPolicy 阻断了流量。
4. Pod 未就绪：Pod 处于 Pending 或 CrashLoopBackOff 状态时不会加入 Endpoints。

排查工具：
- kubectl get endpoints <service-name> 查看后端 IP 列表
- kubectl describe service <service-name>
- curl <ClusterIP>:<port> 在集群内测试连通性
""",
        "meta": {"scene": "ops", "doc": "k8s_service"},
    },
]


def main():
    llm = LLMClient()
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # ── 1. 清空旧数据（测试环境幂等）──────────────────────────────
    print("\n[step 1] 清空 Neo4j 测试数据...")
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
    print("  完成")

    # ── 2. 建图写入 ───────────────────────────────────────────────
    print("\n[step 2] 文本建图 + 写入 Neo4j...")
    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=driver)
    nodes, edges = builder.build(OPS_DOCS, embed=True)
    print(f"  写入完成: {len(nodes)} 节点, {len(edges)} 边")

    # ── 3. 向量检索验证 ───────────────────────────────────────────
    print("\n[step 3] GraphRAG 检索...")
    retriever = GraphRAGRetriever(
        neo4j_driver=driver,
        llm_client=llm,
        top_k=5,
        max_hops=2,
        use_community_summary=False,   # 社区摘要需跑 Leiden，这里先跳过
    )

    query = "Pod CrashLoopBackOff 排查步骤"
    result = retriever.retrieve(query)

    print(f"\n  检索到节点数: {len(result['subgraph']['nodes'])}")
    print("\n── 生成的上下文 ──────────────────────────────────────")
    print(result["context_text"])

    assert len(result["subgraph"]["nodes"]) > 0, "检索结果不应为空"
    assert len(result["context_text"]) > 50,    "上下文不应为空"

    driver.close()
    print("\n端到端测试通过 ✓")


if __name__ == "__main__":
    main()

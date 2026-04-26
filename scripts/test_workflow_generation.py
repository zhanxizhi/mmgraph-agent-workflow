"""端到端测试：任务 → 工作流生成 + 评估指标

用法:
    PYTHONUTF8=1 python scripts/test_workflow_generation.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import neo4j
from src.common.llm_client import LLMClient
from src.graph_builder.builder import MultimodalGraphBuilder
from src.graphrag.retriever import GraphRAGRetriever
from src.agent_workflow.feature_graph import AgentMatcher
from src.agent_workflow.ops_agents import build_ops_feature_graph
from src.agent_workflow.generator import WorkflowGenerator, WorkflowValidator
from src.evaluation.metrics import (
    executability, step_completeness, logical_correctness, evaluate_workflow,
    recall_at_k, graph_node_prf,
)
from src.common.types import Workflow, WorkflowStep

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

TASK = "Kubernetes Pod 一直处于 CrashLoopBackOff 状态，参考架构图给出完整排查工作流"

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
]


def main():
    llm    = LLMClient()
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # ── 1. 图谱构建（如果已有数据可跳过）────────────────────────────
    print("\n[step 1] 构建知识图谱...")
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=driver)
    nodes, edges = builder.build(OPS_DOCS, embed=True)
    print(f"  图谱: {len(nodes)} 节点, {len(edges)} 边")

    # ── 2. 智能体特征图 ───────────────────────────────────────────
    print("\n[step 2] 加载 OPS 智能体特征图...")
    fg = build_ops_feature_graph()
    print(f"  注册智能体: {len(fg.agents)-1} 个")  # -1 排除 fallback
    print(f"  能力词表: {fg.get_capability_vocab()}")

    # ── 3. 工作流生成 ─────────────────────────────────────────────
    print(f"\n[step 3] 生成工作流: '{TASK}'")
    retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                   top_k=5, max_hops=2, use_community_summary=False)
    matcher   = AgentMatcher(feature_graph=fg, llm_client=llm)
    generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)

    workflow = generator.generate(TASK, validate=True)

    print(f"\n  生成步骤数: {len(workflow.steps)}")
    print("\n── 工作流步骤 ──────────────────────────────────────────")
    for i, step in enumerate(workflow.steps, 1):
        deps = f" (依赖: {step.depends_on})" if step.depends_on else ""
        print(f"  {i}. [{step.agent_id}] {step.action}{deps}")

    # ── 4. Mermaid 导出 ────────────────────────────────────────────
    print("\n── Mermaid 图 ──────────────────────────────────────────")
    print(generator.to_mermaid(workflow))

    # ── 5. 验证报告 ────────────────────────────────────────────────
    print("\n── 验证报告 ─────────────────────────────────────────────")
    report = workflow.metadata.get("validation", {})
    print(f"  DAG 无环: {report.get('is_dag')}")
    print(f"  IO 错误: {report.get('io_errors')}")
    print(f"  完整性: {report.get('completeness')}")

    # ── 6. 评估指标 ────────────────────────────────────────────────
    print("\n── 评估指标 ─────────────────────────────────────────────")

    # 可执行率（自动）
    exec_score = executability(workflow)
    print(f"  可执行率: {exec_score}")

    # 图谱节点 PRF（用已知 gold 节点做小测试）
    pred_names  = [n.content for n in nodes]
    gold_names  = ["CrashLoopBackOff", "kubectl", "ConfigMap", "livenessProbe"]
    prf = graph_node_prf(pred_names, gold_names)
    print(f"  图谱节点 PRF: {prf}")

    # 逻辑正确性（LLM-as-judge）
    logic_score = logical_correctness(workflow, llm)
    print(f"  逻辑正确性: {logic_score}")

    assert exec_score == 1.0, "工作流应可执行"
    assert len(workflow.steps) >= 3, "至少应有3个步骤"

    driver.close()
    print("\n工作流生成测试通过 ✓")


if __name__ == "__main__":
    main()

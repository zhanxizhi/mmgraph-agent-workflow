"""电力系统端到端测试：PowerGridQA → 图谱 → 工作流生成 + 评估

用法:
    PYTHONUTF8=1 python scripts/test_power_workflow.py
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
from src.agent_workflow.power_agents import build_power_feature_graph
from src.agent_workflow.generator import WorkflowGenerator
from src.evaluation.metrics import executability, logical_correctness, graph_node_prf
from src.data_loader.power_grid_loader import PowerGridQALoader, OPSScenarioLoader

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")

KQA_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kqa")

# 三个场景的测试任务
TEST_TASKS = {
    "ops": "110kV 变电站母线差动保护动作，三相短路故障，请给出完整的故障排查和恢复操作工作流",
    "kqa": "什么是继电保护的选择性原则？如何保证上下级保护的配合？",
    "da":  "基于近30天的负荷历史数据，预测未来24小时分时段负荷，识别峰谷差并生成分析报告",
}


def main():
    llm    = LLMClient()
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    # ── 1. 加载 PowerGridQA 数据 ───────────────────────────────────────
    print("\n[step 1] 加载 PowerGridQA 数据集...")
    kqa_loader = PowerGridQALoader(KQA_DATA_DIR)
    ops_loader = OPSScenarioLoader(KQA_DATA_DIR)

    kqa_sources = kqa_loader.load_sources(splits=["theory", "reasoning"], max_per_split=100)
    ops_sources = ops_loader.load_sources(max_items=100)
    sources = kqa_sources + ops_sources
    print(f"  KQA sources: {len(kqa_sources)}")
    print(f"  OPS sources: {len(ops_sources)}")
    print(f"  总计: {len(sources)} 条")

    # ── 2. 构建知识图谱 ────────────────────────────────────────────────
    print("\n[step 2] 构建电力知识图谱...")
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=driver)
    # 只用前20条做快速测试（完整实验用全量）
    nodes, edges = builder.build(sources[:20], embed=True)
    print(f"  图谱: {len(nodes)} 节点, {len(edges)} 边")

    # ── 3. 电力智能体特征图 ────────────────────────────────────────────
    print("\n[step 3] 加载电力智能体特征图...")
    fg = build_power_feature_graph()
    print(f"  注册智能体: {len(fg.agents)} 个")
    print(f"  能力词表: {fg.get_capability_vocab()}")

    # ── 4. 三场景工作流生成 ────────────────────────────────────────────
    retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                  top_k=5, max_hops=2, use_community_summary=False)
    matcher   = AgentMatcher(feature_graph=fg, llm_client=llm)
    generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)

    results = {}
    for scene, task in TEST_TASKS.items():
        print(f"\n[step 4/{scene.upper()}] 生成工作流: {task[:40]}...")
        workflow = generator.generate(task, validate=True)
        results[scene] = workflow

        print(f"  步骤数: {len(workflow.steps)}")
        for i, step in enumerate(workflow.steps, 1):
            deps = f" ← {step.depends_on}" if step.depends_on else ""
            print(f"    {i}. [{step.agent_id}] {step.action[:50]}{deps}")

    # ── 5. 评估指标 ────────────────────────────────────────────────────
    print("\n── 评估指标 ─────────────────────────────────────────────")
    for scene, workflow in results.items():
        exec_score  = executability(workflow)
        logic_score = logical_correctness(workflow, llm)
        print(f"\n  [{scene.upper()}]")
        print(f"    可执行率:   {exec_score}")
        print(f"    逻辑正确性: {logic_score}")
        assert exec_score == 1.0, f"{scene} 工作流不可执行"

    # ── 6. 图谱节点质量 ────────────────────────────────────────────────
    pred_names = [n.content[:80] for n in nodes]
    gold_names = ["protection relay", "SCADA", "fault", "load forecasting",
                  "circuit breaker", "substation", "power flow"]
    prf = graph_node_prf(pred_names, gold_names)
    print(f"\n  图谱节点 PRF: {prf}")

    # ── 7. Mermaid 导出（OPS 场景）────────────────────────────────────
    print("\n── OPS 工作流 Mermaid ──────────────────────────────────")
    print(generator.to_mermaid(results["ops"]))

    driver.close()
    print("\n电力系统工作流生成测试通过 ✓")


if __name__ == "__main__":
    main()

"""工作流生成主入口（CLI）

用法:
    # 单次生成
    PYTHONUTF8=1 python -m src.agent_workflow.main generate \
        --task "分析110kV母线差动保护动作故障" \
        --scene ops

    # 批量实验评估
    PYTHONUTF8=1 python -m src.agent_workflow.main eval \
        --scenes ops kqa da \
        --baselines ours_full pure_llm \
        --n-tasks 10
"""
from __future__ import annotations

import argparse
import os
import sys

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv
load_dotenv()

import neo4j

from src.common.llm_client import LLMClient
from src.graph_builder.builder import MultimodalGraphBuilder
from src.graphrag.retriever import GraphRAGRetriever
from src.agent_workflow.feature_graph import AgentMatcher
from src.agent_workflow.power_agents import build_power_feature_graph
from src.agent_workflow.generator import WorkflowGenerator
from src.data_loader.power_grid_loader import PowerGridQALoader, OPSScenarioLoader, OPSDLoader


_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")


def _make_driver():
    uri  = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER",     "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "password")
    return neo4j.GraphDatabase.driver(uri, auth=(user, pwd))


def cmd_generate(args):
    llm    = LLMClient()
    driver = _make_driver()

    fg        = build_power_feature_graph()
    retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                  top_k=5, max_hops=2)
    matcher   = AgentMatcher(feature_graph=fg, llm_client=llm)
    generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)

    print(f"\n任务: {args.task}")
    workflow = generator.generate(args.task, validate=True)

    print(f"\n生成 {len(workflow.steps)} 步工作流：")
    for i, step in enumerate(workflow.steps, 1):
        deps = f" ← {step.depends_on}" if step.depends_on else ""
        print(f"  {i}. [{step.agent_id}] {step.action}{deps}")

    if args.mermaid:
        print("\n── Mermaid ──")
        print(generator.to_mermaid(workflow))

    driver.close()


def cmd_build_graph(args):
    """将电力数据集构建为知识图谱（供后续检索）。"""
    llm    = LLMClient()
    driver = _make_driver()

    kqa_loader = PowerGridQALoader(os.path.join(_DATA_DIR, "kqa"))
    ops_loader = OPSScenarioLoader(os.path.join(_DATA_DIR, "kqa"))

    sources: list[dict] = []
    if "kqa" in args.scenes or not args.scenes:
        s = kqa_loader.load_sources(max_per_split=args.max_per_split)
        sources.extend(s)
        print(f"  KQA sources: {len(s)}")
    if "ops" in args.scenes or not args.scenes:
        s = ops_loader.load_sources(max_items=args.max_per_split)
        sources.extend(s)
        print(f"  OPS sources: {len(s)}")

    print(f"\n构建图谱（{len(sources)} 文档）...")
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")

    builder = MultimodalGraphBuilder(llm_client=llm, neo4j_driver=driver)
    nodes, edges = builder.build(sources, embed=True)
    print(f"图谱完成: {len(nodes)} 节点, {len(edges)} 边")
    driver.close()


def cmd_eval(args):
    from src.evaluation.runner import EvaluationRunner
    kqa_loader = PowerGridQALoader(os.path.join(_DATA_DIR, "kqa"))
    ops_loader = OPSScenarioLoader(os.path.join(_DATA_DIR, "kqa"))

    benchmarks: dict[str, list[dict]] = {}
    if "ops" in args.scenes:
        benchmarks["ops"] = ops_loader.load_benchmark(n=args.n_tasks)
    if "kqa" in args.scenes:
        benchmarks["kqa"] = kqa_loader.load_benchmark(n=args.n_tasks)
    if "da" in args.scenes:
        from src.data_loader.power_grid_loader import OPSDLoader
        da_loader = OPSDLoader(os.path.join(_DATA_DIR, "opsd", "time_series_60min_singleindex.csv"))
        benchmarks["da"] = da_loader.load_benchmark(n=args.n_tasks)

    llm    = LLMClient()
    driver = _make_driver()
    runner = EvaluationRunner(neo4j_driver=driver, llm_client=llm)

    results = runner.run(benchmarks, baselines=args.baselines or None)
    runner.print_summary(results)
    driver.close()


def main():
    parser = argparse.ArgumentParser(description="电力系统智能体工作流生成")
    sub = parser.add_subparsers(dest="cmd")

    # generate
    p_gen = sub.add_parser("generate", help="生成单个工作流")
    p_gen.add_argument("--task", required=True, help="任务描述")
    p_gen.add_argument("--scene", default="ops", choices=["ops", "kqa", "da"])
    p_gen.add_argument("--mermaid", action="store_true", help="输出 Mermaid 图")

    # build-graph
    p_build = sub.add_parser("build-graph", help="构建电力知识图谱")
    p_build.add_argument("--scenes", nargs="*", default=["ops", "kqa"],
                         choices=["ops", "kqa", "da"])
    p_build.add_argument("--max-per-split", type=int, default=200)

    # eval
    p_eval = sub.add_parser("eval", help="运行实验评估")
    p_eval.add_argument("--scenes", nargs="+", default=["ops", "kqa"],
                        choices=["ops", "kqa", "da"])
    p_eval.add_argument("--baselines", nargs="*",
                        choices=["pure_llm", "vector_rag", "text_graphrag",
                                 "graphrag_nomatch", "ours_full"])
    p_eval.add_argument("--n-tasks", type=int, default=10)

    args = parser.parse_args()
    if args.cmd == "generate":
        cmd_generate(args)
    elif args.cmd == "build-graph":
        cmd_build_graph(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

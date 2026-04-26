"""算法 C 权重 (α, β, γ) 敏感性扫描

为论文消融实验补充 "权重选择是否合理" 的定量证据。

策略：
  1. 第一遍对每条任务跑一次 retrieve + decompose + embed_subtasks，把
     与权重无关的中间产物缓存进 ``--cache`` JSONL。
  2. 在缓存上对每个 (α, β, γ) 仅运行匹配阶段，省掉重复的 LLM 调用，
     扫描成本从 |grid| × |任务| 次 LLM 调用降到 |任务| 次。
  3. 对每个权重组合，按拓扑顺序与 gold 工作流对齐计算
     role_assignment_accuracy，输出 CSV + PGFPlots 数据。

用法::

    PYTHONUTF8=1 python scripts/run_weight_sweep.py \
        --limit 15 --grid 0.1 \
        --output data/benchmarks/weight_sweep.csv

如已经跑过缓存，可重复 sweep::

    PYTHONUTF8=1 python scripts/run_weight_sweep.py \
        --use-cache --output data/benchmarks/weight_sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

import neo4j

from src.agent_workflow.feature_graph import AgentFeatureGraph, AgentMatcher
from src.agent_workflow.power_agents import build_power_feature_graph
from src.common.llm_client import LLMClient
from src.common.types import Workflow, WorkflowStep
from src.evaluation.metrics import role_assignment_accuracy
from src.graphrag.retriever import GraphRAGRetriever


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
GOLD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "power_gold_workflows_extended.json")
DEFAULT_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "weight_sweep_cache.jsonl")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "weight_sweep.csv")
FIGURE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis", "figures", "data")


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def gold_to_workflow(item: dict) -> Workflow:
    steps = []
    for i, s in enumerate(item["gold_steps"]):
        dep = [f"s{i}"] if i > 0 else []
        steps.append(WorkflowStep(
            step_id=f"s{i+1}",
            agent_id=s["agent_id"],
            action=s["action"],
            inputs={},
            depends_on=dep,
        ))
    return Workflow(task=item["task"], steps=steps)


def build_cache(
    items: list[dict],
    fg: AgentFeatureGraph,
    retriever: GraphRAGRetriever,
    matcher: AgentMatcher,
    cache_path: str,
) -> list[dict]:
    """对每条任务运行 retrieve + decompose + embed_subtasks，结果存盘。"""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cached: list[dict] = []
    with open(cache_path, "w", encoding="utf-8") as f:
        for item in items:
            task = item["task"]
            t0 = time.time()
            try:
                retrieval = retriever.retrieve(task)
                context = retrieval.get("context_text", "")
                subtasks = matcher.decompose_task(task, context)
                if not subtasks:
                    print(f"  [skip] {item.get('id')}: 任务分解为空")
                    continue
                subtasks = matcher.embed_subtask_capabilities(subtasks)
            except Exception as e:
                print(f"  [skip] {item.get('id')}: {e}")
                continue
            entry = {
                "task_id": item.get("id", ""),
                "scene": item.get("scene", ""),
                "task": task,
                "gold_steps": item["gold_steps"],
                "subtasks": subtasks,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            cached.append(entry)
            print(f"  [cache] {item.get('id')} subtasks={len(subtasks)} {round(time.time()-t0, 1)}s")
    print(f"\nCache written: {cache_path} ({len(cached)} items)")
    return cached


def load_cache(cache_path: str) -> list[dict]:
    items: list[dict] = []
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def make_grid(step: float) -> list[tuple[float, float, float]]:
    """生成 α + β + γ = 1, 各分量按 step 离散化的所有组合。"""
    grid: list[tuple[float, float, float]] = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            grid.append((round(i * step, 4), round(j * step, 4), round(k * step, 4)))
    return grid


def evaluate_weight_combo(
    cached: list[dict],
    fg: AgentFeatureGraph,
    llm: LLMClient,
    alpha: float,
    beta: float,
    gamma: float,
) -> dict[str, Any]:
    """在缓存数据上仅跑匹配阶段（无 LLM 调用），返回汇总指标。

    为了让网格扫描的代价线性于权重组合数而非任务数 × 组合数 × LLM 延迟，
    本函数\emph{不}调用 ``WorkflowGenerator._build_dag``——后者会触发
    LLM 隐式依赖推断。在 RAA 计算中只需要每个子任务的 agent 分配，因此
    直接基于 ``match_all`` 的结果构造 ``Workflow`` 即可。
    """
    matcher = AgentMatcher(
        feature_graph=fg, llm_client=llm,
        alpha=alpha, beta=beta, gamma=gamma,
    )

    raa_scores: list[float] = []
    per_scene: dict[str, list[float]] = {}
    for entry in cached:
        subtasks = [dict(st) for st in entry["subtasks"]]  # 浅拷贝，避免污染缓存
        try:
            assignments = matcher.match_all(subtasks)
        except Exception:
            continue
        pred_steps = [
            WorkflowStep(
                step_id=st["id"], agent_id=agent.id,
                action=st.get("description", ""), inputs={},
                depends_on=st.get("depends_on", []),
            )
            for st, agent, _ in assignments
        ]
        pred_workflow = Workflow(task=entry["task"], steps=pred_steps)
        gold_workflow = Workflow(
            task=entry["task"],
            steps=[
                WorkflowStep(
                    step_id=f"s{i+1}", agent_id=g["agent_id"],
                    action=g["action"], inputs={}, depends_on=[],
                )
                for i, g in enumerate(entry["gold_steps"])
            ],
        )
        score = role_assignment_accuracy(pred_workflow, gold_workflow)
        raa_scores.append(score)
        per_scene.setdefault(entry.get("scene", ""), []).append(score)

    return {
        "raa_mean": sum(raa_scores) / len(raa_scores) if raa_scores else 0.0,
        "raa_per_scene": {s: sum(v) / len(v) if v else 0.0 for s, v in per_scene.items()},
        "n_tasks": len(raa_scores),
    }


def write_results(rows: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["alpha", "beta", "gamma", "raa_mean", "raa_ops", "raa_kqa", "raa_da", "n_tasks"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSweep CSV saved: {output_path}")


def write_pgfplots(rows: list[dict]) -> None:
    """写一份 PGFPlots-friendly 数据，用于论文里画 (α, β) heatmap。"""
    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DATA_DIR, "weight_sweep.dat")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["alpha", "beta", "gamma", "raa"])
        for row in rows:
            writer.writerow([row["alpha"], row["beta"], row["gamma"], f"{row['raa_mean']:.4f}"])
    print(f"PGFPlots data saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-file", default=GOLD_FILE)
    parser.add_argument("--limit", type=int, default=15,
                        help="Number of tasks (after filtering by --scenes) used in the sweep.")
    parser.add_argument("--scenes", default="ops,kqa,da",
                        help="Comma-separated scene filter for cache building.")
    parser.add_argument("--grid", type=float, default=0.1,
                        help="Discretization step for α/β/γ. Smaller = denser grid.")
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--use-cache", action="store_true",
                        help="Skip retrieval/decomposition and reuse an existing cache file.")
    parser.add_argument("--no-pgfplots", action="store_true",
                        help="Skip writing thesis/figures/data/weight_sweep.dat.")
    args = parser.parse_args()

    llm = LLMClient()
    fg = build_power_feature_graph()

    if args.use_cache:
        if not os.path.exists(args.cache):
            raise FileNotFoundError(f"Cache not found: {args.cache}")
        cached = load_cache(args.cache)
        print(f"Loaded cache: {args.cache} ({len(cached)} items)")
    else:
        driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        retriever = GraphRAGRetriever(
            neo4j_driver=driver, llm_client=llm,
            top_k=5, max_hops=2, use_community_summary=False,
        )
        matcher = AgentMatcher(feature_graph=fg, llm_client=llm)

        scenes = {s.strip().lower() for s in args.scenes.split(",") if s.strip()}
        items = load_gold(args.gold_file)
        items = [i for i in items if i.get("scene", "").lower() in scenes]
        if args.limit and args.limit > 0:
            items = items[: args.limit]
        print(f"Building cache for {len(items)} tasks...")
        cached = build_cache(items, fg, retriever, matcher, args.cache)
        driver.close()

    grid = make_grid(args.grid)
    print(f"\nSweeping {len(grid)} weight combinations × {len(cached)} tasks...")

    rows: list[dict] = []
    for idx, (alpha, beta, gamma) in enumerate(grid, 1):
        result = evaluate_weight_combo(cached, fg, llm, alpha, beta, gamma)
        per_scene = result["raa_per_scene"]
        rows.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            "raa_mean": round(result["raa_mean"], 4),
            "raa_ops": round(per_scene.get("ops", 0.0), 4),
            "raa_kqa": round(per_scene.get("kqa", 0.0), 4),
            "raa_da": round(per_scene.get("da", 0.0), 4),
            "n_tasks": result["n_tasks"],
        })
        print(f"  [{idx}/{len(grid)}] α={alpha:.1f} β={beta:.1f} γ={gamma:.1f} → RAA={result['raa_mean']:.3f}")

    rows.sort(key=lambda r: r["raa_mean"], reverse=True)
    write_results(rows, args.output)
    if not args.no_pgfplots:
        write_pgfplots(rows)

    print("\nTop 5 weight combinations by RAA mean:")
    for r in rows[:5]:
        print(f"  α={r['alpha']:.1f} β={r['beta']:.1f} γ={r['gamma']:.1f} → RAA={r['raa_mean']:.4f}")


if __name__ == "__main__":
    main()

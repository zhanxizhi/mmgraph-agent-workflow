"""复杂运维工作流评估：手工设计的 gold workflow 基准

用法:
    PYTHONUTF8=1 python scripts/run_eval_ops_workflow.py
    PYTHONUTF8=1 python scripts/run_eval_ops_workflow.py --plots-only
"""
import argparse
import csv
import os, sys, json, time, random
import math
from collections import Counter, defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import neo4j
from neo4j.exceptions import ServiceUnavailable
from src.common.llm_client import LLMClient
from src.common.types import Workflow, WorkflowStep
from src.agent_workflow.feature_graph import AgentMatcher
from src.agent_workflow.power_agents import build_power_feature_graph
from src.agent_workflow.generator import WorkflowGenerator
from src.graphrag.retriever import GraphRAGRetriever
from src.evaluation.metrics import (
    executability, logical_correctness, step_completeness,
    role_assignment_accuracy,
)

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
GOLD_FILE  = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "power_gold_workflows_extended.json")
LEGACY_GOLD_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "power_gold_workflows.json")
RESULT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "eval_ops_workflow.csv")
WORKFLOW_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "benchmarks", "eval_ops_workflow_workflows.jsonl")
FIGURE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "thesis", "figures", "data")

BASELINES = ["ours_full", "pure_llm", "vector_rag", "graphrag_nomatch"]
SCENES = ["ops", "kqa", "da"]
METRICS = ["exec", "step_completeness", "step_completeness_lex", "role_accuracy", "logic"]


def load_gold(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _num(item: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(item.get(key, default))
    except (TypeError, ValueError):
        return default


def _avg(items: list[dict], key: str) -> float:
    return sum(_num(item, key) for item in items) / len(items) if items else 0.0


def _std(items: list[dict], key: str) -> float:
    if len(items) <= 1:
        return 0.0
    mean = _avg(items, key)
    var = sum((_num(item, key) - mean) ** 2 for item in items) / (len(items) - 1)
    return math.sqrt(var)


def _ci95(items: list[dict], key: str) -> float:
    if len(items) <= 1:
        return 0.0
    return 1.96 * _std(items, key) / math.sqrt(len(items))


def _derived_path(result_path: str, suffix: str, ext: str = ".csv") -> str:
    base, _ = os.path.splitext(result_path)
    return f"{base}_{suffix}{ext}"


def write_summary_tables(rows: list[dict], result_path: str = RESULT_FILE) -> None:
    """Write CSV summaries for reproducibility and thesis tables."""
    if not rows:
        return

    summary_path = _derived_path(result_path, "summary")
    overall_path = _derived_path(result_path, "overall")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_baseline: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("scene", ""), row.get("baseline", ""))].append(row)
        by_baseline[row.get("baseline", "")].append(row)

    metric_fields = []
    for metric in METRICS:
        metric_fields.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_ci95"])
    extra_fields = ["steps_mean", "latency_mean"]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scene", "baseline", "n", *metric_fields, *extra_fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (scene, baseline), items in sorted(grouped.items()):
            out = {"scene": scene, "baseline": baseline, "n": len(items)}
            for metric in METRICS:
                out[f"{metric}_mean"] = f"{_avg(items, metric):.4f}"
                out[f"{metric}_std"] = f"{_std(items, metric):.4f}"
                out[f"{metric}_ci95"] = f"{_ci95(items, metric):.4f}"
            out["steps_mean"] = f"{_avg(items, 'steps'):.2f}"
            out["latency_mean"] = f"{_avg(items, 'latency'):.2f}"
            writer.writerow(out)

    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["baseline", "n", *metric_fields, *extra_fields]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for baseline in BASELINES:
            items = by_baseline.get(baseline, [])
            out = {"baseline": baseline, "n": len(items)}
            for metric in METRICS:
                out[f"{metric}_mean"] = f"{_avg(items, metric):.4f}"
                out[f"{metric}_std"] = f"{_std(items, metric):.4f}"
                out[f"{metric}_ci95"] = f"{_ci95(items, metric):.4f}"
            out["steps_mean"] = f"{_avg(items, 'steps'):.2f}"
            out["latency_mean"] = f"{_avg(items, 'latency'):.2f}"
            writer.writerow(out)

    print(f"Summary saved: {summary_path}")
    print(f"Overall summary saved: {overall_path}")


def write_benchmark_tables(gold_items: list[dict] | None) -> None:
    """Write benchmark composition tables independent of model execution."""
    if not gold_items:
        return

    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)
    by_scene: dict[str, list[dict]] = defaultdict(list)
    by_difficulty: dict[str, list[dict]] = defaultdict(list)
    for item in gold_items:
        by_scene[item.get("scene", "").lower()].append(item)
        by_difficulty[item.get("difficulty", "unknown").lower()].append(item)

    scene_path = os.path.join(FIGURE_DATA_DIR, "benchmark_scene_stats.dat")
    with open(scene_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["scene", "tasks", "avg_steps", "medium", "hard"])
        for scene in SCENES:
            items = by_scene.get(scene, [])
            diff = Counter(item.get("difficulty", "unknown").lower() for item in items)
            avg_steps = (
                sum(len(item.get("gold_steps", [])) for item in items) / len(items)
                if items else 0.0
            )
            writer.writerow([
                scene.upper(), len(items), f"{avg_steps:.2f}",
                diff.get("medium", 0), diff.get("hard", 0),
            ])

    difficulty_path = os.path.join(FIGURE_DATA_DIR, "benchmark_difficulty_stats.dat")
    with open(difficulty_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["difficulty", "tasks", "avg_steps"])
        for difficulty in ["medium", "hard"]:
            items = by_difficulty.get(difficulty, [])
            avg_steps = (
                sum(len(item.get("gold_steps", [])) for item in items) / len(items)
                if items else 0.0
            )
            writer.writerow([difficulty.capitalize(), len(items), f"{avg_steps:.2f}"])

    print(f"Benchmark data saved: {scene_path}")
    print(f"Benchmark data saved: {difficulty_path}")


def write_visualization_tables(
    rows: list[dict],
    gold_items: list[dict] | None = None,
    result_path: str = RESULT_FILE,
) -> None:
    """Write compact PGFPlots data tables used by the thesis figures."""
    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)
    write_summary_tables(rows, result_path=result_path)
    write_benchmark_tables(gold_items)

    by_scene = {(scene, baseline): [] for scene in SCENES for baseline in BASELINES}
    by_baseline = {baseline: [] for baseline in BASELINES}
    for row in rows:
        scene = row["scene"]
        baseline = row["baseline"]
        if scene in SCENES and baseline in BASELINES:
            by_scene[(scene, baseline)].append(row)
            by_baseline[baseline].append(row)

    raa_by_scene_path = os.path.join(FIGURE_DATA_DIR, "raa_by_scene.dat")
    with open(raa_by_scene_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["scene", *BASELINES])
        for scene in SCENES:
            writer.writerow([
                scene.upper(),
                *[f"{_avg(by_scene[(scene, baseline)], 'role_accuracy'):.3f}" for baseline in BASELINES],
            ])
        writer.writerow([
            "AVG",
            *[f"{_avg(by_baseline[baseline], 'role_accuracy'):.3f}" for baseline in BASELINES],
        ])

    ablation_path = os.path.join(FIGURE_DATA_DIR, "ablation_raa.dat")
    ablation_rows = [
        ("Full", "ours_full"),
        ("NoSubgraph", "vector_rag"),
        ("NoMatch", "graphrag_nomatch"),
        ("NoKG", "pure_llm"),
    ]
    with open(ablation_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["config", "raa"])
        for label, baseline in ablation_rows:
            writer.writerow([label, f"{_avg(by_baseline[baseline], 'role_accuracy'):.3f}"])

    metric_overview_path = os.path.join(FIGURE_DATA_DIR, "metric_overview.dat")
    with open(metric_overview_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["metric", *BASELINES])
        metric_labels = [
            ("Exec", "exec"),
            ("SC", "step_completeness"),
            ("RAA", "role_accuracy"),
            ("Logic", "logic"),
        ]
        for label, key in metric_labels:
            writer.writerow([
                label,
                *[f"{_avg(by_baseline[baseline], key):.3f}" for baseline in BASELINES],
            ])

    latency_path = os.path.join(FIGURE_DATA_DIR, "latency_by_baseline.dat")
    with open(latency_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerow(["baseline", "latency", "ci95"])
        labels = [
            ("Full", "ours_full"),
            ("Pure", "pure_llm"),
            ("Vector", "vector_rag"),
            ("NoMatch", "graphrag_nomatch"),
        ]
        for label, baseline in labels:
            items = by_baseline[baseline]
            writer.writerow([
                label,
                f"{_avg(items, 'latency'):.2f}",
                f"{_ci95(items, 'latency'):.2f}",
            ])

    coverage_path = os.path.join(FIGURE_DATA_DIR, "eval_coverage.dat")
    if gold_items:
        total_by_scene = Counter(item.get("scene", "").lower() for item in gold_items)
        evaluated_by_scene = Counter()
        seen_task_scene = set()
        for row in rows:
            scene = row.get("scene", "").lower()
            task_id = row.get("task_id") or row.get("task", "")
            key = (scene, task_id)
            if scene and key not in seen_task_scene:
                evaluated_by_scene[scene] += 1
                seen_task_scene.add(key)
        with open(coverage_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=" ")
            writer.writerow(["scene", "evaluated", "total"])
            for scene in SCENES:
                writer.writerow([
                    scene.upper(),
                    evaluated_by_scene.get(scene, 0),
                    total_by_scene.get(scene, 0),
                ])

    print(f"Visualization data saved: {raa_by_scene_path}")
    print(f"Visualization data saved: {ablation_path}")
    print(f"Visualization data saved: {metric_overview_path}")
    print(f"Visualization data saved: {latency_path}")
    if gold_items:
        print(f"Visualization data saved: {coverage_path}")


def load_result_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def run_pure_llm(task, llm):
    result = llm.structured_output(
        prompt=(
            f"你是电力系统专家。请为以下任务生成工作流步骤：\n{task}\n\n"
            "返回JSON: {\"steps\": [{\"step_id\": \"s1\", \"agent_id\": \"agent\","
            " \"action\": \"...\", \"depends_on\": []}]}"
        ),
        schema={
            "type": "object",
            "properties": {"steps": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "step_id":    {"type": "string"},
                    "agent_id":   {"type": "string"},
                    "action":     {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                }
            }}}
        },
    )
    steps = [
        WorkflowStep(
            step_id=s.get("step_id", f"s{i+1}"),
            agent_id=s.get("agent_id", "unknown"),
            action=s.get("action", ""),
            inputs={},
            depends_on=s.get("depends_on", []),
        )
        for i, s in enumerate(result.get("steps", []))
    ]
    return Workflow(task=task, steps=steps)


def run_vector_rag(task, llm, driver, fg):
    from src.graphrag.retriever import GraphRAGRetriever
    retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                  top_k=5, max_hops=0)
    matcher   = AgentMatcher(feature_graph=fg, llm_client=llm)
    generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)
    return generator.generate(task, validate=False)


def run_graphrag_nomatch(task, retriever, fg, generator):
    result  = retriever.retrieve(task)
    context = result.get("context_text", "")
    from src.agent_workflow.feature_graph import AgentMatcher
    matcher  = AgentMatcher(feature_graph=fg, llm_client=generator.llm)
    subtasks = matcher.decompose_task(task, context)
    agents   = [a for a in fg.agents.values() if a.id != "__fallback__"]
    random.seed(42)
    assignments = [(st, random.choice(agents), 0.5) for st in subtasks]
    return generator._build_dag(task, assignments)


def _make_driver():
    return neo4j.GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS),
        max_connection_lifetime=300,   # 5 min，防空闲断开
        keep_alive=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold-file",
        default=GOLD_FILE,
        help="Gold workflow JSON file. Defaults to the extended 30-task benchmark.",
    )
    parser.add_argument(
        "--legacy-gold",
        action="store_true",
        help="Use the original 6-task gold benchmark for quick comparison.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N selected tasks. 0 means all tasks.",
    )
    parser.add_argument(
        "--scenes",
        default=",".join(SCENES),
        help="Comma-separated scene filter, e.g. ops,kqa or da.",
    )
    parser.add_argument(
        "--skip-logic",
        action="store_true",
        help="Skip LLM-as-judge logical correctness to reduce API cost.",
    )
    parser.add_argument(
        "--baselines",
        default=",".join(BASELINES),
        help="Comma-separated baselines to run. Defaults to all baselines.",
    )
    parser.add_argument(
        "--output",
        default=RESULT_FILE,
        help="CSV path for detailed per-task results.",
    )
    parser.add_argument(
        "--workflow-output",
        default=WORKFLOW_FILE,
        help="JSONL path for generated workflows and gold workflows.",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Skip LLM/Neo4j evaluation and regenerate PGFPlots data from the saved CSV.",
    )
    args = parser.parse_args()

    gold_path = LEGACY_GOLD_FILE if args.legacy_gold else args.gold_file
    gold_items_for_plots = load_gold(gold_path) if os.path.exists(gold_path) else []

    if args.plots_only:
        write_visualization_tables(
            load_result_rows(args.output),
            gold_items=gold_items_for_plots,
            result_path=args.output,
        )
        return

    selected_baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    invalid = [b for b in selected_baselines if b not in BASELINES]
    if invalid:
        raise ValueError(f"Unknown baselines: {invalid}. Valid choices: {BASELINES}")

    llm    = LLMClient()
    driver = _make_driver()
    gold_items = load_gold(gold_path)
    selected_scenes = {scene.strip().lower() for scene in args.scenes.split(",") if scene.strip()}
    gold_items = [item for item in gold_items if item.get("scene", "").lower() in selected_scenes]
    if args.limit and args.limit > 0:
        gold_items = gold_items[:args.limit]
    print(f"Gold benchmark: {gold_path}")
    print(f"Selected tasks: {len(gold_items)}")

    fg        = build_power_feature_graph()
    retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                  top_k=5, max_hops=2, use_community_summary=False)
    matcher   = AgentMatcher(feature_graph=fg, llm_client=llm)
    generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)

    rows = []
    workflow_records = []
    for item in gold_items:
        task  = item["task"]
        gold  = gold_to_workflow(item)
        scene = item["scene"]
        print(f"\n{'='*65}")
        print(f"[{scene.upper()}] {task[:60]}...")
        print(f"{'='*65}")

        for baseline in selected_baselines:
            t0 = time.time()
            try:
                if baseline == "ours_full":
                    wf = generator.generate(task, validate=True)
                elif baseline == "pure_llm":
                    wf = run_pure_llm(task, llm)
                elif baseline == "vector_rag":
                    wf = run_vector_rag(task, llm, driver, fg)
                else:
                    wf = run_graphrag_nomatch(task, retriever, fg, generator)

                exec_s  = executability(wf)
                sc      = step_completeness(wf, gold, embed_fn=llm.embed)
                sc_lex  = step_completeness(wf, gold)  # 词袋回退指标，仅用于消融对比
                logic_s = 0.0 if args.skip_logic else logical_correctness(wf, llm)
                role_s  = role_assignment_accuracy(wf, gold)
            except ServiceUnavailable:
                print(f"  [{baseline}] Neo4j 断开，尝试重连...")
                try:
                    driver.close()
                except Exception:
                    pass
                driver = _make_driver()
                # 重建依赖 driver 的对象
                retriever = GraphRAGRetriever(neo4j_driver=driver, llm_client=llm,
                                              top_k=5, max_hops=2, use_community_summary=False)
                generator = WorkflowGenerator(llm_client=llm, retriever=retriever, matcher=matcher)
                exec_s = sc = sc_lex = logic_s = role_s = 0.0
                wf = Workflow(task=task, steps=[])
            except Exception as e:
                print(f"  [{baseline}] ERROR: {e}")
                exec_s = sc = sc_lex = logic_s = role_s = 0.0
                wf = Workflow(task=task, steps=[])

            latency = round(time.time() - t0, 1)
            print(f"  [{baseline:20s}] exec={exec_s:.2f}  sc={sc:.2f} (lex={sc_lex:.2f})  role={role_s:.2f}  logic={logic_s:.2f}  steps={len(wf.steps)}  {latency}s")
            rows.append({
                "task_id": item.get("id", ""),
                "scene": scene, "baseline": baseline,
                "source": item.get("source", ""),
                "difficulty": item.get("difficulty", ""),
                "task": task[:80],
                "exec": exec_s,
                "step_completeness": sc,
                "step_completeness_lex": sc_lex,
                "role_accuracy": role_s, "logic": logic_s,
                "steps": len(wf.steps), "latency": latency,
            })
            workflow_records.append({
                "task_id": item.get("id", ""),
                "scene": scene,
                "baseline": baseline,
                "source": item.get("source", ""),
                "difficulty": item.get("difficulty", ""),
                "task": task,
                "metrics": {
                    "exec": exec_s,
                    "step_completeness": sc,
                    "role_accuracy": role_s,
                    "logic": logic_s,
                    "steps": len(wf.steps),
                    "latency": latency,
                },
                "gold_workflow": gold.to_dag_json(),
                "pred_workflow": wf.to_dag_json(),
            })

    # ── 汇总 ──────────────────────────────────────────────────────────
    print(f"\n{'='*75}")
    print(f"{'场景':<6} {'基线':<22} {'可执行率':>8} {'步骤完整性':>10} {'角色准确率':>10} {'逻辑正确':>8} {'步骤数':>6}")
    print(f"{'-'*80}")

    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["scene"], r["baseline"])].append(r)

    for (scene, bl), items in sorted(agg.items()):
        avg = lambda k: sum(i[k] for i in items) / len(items)
        print(f"{scene:<6} {bl:<22} {avg('exec'):>8.3f} {avg('step_completeness'):>10.3f}"
              f" {avg('role_accuracy'):>10.3f} {avg('logic'):>8.3f} {avg('steps'):>6.1f}")

    print(f"{'='*75}")

    # 保存
    out = args.output
    if rows:
        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n结果已保存: {out}")
        if workflow_records:
            workflow_dir = os.path.dirname(args.workflow_output)
            if workflow_dir:
                os.makedirs(workflow_dir, exist_ok=True)
            with open(args.workflow_output, "w", encoding="utf-8") as f:
                for record in workflow_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"工作流明细已保存: {args.workflow_output}")
        write_visualization_tables(rows, gold_items=gold_items, result_path=out)
    else:
        print("\nNo rows generated; check --scenes/--limit filters.")
    driver.close()


if __name__ == "__main__":
    main()

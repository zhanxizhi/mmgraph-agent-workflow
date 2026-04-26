"""快速评估：5任务 × 2基线（ours_full vs pure_llm），验证系统完整性

用法（图谱构建完成后）:
    PYTHONUTF8=1 python scripts/run_eval_quick.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import neo4j
from src.common.llm_client import LLMClient
from src.evaluation.runner import EvaluationRunner
from src.data_loader.power_grid_loader import PowerGridQALoader, OPSScenarioLoader

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD", "password")
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kqa")

def main():
    llm    = LLMClient()
    driver = neo4j.GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

    kqa_loader = PowerGridQALoader(DATA_DIR)
    ops_loader = OPSScenarioLoader(DATA_DIR)

    benchmarks = {
        "ops": ops_loader.load_benchmark(n=5),
        "kqa": kqa_loader.load_benchmark(splits=["theory"], n=5),
    }

    runner = EvaluationRunner(neo4j_driver=driver, llm_client=llm)
    results = runner.run(benchmarks, baselines=["ours_full", "pure_llm"])
    runner.print_summary(results)
    driver.close()

if __name__ == "__main__":
    main()

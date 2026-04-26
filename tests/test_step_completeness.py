"""step_completeness 双模式回归测试

不依赖真实 BGE-M3，构造 mock embed_fn 验证：
  * 语义近义中文步骤 → 应被覆盖（旧 Jaccard 路径漏掉的情形）
  * 完全无关步骤 → 不应覆盖
  * 未传 embed_fn 时回退到词袋路径，行为与旧版一致
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.common.types import Workflow, WorkflowStep
from src.evaluation.metrics import step_completeness


def _wf(actions: list[str]) -> Workflow:
    steps = [
        WorkflowStep(step_id=f"s{i}", agent_id="a", action=a, inputs={}, depends_on=[])
        for i, a in enumerate(actions)
    ]
    return Workflow(task="t", steps=steps)


def _stub_embed_fn(target_pairs: dict[tuple[str, str], float]):
    """构造测试用嵌入：根据 (i, j) 文本对的目标余弦相似度反推 2D 向量。"""
    text_to_vec: dict[str, list[float]] = {}

    def _angle_for(score: float) -> float:
        score = max(-1.0, min(1.0, score))
        return math.acos(score)

    def fn(texts):
        for t in texts:
            if t not in text_to_vec:
                text_to_vec[t] = [1.0, 0.0]  # 默认朝 x 轴
        # 应用目标相似度：把第二个文本旋转到与第一个夹角对应的位置
        for (a, b), sim in target_pairs.items():
            theta = _angle_for(sim)
            text_to_vec[b] = [math.cos(theta), math.sin(theta)]
        return [text_to_vec[t] for t in texts]

    return fn


def test_semantic_path_catches_paraphrase():
    gold = _wf(["校验保护整定值"])
    pred = _wf(["检查继电保护定值"])
    embed = _stub_embed_fn({("校验保护整定值", "检查继电保护定值"): 0.92})
    sc = step_completeness(pred, gold, embed_fn=embed)
    assert sc == 1.0, f"语义近义应被覆盖，得到 {sc}"


def test_semantic_path_rejects_unrelated():
    gold = _wf(["校验保护整定值"])
    pred = _wf(["生成故障报告"])
    embed = _stub_embed_fn({("校验保护整定值", "生成故障报告"): 0.10})
    sc = step_completeness(pred, gold, embed_fn=embed)
    assert sc == 0.0, f"无关步骤不应覆盖，得到 {sc}"


def test_threshold_override():
    gold = _wf(["校验保护整定值"])
    pred = _wf(["检查继电保护定值"])
    embed = _stub_embed_fn({("校验保护整定值", "检查继电保护定值"): 0.50})
    # 严格阈值 → 不通过
    assert step_completeness(pred, gold, threshold=0.65, embed_fn=embed) == 0.0
    # 宽松阈值 → 通过
    assert step_completeness(pred, gold, threshold=0.40, embed_fn=embed) == 1.0


def test_lexical_fallback_unchanged():
    gold = _wf(["kubectl describe pod"])
    pred = _wf(["kubectl describe pod"])
    assert step_completeness(pred, gold) == 1.0
    assert step_completeness(_wf(["完全无关内容"]), gold) == 0.0


def test_empty_pred_returns_zero():
    gold = _wf(["a", "b"])
    pred = _wf([])
    assert step_completeness(pred, gold) == 0.0
    assert step_completeness(pred, gold, embed_fn=lambda xs: [[1.0]] * len(xs)) == 0.0


def test_empty_gold_returns_one():
    assert step_completeness(_wf(["a"]), _wf([])) == 1.0


def test_embed_failure_falls_back_silently():
    gold = _wf(["kubectl describe pod"])
    pred = _wf(["kubectl describe pod"])

    def bad_embed(_):
        raise RuntimeError("embedder offline")

    # 嵌入抛错时回退词袋路径，仍能算出 1.0
    assert step_completeness(pred, gold, embed_fn=bad_embed) == 1.0


if __name__ == "__main__":
    test_semantic_path_catches_paraphrase()
    test_semantic_path_rejects_unrelated()
    test_threshold_override()
    test_lexical_fallback_unchanged()
    test_empty_pred_returns_zero()
    test_empty_gold_returns_one()
    test_embed_failure_falls_back_silently()
    print("all step_completeness tests passed.")

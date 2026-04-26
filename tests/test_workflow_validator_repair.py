"""WorkflowValidator 迭代修复回归测试

验证算法 D 的三类修复操作 + 迭代循环上限：
  * repair_cycle：找到并删除环路边
  * repair_interface：删除 IO 冲突的依赖边
  * repair_coverage：为 LLM 报告的缺失步骤追加兜底节点
  * validate_and_repair：达到 max_iterations 时安全收敛、不死循环
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent_workflow.generator import WorkflowValidator
from src.common.types import Workflow, WorkflowStep


def _step(sid, deps=None, io_in=None, io_out=None):
    return WorkflowStep(
        step_id=sid,
        agent_id="agent_x",
        action=f"step {sid}",
        inputs={
            "input_schema": io_in or {},
            "output_schema": io_out or {},
        },
        depends_on=list(deps or []),
    )


class _FakeLLM:
    """可编程的 LLM stub：依次返回预设的 completeness 响应。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def structured_output(self, prompt, schema, system=None, **kwargs):
        self.calls += 1
        if self.responses:
            return self.responses.pop(0)
        return {"score": 1.0, "missing": []}


def test_repair_cycle_drops_offending_edge():
    wf = Workflow(task="t", steps=[
        _step("a", deps=["c"]),
        _step("b", deps=["a"]),
        _step("c", deps=["b"]),  # a → b → c → a
    ])
    v = WorkflowValidator(llm_client=None)
    assert not v.check_dag_acyclic(wf)
    changed = v.repair_cycle(wf)
    assert changed is True
    assert v.check_dag_acyclic(wf), "修复后应为合法 DAG"


def test_repair_cycle_idempotent_on_acyclic_graph():
    wf = Workflow(task="t", steps=[_step("a"), _step("b", deps=["a"])])
    v = WorkflowValidator(llm_client=None)
    assert v.repair_cycle(wf) is False  # 无需修复


def test_repair_interface_drops_dangling_dep():
    wf = Workflow(task="t", steps=[
        _step("a"),
        _step("b", deps=["a", "ghost"]),  # ghost 不存在
    ])
    v = WorkflowValidator(llm_client=None)
    errors = v.check_io_compatibility(wf)
    assert any("ghost" in e for e in errors)
    changed = v.repair_interface(wf)
    assert changed is True
    assert "ghost" not in wf.steps[1].depends_on
    # 修复后应无 IO 错误
    assert v.check_io_compatibility(wf) == []


def test_repair_interface_drops_type_mismatch():
    wf = Workflow(task="t", steps=[
        _step("a", io_out={"out": "image"}),
        _step("b", deps=["a"], io_in={"in": "tabular_csv"}),
    ])
    v = WorkflowValidator(llm_client=None)
    errors = v.check_io_compatibility(wf)
    assert errors, "应识别出类型不兼容"
    changed = v.repair_interface(wf)
    assert changed is True
    assert "a" not in wf.steps[1].depends_on


def test_repair_coverage_appends_missing_steps():
    wf = Workflow(task="t", steps=[_step("a")])
    v = WorkflowValidator(llm_client=None)
    changed = v.repair_coverage(wf, missing=["生成报告", "通知调度"])
    assert changed is True
    assert len(wf.steps) == 3
    assert wf.steps[-2].agent_id == "__fallback__"
    assert wf.steps[-1].depends_on  # 应链到前一个修复步骤


def test_validate_and_repair_converges():
    wf = Workflow(task="t", steps=[
        _step("a", deps=["c"]),
        _step("b", deps=["a"]),
        _step("c", deps=["b"]),
    ])
    fake = _FakeLLM([
        # 第一轮：环路存在 → check_completeness 不会被调用，因为 is_dag=False
        # 第二轮：DAG 合法但 completeness 通过
        {"score": 0.9, "missing": []},
        {"score": 0.9, "missing": []},
    ])
    v = WorkflowValidator(llm_client=fake)
    outcome = v.validate_and_repair(wf, task="t", max_iterations=5)
    assert outcome["final_report"]["is_dag"] is True
    assert outcome["iterations"] >= 1


def test_validate_and_repair_respects_max_iterations():
    """构造一个永远缺步骤的场景，确认到达 max_iterations 不会死循环。"""
    wf = Workflow(task="t", steps=[_step("a")])
    fake = _FakeLLM([
        {"score": 0.0, "missing": ["x"]} for _ in range(20)
    ])
    v = WorkflowValidator(llm_client=fake)
    outcome = v.validate_and_repair(wf, task="t", max_iterations=3)
    assert outcome["iterations"] <= 4  # 含最后一次的最终 report
    # 若未收敛，converged 必须为 False
    assert outcome["converged"] is False


if __name__ == "__main__":
    test_repair_cycle_drops_offending_edge()
    test_repair_cycle_idempotent_on_acyclic_graph()
    test_repair_interface_drops_dangling_dep()
    test_repair_interface_drops_type_mismatch()
    test_repair_coverage_appends_missing_steps()
    test_validate_and_repair_converges()
    test_validate_and_repair_respects_max_iterations()
    print("all WorkflowValidator repair tests passed.")

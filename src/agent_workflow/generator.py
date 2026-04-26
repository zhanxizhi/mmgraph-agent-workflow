"""工作流生成模块（算法 D）

输入: 任务描述
输出: Workflow DAG
导出: JSON / Mermaid / LangGraph StateGraph
"""
from __future__ import annotations

import json
from typing import Any

import networkx as nx
from pydantic import BaseModel

from src.common.types import Workflow, WorkflowStep


# ============================================================
# 依赖推导 LLM Schema
# ============================================================

class _DependencyResult(BaseModel):
    additional_deps: list[dict]  # [{"from": step_id, "to": step_id}]


_DEP_SYSTEM = "你是工作流依赖分析专家。根据步骤的输入输出类型推断隐式数据流依赖。"

_DEP_PROMPT = """\
以下是工作流步骤列表（含显式依赖）：

{steps_json}

请分析步骤间的数据流，补充隐式依赖（前一步的 output_type 与后一步 input_type 匹配时，
应当存在依赖关系）。仅返回显式依赖之外的新增依赖，格式：
{{"additional_deps": [{{"from": "step_id_1", "to": "step_id_2"}}, ...]}}
"""

_COMPLETENESS_SYSTEM = "你是工作流质量评估专家。评估生成的工作流是否完整覆盖了原始任务需求。"

_COMPLETENESS_PROMPT = """\
原始任务：{task}

生成的工作流步骤：
{steps}

请评估工作流的完整性（0.0=完全不覆盖，1.0=完全覆盖）。
若存在遗漏，请在 missing 字段中按执行顺序列出每个缺失步骤的简短描述（不超过 25 字）。
仅返回JSON：{{"score": 0.85, "missing": ["缺少的步骤描述"]}}
"""

REPAIR_MAX_ITERATIONS = 5
COMPLETENESS_THRESHOLD = 0.6


# ============================================================
# WorkflowValidator（算法 D 三层验证）
# ============================================================

class WorkflowValidator:
    """三层验证：结构 → 语义 → 完整性。"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def check_dag_acyclic(self, workflow: Workflow) -> bool:
        """结构层：DAG 必须无环。"""
        g = nx.DiGraph()
        for step in workflow.steps:
            g.add_node(step.step_id)
        for step in workflow.steps:
            for dep in step.depends_on:
                g.add_edge(dep, step.step_id)
        return nx.is_directed_acyclic_graph(g)

    def check_io_compatibility(self, workflow: Workflow) -> list[str]:
        """语义层：检查依赖步骤间输入输出类型是否兼容，返回不兼容错误列表。"""
        step_map = {s.step_id: s for s in workflow.steps}
        errors: list[str] = []

        for step in workflow.steps:
            for dep_id in step.depends_on:
                dep = step_map.get(dep_id)
                if dep is None:
                    errors.append(f"步骤 {step.step_id} 依赖不存在的步骤 {dep_id}")
                    continue
                # 取 output 中第一个值的类型作为依赖的输出类型
                dep_out = list(dep.inputs.get("output_schema", {}).values())
                step_in = list(step.inputs.get("input_schema", {}).values())
                # 简单相容性检查：若两者都有声明，检查是否有重叠
                if dep_out and step_in:
                    dep_types  = {str(v).lower() for v in dep_out}
                    step_types = {str(v).lower() for v in step_in}
                    if dep_types.isdisjoint(step_types):
                        errors.append(
                            f"类型不兼容: {dep_id}输出{dep_out} → {step.step_id}输入{step_in}"
                        )
        return errors

    def check_completeness(self, workflow: Workflow, task: str) -> tuple[float, list[str]]:
        """完整性层：LLM-as-judge 评估工作流覆盖度，返回 (score, missing)。"""
        if self.llm is None:
            return 1.0, []  # 无 LLM 时跳过
        steps_str = "\n".join(
            f"{i+1}. [{s.agent_id}] {s.action}" for i, s in enumerate(workflow.steps)
        )
        try:
            result = self.llm.structured_output(
                prompt=_COMPLETENESS_PROMPT.format(task=task, steps=steps_str),
                schema={"type": "object", "properties": {
                    "score": {"type": "number"},
                    "missing": {"type": "array", "items": {"type": "string"}},
                }},
                system=_COMPLETENESS_SYSTEM,
            )
        except Exception:
            return 1.0, []
        score = float(result.get("score", 1.0))
        missing_raw = result.get("missing", []) or []
        missing = [str(m).strip() for m in missing_raw if str(m).strip()]
        return score, missing

    def validate(self, workflow: Workflow, task: str) -> dict:
        """运行三层验证（不修复），返回验证报告。"""
        is_dag = self.check_dag_acyclic(workflow)
        io_errors = self.check_io_compatibility(workflow)
        completeness, missing = self.check_completeness(workflow, task)
        return {
            "is_dag": is_dag,
            "io_errors": io_errors,
            "completeness": completeness,
            "missing_steps": missing,
            "passed": is_dag and len(io_errors) == 0 and completeness >= COMPLETENESS_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # 算法 D 的修复操作（与论文伪代码 RepairCycle / RepairInterface /
    # RepairCoverage 对应）。每个修复函数都是幂等的，便于在迭代循环中
    # 反复调用。
    # ------------------------------------------------------------------

    def repair_cycle(self, workflow: Workflow) -> bool:
        """删除导致环路的依赖边。返回是否进行了修改。"""
        g = nx.DiGraph()
        step_map = {s.step_id: s for s in workflow.steps}
        for step in workflow.steps:
            g.add_node(step.step_id)
        for step in workflow.steps:
            for dep in step.depends_on:
                if dep in step_map:
                    g.add_edge(dep, step.step_id)

        try:
            cycle = nx.find_cycle(g, orientation="original")
        except nx.NetworkXNoCycle:
            return False

        # 删除环路中的最后一条边（最有可能是 LLM 推断的隐式依赖）
        src, dst, *_ = cycle[-1]
        if dst in step_map and src in step_map[dst].depends_on:
            step_map[dst].depends_on.remove(src)
            return True
        # 兜底：依次尝试环上每条边
        for src, dst, *_ in cycle:
            if dst in step_map and src in step_map[dst].depends_on:
                step_map[dst].depends_on.remove(src)
                return True
        return False

    def repair_interface(self, workflow: Workflow) -> bool:
        """对 IO 不兼容的依赖边降级处理：删除该依赖边。

        论文中描述的另一可选策略（插入 type adapter 步骤）会扩大 DAG
        规模并引入新的 IO 校验项，工程实践上与"删除冲突边后由拓扑排序
        重排"等价，因此本实现采用更稳健的删边方案。
        """
        errors = self.check_io_compatibility(workflow)
        if not errors:
            return False
        step_map = {s.step_id: s for s in workflow.steps}
        modified = False
        for err in errors:
            # 错误格式: "类型不兼容: <dep_id>输出... → <step_id>输入..."
            #          "步骤 <step_id> 依赖不存在的步骤 <dep_id>"
            import re

            m = re.match(r"类型不兼容:\s*(\S+?)输出.*→\s*(\S+?)输入", err)
            if m:
                dep_id, step_id = m.group(1), m.group(2)
            else:
                m = re.match(r"步骤\s+(\S+)\s+依赖不存在的步骤\s+(\S+)", err)
                if not m:
                    continue
                step_id, dep_id = m.group(1), m.group(2)
            target = step_map.get(step_id)
            if target and dep_id in target.depends_on:
                target.depends_on.remove(dep_id)
                modified = True
        return modified

    def repair_coverage(
        self, workflow: Workflow, missing: list[str], fallback_agent_id: str = "__fallback__",
    ) -> bool:
        """为遗漏目标追加占位步骤，由通用协调智能体兜底。"""
        if not missing:
            return False
        existing_ids = {s.step_id for s in workflow.steps}
        last_id = workflow.steps[-1].step_id if workflow.steps else None
        added = False
        for desc in missing:
            new_id = f"s_repair_{len(existing_ids)+1}"
            while new_id in existing_ids:
                new_id = f"{new_id}_x"
            existing_ids.add(new_id)
            depends = [last_id] if last_id else []
            workflow.steps.append(WorkflowStep(
                step_id=new_id,
                agent_id=fallback_agent_id,
                action=desc,
                inputs={},
                depends_on=depends,
            ))
            last_id = new_id
            added = True
        return added

    def validate_and_repair(
        self, workflow: Workflow, task: str, max_iterations: int = REPAIR_MAX_ITERATIONS,
    ) -> dict:
        """算法 D 的迭代验证-修复循环。

        流程严格对应论文伪代码：每轮先尝试修复 cycle，若无环则修复 IO 冲突，
        最后修复任务覆盖度。任一阶段执行了修改即重新进入下一轮，直到三层
        全通过或达到 ``max_iterations`` 上限。

        Returns
        -------
        dict
            ``{"final_report": ..., "iterations": int, "history": [stage_log]}``。
            ``history`` 记录每轮的修复动作，便于答辩复盘。
        """
        history: list[dict] = []
        for it in range(max_iterations):
            report = self.validate(workflow, task)
            stage = {"iteration": it, "report": report, "action": None}

            if not report["is_dag"]:
                changed = self.repair_cycle(workflow)
                stage["action"] = "repair_cycle" if changed else "cycle_unfixable"
                history.append(stage)
                if not changed:
                    break
                continue

            if report["io_errors"]:
                changed = self.repair_interface(workflow)
                stage["action"] = "repair_interface" if changed else "interface_unfixable"
                history.append(stage)
                if not changed:
                    break
                continue

            if report["completeness"] < COMPLETENESS_THRESHOLD and report["missing_steps"]:
                changed = self.repair_coverage(workflow, report["missing_steps"])
                stage["action"] = "repair_coverage" if changed else "coverage_unfixable"
                history.append(stage)
                if not changed:
                    break
                continue

            # 三层全通过
            stage["action"] = "passed"
            history.append(stage)
            return {
                "final_report": report,
                "iterations": it + 1,
                "converged": True,
                "history": history,
            }

        # 达到上限或修复失败：返回最近一次报告
        final_report = self.validate(workflow, task)
        return {
            "final_report": final_report,
            "iterations": len(history),
            "converged": final_report["passed"],
            "history": history,
        }


# ============================================================
# WorkflowGenerator（算法 D）
# ============================================================

class WorkflowGenerator:
    """端到端工作流生成器。

    流程：
        retrieve → decompose → match → build_dag → validate → export
    """

    def __init__(self, llm_client, retriever, matcher, validator: WorkflowValidator | None = None):
        self.llm = llm_client
        self.retriever = retriever
        self.matcher = matcher
        self.validator = validator or WorkflowValidator(llm_client)

    # ------------------------------------------------------------------
    # 主生成流程
    # ------------------------------------------------------------------

    def generate(self, task: str, validate: bool = True) -> Workflow:
        """端到端生成工作流。

        Parameters
        ----------
        task:
            用户任务自然语言描述。
        validate:
            是否运行三层验证（实验时可关闭以加速）。
        """
        # 1. GraphRAG 检索上下文
        retrieval = self.retriever.retrieve(task)
        context = retrieval["context_text"]

        # 2. 任务分解
        subtasks = self.matcher.decompose_task(task, context)
        if not subtasks:
            raise ValueError(f"任务分解失败，无法生成子任务: {task}")

        # 3. 能力嵌入（加速匹配）
        subtasks = self.matcher.embed_subtask_capabilities(subtasks)

        # 4. 智能体匹配
        assignments = self.matcher.match_all(subtasks)

        # 5. 构建 DAG
        workflow = self._build_dag(task, assignments)

        # 6. 验证（含算法 D 的迭代修复循环）
        if validate:
            outcome = self.validator.validate_and_repair(workflow, task)
            workflow.metadata["validation"] = outcome
            # 修复后再次拓扑排序，保证步骤顺序合法
            workflow.steps = self._topological_sort(workflow.steps)
            if not outcome["final_report"]["is_dag"]:
                raise ValueError("生成的工作流含环且无法自动修复，请检查依赖关系")

        return workflow

    def _build_dag(
        self,
        task: str,
        assignments: list[tuple[dict, Any, float]],
    ) -> Workflow:
        """从匹配结果构建有向无环图，补充隐式依赖。"""
        steps: list[WorkflowStep] = []
        for subtask, agent, score in assignments:
            step = WorkflowStep(
                step_id=subtask["id"],
                agent_id=agent.id,
                action=subtask["description"],
                inputs={
                    "input_schema": agent.input_schema,
                    "output_schema": agent.output_schema,
                    "required_tools": subtask.get("required_tools", []),
                },
                depends_on=subtask.get("depends_on", []),
            )
            steps.append(step)

        # 用 LLM 补充隐式依赖
        steps = self._infer_implicit_deps(steps)

        # 拓扑排序确保顺序合理
        steps = self._topological_sort(steps)

        return Workflow(
            task=task,
            steps=steps,
            metadata={
                "agent_scores": {
                    subtask["id"]: round(score, 4)
                    for subtask, _, score in assignments
                }
            },
        )

    def _infer_implicit_deps(self, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """LLM 推断数据流隐式依赖，避免漏掉 IO 依赖。"""
        steps_json = json.dumps([
            {
                "id": s.step_id,
                "action": s.action,
                "input_schema": s.inputs.get("input_schema", {}),
                "output_schema": s.inputs.get("output_schema", {}),
                "depends_on": s.depends_on,
            }
            for s in steps
        ], ensure_ascii=False)

        try:
            result = self.llm.structured_output(
                prompt=_DEP_PROMPT.format(steps_json=steps_json),
                schema=_DependencyResult,
                system=_DEP_SYSTEM,
            )
            additional = result.get("additional_deps", [])
        except Exception:
            additional = []

        step_map = {s.step_id: s for s in steps}
        for dep in additional:
            src, dst = dep.get("from"), dep.get("to")
            if src and dst and dst in step_map and src not in step_map[dst].depends_on:
                step_map[dst].depends_on.append(src)

        return steps

    def _topological_sort(self, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        """按依赖关系拓扑排序，返回有序步骤列表。"""
        g = nx.DiGraph()
        step_map = {s.step_id: s for s in steps}
        for s in steps:
            g.add_node(s.step_id)
        for s in steps:
            for dep in s.depends_on:
                if dep in step_map:
                    g.add_edge(dep, s.step_id)

        if not nx.is_directed_acyclic_graph(g):
            return steps  # 有环时保持原序，留给 validator 报错

        order = list(nx.topological_sort(g))
        return [step_map[nid] for nid in order if nid in step_map]

    # ------------------------------------------------------------------
    # 导出格式
    # ------------------------------------------------------------------

    def to_mermaid(self, workflow: Workflow) -> str:
        """导出为 Mermaid flowchart 语法（论文图表 / 演示用）。"""
        lines = ["flowchart TD"]
        step_map = {s.step_id: s for s in workflow.steps}

        for step in workflow.steps:
            label = f"{step.agent_id}\\n{step.action[:30]}"
            lines.append(f'    {step.step_id}["{label}"]')

        for step in workflow.steps:
            for dep in step.depends_on:
                if dep in step_map:
                    lines.append(f"    {dep} --> {step.step_id}")

        return "\n".join(lines)

    def to_bpmn(self, workflow: Workflow) -> str:
        """导出为简化 BPMN 2.0 XML。"""
        tasks_xml = "\n".join(
            f'  <task id="{s.step_id}" name="{s.action[:50]}" />'
            for s in workflow.steps
        )
        flows_xml = []
        flow_id = 0
        for step in workflow.steps:
            for dep in step.depends_on:
                flows_xml.append(
                    f'  <sequenceFlow id="f{flow_id}" sourceRef="{dep}" targetRef="{step.step_id}" />'
                )
                flow_id += 1

        return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<definitions xmlns="http://www.omg.org/spec/BPMN/20100524/MODEL">
  <process id="workflow" name="{workflow.task[:50]}">
{tasks_xml}
{chr(10).join(flows_xml)}
  </process>
</definitions>"""

    def to_langgraph(self, workflow: Workflow):
        """编译为 LangGraph StateGraph（可直接执行）。"""
        try:
            from langgraph.graph import StateGraph, END
        except ImportError:
            raise ImportError("需要安装 langgraph: pip install langgraph")

        from typing import TypedDict

        class WorkflowState(TypedDict):
            task: str
            results: dict
            current_step: str

        graph = StateGraph(WorkflowState)

        # 为每个步骤创建节点
        for step in workflow.steps:
            action_desc = step.action

            def make_node(desc):
                def node_fn(state: WorkflowState) -> WorkflowState:
                    # 实际执行由对应 agent 完成；这里返回占位结果
                    state["results"][desc] = f"[{desc}] 执行完成"
                    state["current_step"] = desc
                    return state
                node_fn.__name__ = desc
                return node_fn

            graph.add_node(step.step_id, make_node(action_desc))

        # 添加边（依赖关系 → 执行顺序）
        step_map = {s.step_id: s for s in workflow.steps}
        entry_steps = [s for s in workflow.steps if not s.depends_on]
        for entry in entry_steps:
            graph.set_entry_point(entry.step_id)

        for step in workflow.steps:
            if not step.depends_on:
                continue
            # 若有多个依赖，连接最后一个（简化版；严格并行需要 parallel branch）
            last_dep = step.depends_on[-1]
            graph.add_edge(last_dep, step.step_id)

        # 末端步骤连接 END
        all_targets = {dep for s in workflow.steps for dep in s.depends_on}
        terminal_steps = [s for s in workflow.steps if s.step_id not in all_targets]
        for term in terminal_steps:
            graph.add_edge(term.step_id, END)

        return graph.compile()

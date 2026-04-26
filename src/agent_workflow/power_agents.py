"""电力系统场景预定义智能体

对应三个实验场景：
  OPS → 电网故障排查
  KQA → 电力知识问答
  DA  → 电力数据分析流水线
"""
from __future__ import annotations

from src.common.types import AgentSpec
from src.agent_workflow.feature_graph import AgentFeatureGraph


def build_power_feature_graph() -> AgentFeatureGraph:
    """构建并返回电力系统场景的智能体特征图。"""
    fg = AgentFeatureGraph()

    # ── OPS 场景：电网故障排查 ────────────────────────────────────

    fg.register(AgentSpec(
        id="fault_detector",
        name="故障检测智能体",
        description="实时检测电网保护装置动作、告警信号，定位故障区段",
        capabilities=["fault_detection", "alarm_analysis", "protection_relay_check"],
        tools=["scada_api", "pms_query", "alarm_filter"],
        input_schema={"alarm_signals": "list", "time_range": "str"},
        output_schema={"fault_location": "str", "fault_type": "str", "severity": "str"},
    ))

    fg.register(AgentSpec(
        id="power_flow_analyzer",
        name="潮流分析智能体",
        description="运行潮流计算，分析线路负载率、节点电压越限情况",
        capabilities=["power_flow_analysis", "voltage_check", "overload_detection"],
        tools=["pandapower", "pypower", "matpower"],
        input_schema={"grid_topology": "dict", "load_data": "dict"},
        output_schema={"bus_voltages": "dict", "line_loadings": "dict", "violations": "list"},
    ))

    fg.register(AgentSpec(
        id="topology_analyzer",
        name="电网拓扑分析智能体",
        description="解析电网单线图，识别开关状态、设备连接关系",
        capabilities=["topology_analysis", "network_reconstruction", "switch_state_check"],
        tools=["cim_parser", "networkx", "graph_query"],
        input_schema={"cim_file": "str", "substation_id": "str"},
        output_schema={"connected_components": "list", "island_status": "str", "switch_states": "dict"},
    ))

    fg.register(AgentSpec(
        id="protection_checker",
        name="继电保护检查智能体",
        description="校验保护定值配置，分析保护动作的合理性",
        capabilities=["protection_relay_check", "coordination_analysis", "setting_validation"],
        tools=["pms_query", "relay_config_reader"],
        input_schema={"relay_id": "str", "fault_current": "float"},
        output_schema={"is_correct_action": "bool", "coordination_issues": "list", "recommendation": "str"},
    ))

    fg.register(AgentSpec(
        id="restoration_planner",
        name="恢复操作规划智能体",
        description="根据故障情况制定电网恢复操作票，给出倒闸操作顺序",
        capabilities=["restoration_planning", "switching_sequence", "risk_assessment"],
        tools=["oms_api", "pms_query", "rule_engine"],
        input_schema={"fault_info": "dict", "grid_state": "dict"},
        output_schema={"operation_ticket": "list", "estimated_restore_time": "str", "risk_level": "str"},
    ))

    # ── KQA 场景：电力知识问答 ────────────────────────────────────

    fg.register(AgentSpec(
        id="knowledge_retriever",
        name="电力知识检索智能体",
        description="从电力技术规程、标准和手册中检索相关知识",
        capabilities=["knowledge_retrieval", "standard_lookup", "technical_qa"],
        tools=["graph_query", "vector_search", "document_reader"],
        input_schema={"question": "str", "domain": "str"},
        output_schema={"answer": "str", "sources": "list", "confidence": "float"},
    ))

    fg.register(AgentSpec(
        id="regulation_checker",
        name="规程合规检查智能体",
        description="检查操作方案是否符合电力安全规程和调度规程",
        capabilities=["regulation_check", "compliance_validation", "safety_assessment"],
        tools=["rule_engine", "graph_query", "document_reader"],
        input_schema={"operation_plan": "str", "regulation_type": "str"},
        output_schema={"is_compliant": "bool", "violations": "list", "suggestions": "str"},
    ))

    # ── DA 场景：电力数据分析流水线 ───────────────────────────────

    fg.register(AgentSpec(
        id="load_forecaster",
        name="负荷预测智能体",
        description="基于历史负荷和气象数据进行短期/中期负荷预测",
        capabilities=["load_forecasting", "time_series_analysis", "weather_correlation"],
        tools=["pandas", "prophet", "sklearn", "opsd_api"],
        input_schema={"historical_load": "dataframe", "weather_data": "dataframe"},
        output_schema={"forecast": "dataframe", "mape": "float", "confidence_interval": "dict"},
    ))

    fg.register(AgentSpec(
        id="anomaly_detector",
        name="电力数据异常检测智能体",
        description="检测电力时序数据中的异常点、数据缺失和计量异常",
        capabilities=["anomaly_detection", "data_quality_check", "outlier_detection"],
        tools=["pandas", "scipy", "sklearn", "pyod"],
        input_schema={"meter_data": "dataframe", "threshold_config": "dict"},
        output_schema={"anomalies": "list", "quality_score": "float", "cleaned_data": "dataframe"},
    ))

    fg.register(AgentSpec(
        id="renewable_analyzer",
        name="新能源出力分析智能体",
        description="分析风电、光伏出力特性，计算消纳率和弃电量",
        capabilities=["renewable_analysis", "curtailment_calculation", "capacity_factor"],
        tools=["pandas", "matplotlib", "opsd_api", "entsoe_api"],
        input_schema={"generation_data": "dataframe", "capacity": "float"},
        output_schema={"capacity_factor": "float", "curtailment_rate": "float", "summary_chart": "image"},
    ))

    fg.register(AgentSpec(
        id="report_generator",
        name="电力报告生成智能体",
        description="整合分析结果，生成电网运行分析报告或故障分析报告",
        capabilities=["report_generation", "data_visualization", "summary_writing"],
        tools=["matplotlib", "pandas", "jinja2"],
        input_schema={"analysis_results": "dict", "report_type": "str"},
        output_schema={"report_text": "str", "charts": "list", "recommendations": "list"},
    ))

    return fg

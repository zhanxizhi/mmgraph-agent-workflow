"""OPS 场景预定义智能体（运维故障排查）

对应实验方案文档场景一：K8s / Ansible 运维。
"""
from __future__ import annotations

from src.common.types import AgentSpec
from src.agent_workflow.feature_graph import AgentFeatureGraph


def build_ops_feature_graph() -> AgentFeatureGraph:
    """构建并返回 OPS 场景的智能体特征图。"""
    fg = AgentFeatureGraph()

    fg.register(AgentSpec(
        id="log_analyzer",
        name="日志分析智能体",
        description="分析容器和系统日志，识别错误模式和异常",
        capabilities=["log_analysis", "pattern_recognition", "error_detection"],
        tools=["kubectl", "grep", "awk"],
        input_schema={"log_source": "str", "time_range": "str"},
        output_schema={"error_patterns": "list", "summary": "str"},
    ))

    fg.register(AgentSpec(
        id="pod_inspector",
        name="Pod 状态检查智能体",
        description="检查 Pod 运行状态、事件和资源使用",
        capabilities=["pod_inspection", "resource_monitoring", "status_check"],
        tools=["kubectl", "k9s"],
        input_schema={"pod_name": "str", "namespace": "str"},
        output_schema={"status": "str", "events": "list", "resource_usage": "dict"},
    ))

    fg.register(AgentSpec(
        id="config_validator",
        name="配置验证智能体",
        description="验证 Kubernetes 配置文件、ConfigMap 和 Secret 是否正确",
        capabilities=["config_validation", "schema_check", "secret_management"],
        tools=["kubectl", "helm", "kubeval"],
        input_schema={"config_path": "str", "resource_type": "str"},
        output_schema={"is_valid": "bool", "errors": "list", "warnings": "list"},
    ))

    fg.register(AgentSpec(
        id="network_debugger",
        name="网络诊断智能体",
        description="诊断 Service、Ingress 和网络策略问题",
        capabilities=["network_diagnosis", "connectivity_test", "dns_check"],
        tools=["kubectl", "curl", "nslookup", "tcpdump"],
        input_schema={"service_name": "str", "namespace": "str"},
        output_schema={"reachable": "bool", "endpoints": "list", "latency_ms": "float"},
    ))

    fg.register(AgentSpec(
        id="resource_monitor",
        name="资源监控智能体",
        description="监控集群节点和 Pod 的 CPU/内存资源使用情况",
        capabilities=["resource_monitoring", "capacity_planning", "alert_detection"],
        tools=["kubectl", "prometheus", "grafana"],
        input_schema={"target": "str", "metric_type": "str"},
        output_schema={"cpu_usage": "float", "memory_usage": "float", "alerts": "list"},
    ))

    fg.register(AgentSpec(
        id="deployment_manager",
        name="部署管理智能体",
        description="管理 Deployment 的滚动更新、回滚和扩缩容",
        capabilities=["deployment_management", "rollback", "scaling"],
        tools=["kubectl", "helm", "argo"],
        input_schema={"deployment_name": "str", "namespace": "str", "action": "str"},
        output_schema={"success": "bool", "new_version": "str", "replica_count": "int"},
    ))

    fg.register(AgentSpec(
        id="image_checker",
        name="镜像检查智能体",
        description="检查容器镜像是否存在、可拉取，并分析镜像配置",
        capabilities=["image_inspection", "registry_check", "vulnerability_scan"],
        tools=["docker", "skopeo", "trivy"],
        input_schema={"image": "str", "registry": "str"},
        output_schema={"exists": "bool", "size_mb": "float", "vulnerabilities": "list"},
    ))

    fg.register(AgentSpec(
        id="root_cause_analyzer",
        name="根因分析智能体",
        description="综合多源信息进行根因分析，输出故障报告和修复建议",
        capabilities=["root_cause_analysis", "report_generation", "recommendation"],
        tools=["kubectl", "grep", "awk"],
        input_schema={"symptoms": "list", "logs": "str", "events": "list"},
        output_schema={"root_cause": "str", "fix_steps": "list", "report": "str"},
    ))

    return fg

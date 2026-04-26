#!/usr/bin/env bash
# 下载电力系统实验数据集
# 用法: bash scripts/download_datasets.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"/{kqa,graph_base,cim_graph,ops}

echo "======================================================"
echo "  电力系统数据集下载"
echo "  目标目录: $RAW"
echo "======================================================"

# ── KQA 场景：PowerGridQA ────────────────────────────────
echo ""
echo "[1/3] 克隆 PowerGridQA..."
if [ -d "$RAW/kqa/.git" ]; then
    echo "  已存在，跳过"
else
    git clone --depth 1 https://github.com/Hannaancode/PowerGridQA "$RAW/kqa"
fi

# ── 图谱底座：Electricity Knowledge Graph ────────────────
echo ""
echo "[2/3] 克隆 energy-knowledge-graph..."
if [ -d "$RAW/graph_base/.git" ]; then
    echo "  已存在，跳过"
else
    git clone --depth 1 https://github.com/sensorlab/energy-knowledge-graph "$RAW/graph_base"
fi

# ── CIM 标准图库 ──────────────────────────────────────────
echo ""
echo "[3/3] 克隆 CIM-Graph..."
if [ -d "$RAW/cim_graph/.git" ]; then
    echo "  已存在，跳过"
else
    git clone --depth 1 https://github.com/PNNL-CIM-Tools/CIM-Graph "$RAW/cim_graph"
fi

echo ""
echo "======================================================"
echo "  GitHub 仓库下载完成！"
echo ""
echo "  OPSD 时序数据（DA 场景）需手动下载："
echo "    https://data.open-power-system-data.org/time_series/"
echo "    推荐文件: time_series_60min_singleindex.csv"
echo "    保存到:   $RAW/opsd/"
echo ""
echo "  ENTSO-E Transparency Platform（DA/OPS 扩展场景）需按平台权限手动导出："
echo "    https://transparency.entsoe.eu/"
echo "    保存到:   $RAW/entsoe/"
echo ""
echo "  Texas A&M 电网测试案例（OPS/KQA 扩展场景）需按案例页面手动下载："
echo "    https://electricgrids.engr.tamu.edu/electric-grid-test-cases/"
echo "    保存到:   $RAW/tamu_grids/"
echo ""
echo "  US 异常报告（OPS 场景）需手动下载："
echo "    https://data.openei.org/submissions/6458"
echo "    保存到:   $RAW/ops/"
echo "======================================================"

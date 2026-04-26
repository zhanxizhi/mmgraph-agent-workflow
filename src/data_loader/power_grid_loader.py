"""电力系统数据集加载器

将原始数据集转换为 MultimodalGraphBuilder.build() 接受的 sources 格式：
    [{"type": "text", "content": "...", "meta": {...}}, ...]

支持数据集：
- PowerGridQA (KQA 场景): Nerc / Theory / Reasoning 三类 JSONL
- OPSD (DA 场景): 时序 CSV（可选，需手动下载）
- OPS: 从 PowerGridQA NERC 报告中抽取故障场景

已核查但当前 loader 尚未直接解析的数据源：
- ENTSO-E Transparency Platform: 可作为 DA/OPS 的实时或历史运行数据扩展
- Texas A&M Electric Grid Test Cases: 可作为合成电网拓扑与潮流案例扩展
- OpenEI/OEDI outage dataset: 可作为美国停电事件与异常报告扩展
- Electricity Knowledge Graph 与 CIM-Graph: 当前主要用于图谱 schema/术语参考
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator


# ── PowerGridQA ────────────────────────────────────────────────────────────

class PowerGridQALoader:
    """加载 PowerGridQA 数据集，生成 sources / benchmark 两种格式。"""

    _FILES = {
        "nerc":      "Nerc questions.jsonl",
        "theory":    "Questions Power System Theory.jsonl",
        "reasoning": "Reasoning questions.jsonl",
    }

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _iter_file(self, filename: str) -> Iterator[dict]:
        path = self.data_dir / filename
        if not path.exists():
            return
        # utf-8-sig strips BOM if present
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "question" in obj:
                        yield obj
                except json.JSONDecodeError:
                    continue

    def load_sources(
        self,
        splits: list[str] | None = None,
        max_per_split: int = 500,
        scene: str = "kqa",
    ) -> list[dict]:
        """返回 sources 列表，用于构建知识图谱。

        每条 Q&A 对拼接为一段连贯文本，附带元数据。

        Parameters
        ----------
        splits:       选择子集，默认全部 ("nerc", "theory", "reasoning")
        max_per_split: 每个子集最多取多少条（避免图谱过大）
        scene:        写入 meta["scene"]
        """
        splits = splits or list(self._FILES.keys())
        sources: list[dict] = []
        for split in splits:
            filename = self._FILES.get(split)
            if not filename:
                continue
            items = list(self._iter_file(filename))
            if len(items) > max_per_split:
                random.seed(42)
                items = random.sample(items, max_per_split)
            for item in items:
                if not item.get("answer"):
                    continue
                content = f"问：{item['question']}\n答：{item['answer']}"
                sources.append({
                    "type": "text",
                    "content": content,
                    "meta": {
                        "scene": scene,
                        "split": split,
                        "question": item["question"],
                    },
                })
        return sources

    def load_benchmark(
        self,
        splits: list[str] | None = None,
        n: int = 50,
    ) -> list[dict]:
        """返回基准任务列表，用于评估 KQA 场景。

        每条格式：{"task": str, "gold_answer": str, "split": str}
        """
        splits = splits or list(self._FILES.keys())
        pool: list[dict] = []
        for split in splits:
            filename = self._FILES.get(split)
            if not filename:
                continue
            items = list(self._iter_file(filename))
            for item in items:
                if not item.get("answer"):
                    continue
                pool.append({
                    "task":        item["question"],
                    "gold_answer": item["answer"],
                    "split":       split,
                    "scene":       "kqa",
                })
        random.seed(42)
        if len(pool) > n:
            pool = random.sample(pool, n)
        return pool


# ── OPS 场景：从 NERC 报告抽取故障场景 ──────────────────────────────────────

class OPSScenarioLoader:
    """从 PowerGridQA NERC 子集中抽取故障排查场景作为 OPS 基准任务。"""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def _iter_nerc(self, max_items: int = 0) -> Iterator[dict]:
        path = self.data_dir / "Nerc questions.jsonl"
        if not path.exists():
            return
        with open(path, encoding="utf-8-sig") as f:
            count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and item.get("question") and item.get("answer"):
                        yield item
                        count += 1
                        if max_items and count >= max_items:
                            return
                except json.JSONDecodeError:
                    continue

    def load_sources(self, max_items: int = 200) -> list[dict]:
        """将 NERC Q&A 转为 OPS 场景的知识源（故障报告文本）。"""
        sources: list[dict] = []
        for item in self._iter_nerc(max_items=max_items):
            content = (
                f"故障场景：{item['question']}\n"
                f"处置方案：{item['answer']}"
            )
            sources.append({
                "type": "text",
                "content": content,
                "meta": {"scene": "ops", "split": "nerc"},
            })
        return sources

    def load_benchmark(self, n: int = 50) -> list[dict]:
        """返回 OPS 场景基准任务（以故障描述为 task，处置方案为 gold）。"""
        items: list[dict] = [
            {"task": item["question"], "gold_answer": item["answer"], "scene": "ops"}
            for item in self._iter_nerc()
        ]
        random.seed(42)
        if len(items) > n:
            items = random.sample(items, n)
        return items


# ── DA 场景：OPSD 时序（可选） ────────────────────────────────────────────

class OPSDLoader:
    """加载 OPSD 时序 CSV（需手动下载后使用）。

    文件路径: data/raw/opsd/time_series_60min_singleindex.csv
    """

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)

    @property
    def available(self) -> bool:
        return self.csv_path.exists()

    def load_sources(self, n_rows: int = 5000) -> list[dict]:
        """将 OPSD 时序数据的统计描述转换为文本 sources。"""
        if not self.available:
            return []
        try:
            import pandas as pd
        except ImportError:
            return []

        df = pd.read_csv(self.csv_path, nrows=n_rows, parse_dates=[0], index_col=0)
        sources: list[dict] = []
        for col in df.columns[:20]:  # 取前20列避免过多
            series = df[col].dropna()
            if series.empty:
                continue
            stats = series.describe().to_dict()
            content = (
                f"OPSD 时序指标：{col}\n"
                f"样本数: {int(stats.get('count', 0))}, "
                f"均值: {stats.get('mean', 0):.2f}, "
                f"标准差: {stats.get('std', 0):.2f}, "
                f"最小值: {stats.get('min', 0):.2f}, "
                f"最大值: {stats.get('max', 0):.2f}"
            )
            sources.append({
                "type": "text",
                "content": content,
                "meta": {"scene": "da", "column": col},
            })
        return sources

    def load_benchmark(self, n: int = 50) -> list[dict]:
        """生成 DA 场景基准任务（预定义数据分析任务）。"""
        templates = [
            "分析过去一周的负荷曲线，识别峰值时段",
            "计算太阳能发电的容量因子和弃光率",
            "检测电力时序数据中的异常点和数据缺失",
            "对未来24小时的负荷进行短期预测",
            "分析风电出力与气温的相关性",
            "生成本月电网运行分析报告",
            "识别负荷异常增长区域并预警",
            "计算各类新能源消纳率",
        ]
        tasks = []
        for i in range(min(n, len(templates))):
            tasks.append({
                "task":  templates[i % len(templates)],
                "scene": "da",
            })
        return tasks

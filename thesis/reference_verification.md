# 参考文献核查与修正记录

核查日期：2026-05-07

说明：本记录用于说明 `ref.bib` 中参考文献的真实性核查、网页资料处理和正式出版信息修正情况。核查重点包括：题名、作者、期刊/会议、卷期页码、DOI/arXiv 编号是否能够在官方或较权威来源中找到。

## 处理结论

- 中文论文 6 篇均可检索到对应期刊页面、DOI 页面或机构知识库记录，未发现明显伪造或无法检索条目。
- 不适合列入参考文献表的网页、软件主页、数据网页等条目已从 `ref.bib` 移出，并在正文中改为脚注引用。
- 已将能够确认正式会议版本的英文论文从 arXiv/预印本信息修正为正式会议或论文集信息。
- 未找到稳妥正式会议/期刊记录的条目继续保留 arXiv/技术报告形式，避免虚构卷期、页码或会议名。

## 中文论文核查结果

| BibTeX key | 核查结果 | 核查入口 |
|---|---|---|
| `luo2024llmeval` | 已核实，中文文献 | 北京大学机构知识库；题名“ 大语言模型评测综述 ”可检索 |
| `xu2016kgreview` | 已核实，中文文献 | https://www.juestc.uestc.edu.cn/cn/article/id/41 |
| `liu2016krl` | 已核实，中文文献 | https://crad.ict.ac.cn/cn/article/doi/10.7544/issn1000-1239.2016.20160020 |
| `xiao2022substationkg` | 已核实，中文文献 | https://www.cepc.com.cn/CN/10.12204/j.issn.1000-7229.2022.03.008 |
| `yao2024smartgridkg` | 已核实，中文文献；仍按网络首发/优先发表处理 | https://xddl.ncepujournal.com/article/doi/10.19725/j.cnki.1007-2322.2024.0314 |
| `cao2025kgllmreview` | 已核实，中文文献 | https://www.arocmag.cn/abs/2024.12.0532 |

## 已修正为正式出版信息的英文条目

| BibTeX key | 修正后的正式出版信息 | 核查入口 |
|---|---|---|
| `grag2024` | Findings of the Association for Computational Linguistics: NAACL 2025, 4145--4157, DOI: 10.18653/v1/2025.findings-naacl.232 | https://aclanthology.org/2025.findings-naacl.232/ |
| `wu2023autogen` | Proceedings of the 1st Conference on Language Modeling, 2024 | https://openreview.net/forum?id=BAakY1hNKS |
| `hong2023metagpt` | Proceedings of the International Conference on Learning Representations, 2024 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/6507b115562bb0a305f1958ccc87355a-Abstract-Conference.html |
| `asai2023selfrag` | Proceedings of the International Conference on Learning Representations, 2024 | https://openreview.net/forum?id=hSyW5go0v8 |
| `jeong2024adaptive` | NAACL 2024 Long Papers, 7036--7050, DOI: 10.18653/v1/2024.naacl-long.389 | https://aclanthology.org/2024.naacl-long.389/ |
| `gutierrez2024hipporag` | Advances in Neural Information Processing Systems, 37, 59532--59569, DOI: 10.52202/079017-1902 | https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract-Conference.html |
| `sarthi2024raptor` | Proceedings of the International Conference on Learning Representations, 2024 | https://openreview.net/forum?id=GN921JHCRw |
| `chen2024bgem3` | Findings of the Association for Computational Linguistics: ACL 2024, 2318--2335, DOI: 10.18653/v1/2024.findings-acl.137 | https://aclanthology.org/2024.findings-acl.137/ |
| `liu2023llava` | Advances in Neural Information Processing Systems, 36, 34892--34916 | https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html |
| `liu2023mmc` | NAACL 2024 Long Papers, 1287--1310, DOI: 10.18653/v1/2024.naacl-long.70 | https://aclanthology.org/2024.naacl-long.70/ |

## 保留为 arXiv/技术报告的条目

| BibTeX key | 处理结果 |
|---|---|
| `edge2024graphrag` | 可检索到 arXiv / Microsoft Research 信息，暂未改为正式期刊或会议 |
| `openai2023gpt4` | 技术报告，保留 arXiv 形式 |
| `yan2024crag` | 可检索到 arXiv，未找到稳妥的正式会议/期刊版本 |
| `taskweaver2024` | Microsoft Research 技术报告/项目论文，保留 arXiv 形式 |
| `doclayoutyolo2024` | 可检索到 arXiv，未找到稳妥的正式会议/期刊版本 |

## 已移至脚注的网页或软件类资料

以下条目不再放入参考文献表，而是在正文对应位置以脚注形式给出来源链接：

- `langgraph2024`
- `anthropic2024claude`
- `crewai2024`
- `nerc2023ll`
- `cimanticgraphs2024`
- `tamu2026gridcases`
- `openei2026outage`

## 编译验证

已运行：

```text
biber main
xelatex -interaction=nonstopmode main.tex
```

结果：参考文献成功生成；未发现 citation undefined、biblatex rerun 或 Biber 错误。日志中仍存在原模板相关的字体替换、overfull hbox、document class name 等非参考文献问题。

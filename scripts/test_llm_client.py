"""快速验证 LLMClient 是否正常工作。

用法:
    python scripts/test_llm_client.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from src.common.llm_client import LLMClient


def test_chat(client: LLMClient):
    print("\n=== test_chat ===")
    result = client.chat_simple("用一句话介绍 GraphRAG 是什么。")
    print(f"response: {result}")
    assert isinstance(result, str) and len(result) > 0
    print("PASS")


def test_embed(client: LLMClient):
    print("\n=== test_embed ===")
    print(f"  嵌入模式: {'本地 BGE-M3' if client.embed_model.startswith('local/') else 'API'}")
    print("  (首次运行会下载模型，约 2.3 GB，请耐心等待...)")
    texts = ["GraphRAG 是一种图增强检索方法", "LangGraph 用于编排智能体工作流"]
    embeddings = client.embed(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    print(f"  embed dim: {len(embeddings[0])}")
    print("PASS")


def test_structured_output(client: LLMClient):
    print("\n=== test_structured_output ===")

    class TaskDecomposition(BaseModel):
        subtasks: list[str]
        rationale: str

    result = client.structured_output(
        prompt='把"分析销售数据并生成月报"分解为3个子任务',
        schema=TaskDecomposition,
    )
    print(f"structured result: {result}")
    assert "subtasks" in result
    assert len(result["subtasks"]) > 0
    print("PASS")


if __name__ == "__main__":
    client = LLMClient()
    print(f"使用模型: {client.model}，嵌入模型: {client.embed_model}")

    test_chat(client)
    test_embed(client)
    test_structured_output(client)

    print("\n所有测试通过 ✓")

"""统一 LLM 调用接口，支持 OpenAI / DeepSeek / Qwen 切换 + 本地嵌入

Provider 路由规则（通过 model 前缀或环境变量）：
  - 默认：OPENAI_BASE_URL + OPENAI_API_KEY
  - model 以 "deepseek" 开头：DEEPSEEK_API_KEY，base_url = https://api.deepseek.com/v1
  - model 以 "qwen" 开头：QWEN_API_KEY，base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
  - 也可直接传 base_url / api_key 覆盖

嵌入模型说明：
  EMBED_MODEL=local/BAAI/bge-m3   → 使用 sentence-transformers 本地推理（推荐）
  EMBED_MODEL=text-embedding-3-small → 使用 OpenAI 兼容 API
"""
from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel


# ---------- provider 路由 ----------

_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "QWEN_API_KEY",
    },
}


def _resolve_client(model: str, base_url: str | None, api_key: str | None) -> OpenAI:
    """根据 model 名称前缀选择 provider，返回 OpenAI 兼容客户端。"""
    prefix = model.split("-")[0].lower()
    provider = _PROVIDER_DEFAULTS.get(prefix)

    resolved_base_url = base_url or (
        provider["base_url"] if provider else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )
    resolved_api_key = api_key or (
        os.getenv(provider["api_key_env"]) if provider else os.getenv("OPENAI_API_KEY")
    )

    if not resolved_api_key:
        env_var = provider["api_key_env"] if provider else "OPENAI_API_KEY"
        raise EnvironmentError(f"缺少 API Key，请设置环境变量 {env_var}")

    return OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)


def _is_local_embed(embed_model: str) -> bool:
    """EMBED_MODEL 以 'local/' 开头则使用本地 sentence-transformers。"""
    return embed_model.startswith("local/")


def _load_local_embedder(embed_model: str):
    """懒加载 sentence-transformers 模型，返回 SentenceTransformer 实例。"""
    model_name = embed_model[len("local/"):]  # 去掉 "local/" 前缀
    # 支持 EMBED_MODEL_PATH 指向本地已下载目录
    model_path = os.getenv("EMBED_MODEL_PATH") or model_name
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("本地嵌入需要 sentence-transformers，请运行: pip install sentence-transformers")
    return SentenceTransformer(model_path)


def _resolve_openai_embed_client() -> OpenAI:
    """API 嵌入走 OpenAI 兼容接口（OPENAI_API_KEY + OPENAI_BASE_URL）。"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("API 嵌入需要 OPENAI_API_KEY，若使用本地嵌入请将 EMBED_MODEL 设为 local/...")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


class LLMClient:
    """统一 LLM 调用接口。

    对话走 API（DeepSeek/OpenAI/Qwen），嵌入支持本地 BGE-M3 或 API 两种模式。

    Parameters
    ----------
    model:
        对话模型名称，默认读 LLM_MODEL 环境变量（fallback: gpt-4o-mini）。
    embed_model:
        嵌入模型。"local/BAAI/bge-m3" 使用本地推理，其他值走 OpenAI API。
        默认读 EMBED_MODEL 环境变量（fallback: local/BAAI/bge-m3）。
    temperature:
        对话温度，默认 0.0（实验可复现）。
    base_url:
        覆盖对话 provider 默认的 base_url。
    api_key:
        覆盖对话 provider 默认的 api_key。
    """

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature

        # 对话客户端（按 model 前缀路由 provider）
        self._client = _resolve_client(self.model, base_url, api_key)

        # 嵌入：本地模式或 API 模式，懒加载
        self.embed_model = embed_model or os.getenv("EMBED_MODEL", "local/BAAI/bge-m3")
        self._local_embedder = None   # 懒加载，首次 embed() 时初始化
        self._embed_client = None     # API 模式时使用

    # ------------------------------------------------------------------
    # 对话接口
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> str:
        """同步单轮对话，返回 assistant 文本内容。

        Parameters
        ----------
        messages:
            OpenAI 消息格式，例如 [{"role": "user", "content": "..."}]
        temperature:
            覆盖实例默认温度。
        max_tokens:
            最大生成 token 数。
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content

    def chat_simple(self, prompt: str, system: str | None = None, **kwargs) -> str:
        """便捷方法：单条 user 消息，可选 system prompt。"""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    # ------------------------------------------------------------------
    # 嵌入接口
    # ------------------------------------------------------------------

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """文本批量向量化，返回嵌入列表（与输入顺序一致）。

        自动根据 EMBED_MODEL 选择本地推理或 API 调用：
        - local/BAAI/bge-m3  → sentence-transformers 本地推理
        - text-embedding-*   → OpenAI 兼容 API

        Parameters
        ----------
        texts:
            待编码的文本列表。
        batch_size:
            每批处理的文本数（本地推理时控制显存，API 时控制请求大小）。
        """
        if not texts:
            return []

        if _is_local_embed(self.embed_model):
            return self._embed_local(texts, batch_size)
        else:
            return self._embed_api(texts, batch_size)

    def _embed_local(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """使用本地 sentence-transformers 推理。"""
        if self._local_embedder is None:
            self._local_embedder = _load_local_embedder(self.embed_model)

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = self._local_embedder.encode(batch, normalize_embeddings=True)
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    def _embed_api(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """使用 OpenAI 兼容 API 推理。"""
        if self._embed_client is None:
            self._embed_client = _resolve_openai_embed_client()

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._embed_client.embeddings.create(
                model=self.embed_model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_one(self, text: str) -> list[float]:
        """单条文本向量化的便捷方法。"""
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # 结构化输出
    # ------------------------------------------------------------------

    def structured_output(
        self,
        prompt: str,
        schema: type[BaseModel] | dict,
        system: str | None = None,
        max_tokens: int = 2048,
        **kwargs,
    ) -> dict[str, Any]:
        """强制 JSON 结构化输出。

        Parameters
        ----------
        prompt:
            用户 prompt。
        schema:
            Pydantic BaseModel 类（推荐）或 JSON Schema dict。
        system:
            可选的 system prompt。
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # OpenAI 原生 parse API 仅 OpenAI 支持；DeepSeek/Qwen 统一走 json_object 模式
        prefix = self.model.split("-")[0].lower()
        supports_parse = prefix not in ("deepseek", "qwen")

        if supports_parse and isinstance(schema, type) and issubclass(schema, BaseModel):
            return self._structured_output_pydantic(messages, schema, max_tokens, **kwargs)
        else:
            return self._structured_output_json_schema(messages, schema, max_tokens, **kwargs)

    def _structured_output_pydantic(
        self,
        messages: list[dict],
        schema: type[BaseModel],
        max_tokens: int,
        **kwargs,
    ) -> dict[str, Any]:
        """使用 OpenAI beta.chat.completions.parse（原生 Pydantic 支持）。"""
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=schema,
            max_tokens=max_tokens,
            **kwargs,
        )
        parsed = response.choices[0].message.parsed
        return parsed.model_dump() if parsed is not None else {}

    def _structured_output_json_schema(
        self,
        messages: list[dict],
        schema: type[BaseModel] | dict,
        max_tokens: int,
        **kwargs,
    ) -> dict[str, Any]:
        """回退方案：response_format=json_object，手动解析。兼容 Pydantic 类和 dict schema。"""
        # Pydantic 类转 JSON Schema dict
        schema_dict = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else schema
        # 把 schema 注入到最后一条 user 消息
        last = messages[-1]
        schema_hint = f"\n\n请严格按照以下 JSON Schema 返回，不要输出多余内容：\n{json.dumps(schema_dict, ensure_ascii=False)}"
        messages = messages[:-1] + [{"role": last["role"], "content": last["content"] + schema_hint}]

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=self.temperature,
            **kwargs,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        # 若传入了 Pydantic 类，用它验证并转回 dict（统一返回格式）
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(data).model_dump()
        return data

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        """从 YAML 配置字典创建实例。

        config 示例::

            llm:
              model: gpt-4o-mini
              temperature: 0.0
              max_tokens: 2048
        """
        llm_cfg = config.get("llm", config)
        return cls(
            model=llm_cfg.get("model"),
            temperature=llm_cfg.get("temperature", 0.0),
        )

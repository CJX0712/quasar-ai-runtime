"""L1 适配层：L0 契约的各种实现。

上层（L2/L3/L4）永远 import 契约而不是这里的类名；这里的每个类都只是
"某个契约的一种可选实现"，由 runtime/container.py 按配置注入。
"""

from ._http import local_client, probe_json, remote_client
from .bm25_index import Bm25LexicalIndex, tokenize
from .hash_embed import HashEmbedding
from .lexical_rerank import LexicalReranker
from .llm_rerank import LLMReranker
from .loaders import (
    CsvLoader,
    JsonLoader,
    PdfLoader,
    TextLoader,
    default_loaders,
    loader_for,
)
from .numpy_store import NumpyVectorStore
from .ollama_embed import OllamaEmbedding
from .ollama_llm import OllamaLLM
from .openai_compat import OpenAICompatLLM
from .qdrant_store import QdrantVectorStore
from .scripted_llm import ScriptedLLM
from .sqlite_memory import SqliteMemoryStore
from .trace_sinks import JsonlTrace, MemoryTrace, NullTrace

__all__ = [
    "local_client",
    "remote_client",
    "probe_json",
    "Bm25LexicalIndex",
    "tokenize",
    "HashEmbedding",
    "LexicalReranker",
    "LLMReranker",
    "TextLoader",
    "CsvLoader",
    "JsonLoader",
    "PdfLoader",
    "default_loaders",
    "loader_for",
    "NumpyVectorStore",
    "QdrantVectorStore",
    "OllamaEmbedding",
    "OllamaLLM",
    "OpenAICompatLLM",
    "ScriptedLLM",
    "SqliteMemoryStore",
    "JsonlTrace",
    "MemoryTrace",
    "NullTrace",
]

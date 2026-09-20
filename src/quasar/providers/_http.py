"""共享 HTTP 客户端工厂。

为什么需要这个文件：本机开着 HTTP_PROXY 代理（见 README 的环境排查一节）。
如果让 httpx 读环境变量，访问 127.0.0.1 的 Ollama 也会被送去代理，症状是
"Ollama 明明在跑却连不上"。所以本地服务一律 trust_env=False。
"""

from __future__ import annotations

import httpx

from ..contracts.errors import ProviderUnavailable


def local_client(base_url: str, timeout: float) -> httpx.AsyncClient:
    """访问本机服务（Ollama 等）——不读代理环境变量。"""
    return httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout, trust_env=False)


def remote_client(base_url: str, timeout: float, api_key: str = "") -> httpx.AsyncClient:
    """访问远端服务——允许读代理环境变量，并带上鉴权头。"""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    return httpx.AsyncClient(
        base_url=base_url.rstrip("/"), timeout=timeout, headers=headers, trust_env=True
    )


async def probe_json(client: httpx.AsyncClient, url: str) -> tuple[bool, str]:
    """探测一个 GET 端点，返回 (是否可用, 说明)。永不抛异常。"""
    try:
        resp = await client.get(url)
    except httpx.HTTPError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if resp.status_code >= 400:
        return False, f"HTTP {resp.status_code}"
    return True, "ok"


def wrap_transport_error(exc: Exception, target: str) -> ProviderUnavailable:
    return ProviderUnavailable(f"无法连接 {target}：{type(exc).__name__}: {exc}", target=target)

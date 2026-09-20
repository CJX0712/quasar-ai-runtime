"""Quasar 错误类型。

约定：所有可预期的失败都抛 QuasarError 的子类，并携带机器可读的 code，
便于上层（API 层）统一映射成 HTTP 状态码，也便于测试断言。
"""

from __future__ import annotations


class QuasarError(Exception):
    """所有 Quasar 异常的基类。"""

    code = "quasar_error"
    http_status = 500

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = dict(context)

    def to_dict(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, "context": self.context}


class ConfigError(QuasarError):
    code = "config_error"
    http_status = 400


class ProviderError(QuasarError):
    """模型/嵌入/重排等外部能力调用失败。"""

    code = "provider_error"
    http_status = 502


class ProviderUnavailable(ProviderError):
    """能力不可用（服务没起、模型没拉、网络不通）。"""

    code = "provider_unavailable"
    http_status = 503


class StoreError(QuasarError):
    code = "store_error"
    http_status = 500


class IngestError(QuasarError):
    code = "ingest_error"
    http_status = 400


class GuardRefusal(QuasarError):
    """护栏判定：无足够证据，拒绝作答。这是正常业务结果，不是故障。"""

    code = "guard_refusal"
    http_status = 200


class BudgetExceeded(QuasarError):
    code = "budget_exceeded"
    http_status = 429


class ToolError(QuasarError):
    code = "tool_error"
    http_status = 400

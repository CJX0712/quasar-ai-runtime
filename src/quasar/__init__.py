"""Quasar AI Runtime - 模块化多智能体 AI 运行时。

作者：晨星
许可：MIT

分层（依赖只向下，L0 是唯一依赖终点）：

    L4 接口层   interfaces/   CLI · HTTP API · Web 控制台
    L3 编排层   runtime/      DI 容器 · 链路装配 · 追踪
    L2 能力层   modules/      采集 · 检索 · 记忆 · 智能体 · 护栏 · 工具 · 评测
    L1 适配层   providers/    契约的多种实现（可替换）
    L0 契约层   contracts/    Protocol + DTO
"""

__version__ = "0.1.0"
__author__ = "晨星"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]

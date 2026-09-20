"""智能体的提示词模板。

单独成文件：提示词是产品行为的一部分（决定"不许瞎说"能不能落地），
把它埋在循环逻辑里会导致调提示词必须读懂控制流。
"""

from __future__ import annotations

SYSTEM_PROMPT = """你是 Quasar 知识库问答智能体。严格遵守以下规则：

1. 只依据"检索到的资料"或工具返回的材料作答，绝不使用资料之外的知识。
2. 每一句事实陈述后面必须标注来源编号，格式为 [n]，n 必须出现在材料的编号中。
3. 如果材料不足以回答问题，直接说明材料不足以回答，不要推测、不要补全。
4. 用简体中文作答，结论先行，简洁分点，不要复述问题本身。
5. 可以在需要时调用工具补充材料，但不要为了凑材料而调用。
"""

TOOL_HINT = """你可以调用工具补充检索材料。当已有材料不足以回答时，先调用 retrieval_search。"""


def evidence_block(rendered: str) -> str:
    return rendered


def retry_instruction(reason: str) -> str:
    """护栏判定不合格时，用一条诊断式指令让模型重写，而不是含糊地说"再改改"。"""
    return (
        f"你上一次的回答未通过证据校验。未通过原因：{reason}\n"
        "请重写回答，并确保：\n"
        "1) 每一个包含事实陈述的段落都必须带至少一个 [n] 引用；\n"
        "2) 引用的 n 必须是上面材料里真实存在的编号，不得编造；\n"
        "3) 引用的句子必须与对应材料的文字有明显重合，不要引用后再自由发挥。"
    )


def seed_instruction() -> str:
    """模型漏调工具时的兜底指令。

    llm 规划器把"要不要取证"交给了模型，而 7B 级模型常常直接开答。
    此时不能顺势拒答——那是把"模型没调工具"包装成"知识库里没有这个知识"。
    正确做法是系统替它把材料补齐，让这一问仍然能产出带引用的答案。
    """
    return (
        "你没有调用任何工具，因此系统替你完成了一次检索，材料见本条消息末尾。\n"
        "请只依据这些材料重新作答，结论先行，并在每一处事实陈述后标注 [n] 依据。"
    )


def no_evidence_hint() -> str:
    """拒答时补在原因后面的排障指引。"""
    return (
        "（agent.planner=llm：模型本轮未通过工具取得材料，兜底预检索也未命中。"
        "若希望每问都确定性预检索，把 agent.planner 改为 rules）"
    )


def memory_block(recent_lines: list[str], recalled_lines: list[str]) -> str:
    parts: list[str] = []
    if recent_lines:
        parts.append("最近的对话：\n" + "\n".join(recent_lines))
    if recalled_lines:
        parts.append("可能相关的历史轮次：\n" + "\n".join(recalled_lines))
    return "\n\n".join(parts)


__all__ = [
    "SYSTEM_PROMPT",
    "TOOL_HINT",
    "retry_instruction",
    "seed_instruction",
    "no_evidence_hint",
    "memory_block",
    "evidence_block",
]

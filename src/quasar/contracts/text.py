"""纯文本工具：分词、切句、重合度。

放在契约层而不是某个模块里的理由：它是零依赖的纯函数，同时被 L1（BM25 索引、
SQLite 记忆）和 L2（证据护栏、评测指标）使用。放到任何一侧都会逼出跨层 import，
而跨层 import 正是这套架构要消灭的东西。
"""

from __future__ import annotations

import re

_ASCII_WORD = re.compile(r"[A-Za-z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_SENTENCE = re.compile(r"(?<=[。！？；!?;\n])|(?<=\.\s)")

_CJK_STOP = set("的了是在和有与我你他她它们个之也就都而及与或很把被对为以于上下中")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "is", "are", "was", "were",
    "in", "on", "at", "for", "with", "as", "by", "be", "this", "that", "it",
    "from", "will", "can", "not", "has", "have", "had", "but", "if", "then",
}


def tokenize(text: str) -> list[str]:
    """CJK 单字 + 相邻双字 + 拉丁词；已去停用词。

    为什么不用 jieba：jieba 只发 sdist，需要 C 编译器，在"干净环境一次装好"
    这条硬约束下是负分项。单字 + 双字切分对 BM25 这类词袋模型足够用。
    """
    lowered = (text or "").lower()
    tokens = [w for w in _ASCII_WORD.findall(lowered) if w not in _STOPWORDS and len(w) > 1]
    for run in _CJK_RUN.findall(lowered):
        chars = [c for c in run if c not in _CJK_STOP]
        tokens.extend(chars)
        tokens.extend(chars[i] + chars[i + 1] for i in range(len(chars) - 1))
    return tokens


def tokenize_query(text: str) -> list[str]:
    """查询侧分词：当查询里存在多字词时，丢掉单字，只保留多字词。

    为什么查询侧要和文档侧不同：单字在中文里几乎没有区分度——"量"、"子"、"时"
    出现在几乎任何技术文档里。文档侧保留单字无害（它们帮助短文档被更大的
    查询命中），但查询侧保留单字会让**每一次检索都命中每一篇文档**，
    把"检索不到"这个事实彻底掩盖掉，于是相关性筛除与拒答全部失效。

    实测：查询「量子纠缠退相干时间」在一个只讲软件架构的语料上，
    查询侧不分词时命中 3 条（靠"量"、"时"这类单字），过滤后为 0 条。

    只有当查询里确实存在多字词时才过滤；纯单字查询（如"税"）保持原样，
    否则会把合法查询变成空查询。
    """
    tokens = tokenize(text)
    if not tokens:
        return []
    multi = [t for t in tokens if len(t) > 1]
    if not multi:
        return tokens
    return multi


def unique_tokens(text: str) -> set[str]:
    return set(tokenize(text))


def split_sentences(text: str) -> list[str]:
    """中英混排切句。用于护栏按"论断"粒度检查引用，而非按整段。"""
    parts = [p.strip() for p in _SENTENCE.split(text or "")]
    return [p for p in parts if p]


def overlap_ratio(source: str, reference: str) -> float:
    """source 的内容词有多少比例出现在 reference 里。区间 [0, 1]。

    分母用 source 的 token 数：我们关心的是"这句话有多少内容能在材料里找到出处"，
    而不是"材料有多少被这句话覆盖"。

    注意：两侧都用文档侧分词。要衡量**问题**与材料的相关度，必须用
    query_overlap_ratio，否则单字会把覆盖率抬高到失去区分度。
    """
    source_tokens = tokenize(source)
    if not source_tokens:
        return 0.0
    reference_tokens = set(tokenize(reference))
    if not reference_tokens:
        return 0.0
    hits = sum(1 for t in source_tokens if t in reference_tokens)
    return round(hits / len(source_tokens), 6)


def query_overlap_ratio(question: str, reference: str) -> float:
    """问题侧重合度：问题的内容词有多少出现在 reference 里。区间 [0, 1]。

    与 overlap_ratio 的唯一区别是问题侧走 tokenize_query（丢掉单字）。
    为什么必须区分，见 tokenize_query 的文档字符串——那里已经写明
    "查询侧保留单字会让每一次检索都命中每一篇文档……于是相关性筛除与拒答全部失效"。
    这句话对检索成立，对**护栏的相关性筛除同样成立**：护栏拿到的材料来自检索，
    但"材料覆不覆盖这个问题"这个判断仍然要用查询侧分词，否则单字（是/多/少/间）
    会把无关问题的覆盖率抬到门槛之上。

    实测（9 块语料，best_coverage）：

        用文档侧分词：真相关问题 0.571–0.818 · 语料外问题 0.143–0.360  ← 区间重叠
        用查询侧分词：真相关问题 0.375–0.600 · 语料外问题 0.000–0.167  ← 干净分开

    重叠的那一组意味着"相关性筛除"实际上拦不住"量子纠缠退相干时间"这类
    完全无关的问题（0.360 > 默认门槛 0.300）。
    """
    source_tokens = tokenize_query(question)
    if not source_tokens:
        return 0.0
    reference_tokens = set(tokenize(reference))
    if not reference_tokens:
        return 0.0
    hits = sum(1 for t in source_tokens if t in reference_tokens)
    return round(hits / len(source_tokens), 6)


def is_claim_like(text: str, *, min_chars: int = 12, min_tokens: int = 3) -> bool:
    """判断一段文字是否"在陈述事实"——标题、客套、引导语不算论断。

    规则：够长、内容词够多、且不以冒号结尾（以冒号结尾的多半是引出下文的标签行）。
    """
    stripped = (text or "").strip()
    if len(stripped) < min_chars:
        return False
    if stripped.endswith(("：", ":")):
        return False
    return len(unique_tokens(stripped)) >= min_tokens


def format_evidence(
    items,
    *,
    start_index: int = 1,
    max_chars: int = 700,
    heading: str = "以下是检索到的资料，回答时必须用 [n] 标注依据，且不得使用资料之外的信息：",
) -> str:
    """把检索结果渲染成带编号的材料块。

    编号格式 `[n] (chunk_id)` 是整条链路的事实标准：
      - 模型据此写引用；
      - 证据护栏据此把 [n] 映射回具体块；
      - 离线脚本模型据此构造可验证的答案。
    三处共用同一个渲染函数，格式漂移就不可能发生。
    """
    lines = [heading] if heading else []
    for offset, item in enumerate(items):
        chunk = getattr(item, "chunk", item)
        index = start_index + offset
        body = (chunk.text or "").strip().replace("\n", " ")
        if len(body) > max_chars:
            body = body[:max_chars] + "…"
        lines.append(f"[{index}] ({chunk.id})\n{body}")
    return "\n\n".join(lines)


__all__ = [
    "tokenize",
    "tokenize_query",
    "unique_tokens",
    "split_sentences",
    "overlap_ratio",
    "query_overlap_ratio",
    "is_claim_like",
    "format_evidence",
]

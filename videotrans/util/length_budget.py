# -*- coding: utf-8 -*-
"""
字幕翻译长度预算 (P0 优化)

根据目标语言的平均发声速率 (chars-per-second), 为每条字幕计算一个
"最多允许的译文字符数", 作为 LLM 翻译时的硬约束。这样可以从源头减少
"配音比字幕长" 的超出, 避免后续 atempo 变速导致的音质劣化。

主接口:
  - cps_for(target_code): 返回该语言的 chars/second
  - compute_budget_chars(duration_ms, target_code, safety=0.95): 返回 int
  - strip_budget_marker(text): 清掉 LLM 偶尔回带的 [≤N] 前缀
"""
from __future__ import annotations

import re
from typing import Optional

# 单位: 字符/秒 (含空格 & 标点). 保守估计, 宁可紧一点不留余量。
# 参考来源: TTS 实测 + 语言发声速率研究.
# 中日韩按"字/秒"(单字符密度高), 欧语按"字符/秒"(含空格)。
_CPS_TABLE = {
    # 中文: TTS 实测 4-5 字/秒
    'zh-cn': 5.0,
    'zh-tw': 5.0,
    'zh': 5.0,
    'yue': 5.0,
    # 日语假名+汉字
    'ja': 7.0,
    # 韩语
    'ko': 6.5,
    # 英语含空格
    'en': 14.0,
    # 印欧语系
    'fr': 13.5,
    'de': 12.5,
    'es': 14.0,
    'it': 13.5,
    'pt': 13.5,
    'nl': 13.0,
    'sv': 13.0,
    'pl': 12.0,
    'cs': 12.0,
    'hu': 12.0,
    'uk': 12.0,
    'ru': 12.0,
    # 亚洲
    'th': 12.0,
    'vi': 13.0,
    'id': 13.5,
    'ms': 13.5,
    'fil': 13.5,
    'hi': 12.0,
    'bn': 11.0,
    'ur': 12.0,
    'fa': 12.0,
    # 中东
    'ar': 11.0,
    'he': 12.0,
    'tr': 12.0,
    'kk': 12.0,
}

_DEFAULT_CPS = 13.0
# 绝对下限: 字幕再短, 译文也至少给 4 个字符的表达空间
_MIN_BUDGET = 4


def cps_for(target_code: Optional[str]) -> float:
    if not target_code:
        return _DEFAULT_CPS
    code = str(target_code).strip().lower()
    if code in _CPS_TABLE:
        return _CPS_TABLE[code]
    # 前缀回退: 'en-us' → 'en'
    primary = code.split('-')[0]
    if primary in _CPS_TABLE:
        return _CPS_TABLE[primary]
    return _DEFAULT_CPS


def compute_budget_chars(duration_ms: float, target_code: Optional[str], safety: float = 0.95) -> int:
    """
    duration_ms: 字幕槽位 = end_time - start_time (毫秒)
    target_code: 目标语言代码, 如 'en', 'zh-cn'
    safety: 安全系数 (留变速+自然停顿余量), 默认 0.95
    """
    if duration_ms is None or duration_ms <= 0:
        return _MIN_BUDGET
    cps = cps_for(target_code)
    budget = int(round((duration_ms / 1000.0) * cps * safety))
    return max(_MIN_BUDGET, budget)


_BUDGET_MARKER_RE = re.compile(r'^\s*\[\s*[≤<]=?\s*\d+\s*\]\s*', re.UNICODE)


def strip_budget_marker(text: str) -> str:
    """LLM 有时会把 [≤N] 原样回带, 这里剥掉。"""
    if not text:
        return text
    return _BUDGET_MARKER_RE.sub('', text, count=1)


# 注入到 prompt 头部的规则文本 (紧凑, 避免吃 token)
BUDGET_PROMPT_FRAGMENT = """
# PER-LINE CHARACTER BUDGET (HARD CONSTRAINT)
Each input line/block starts with a marker like `[≤44]`. That number is the **MAXIMUM** total characters allowed in your translation of that line/block (including spaces & punctuation).
Rules:
- Count characters. Never exceed the budget. If literal translation is too long, rephrase/condense until it fits.
- **NEVER output the `[≤N]` marker** in your translation.
- Preserving line count / block count is still the top priority. The budget applies **within** each existing line/block — do NOT split or merge to satisfy it.
- If you truly cannot compress further, get as close as possible; a tiny overshoot is tolerable, but large overshoots are a defect.
"""

# 情感/语气保留规则: 让 TTS 能吃到标点和强调信号, 减少配音生硬感
EMPHASIS_PROMPT_FRAGMENT = """
# EMOTION & EMPHASIS PRESERVATION (for TTS Prosody)
The translation will be fed to a TTS engine that uses punctuation as prosody cues. Preserve the source's emotional intensity:
- **Exclamation marks (!)**: if the source ends with `!`, your translation MUST end with `!` (never demote to `.`). Multiple `!!` → keep multiple.
- **Question marks (?)**: preserve as-is, including rhetorical questions.
- **Ellipsis (...)**: preserve for hesitation/trailing-off; do not replace with period.
- **Dashes (— or --)**: preserve for interruption/abrupt stops.
- **ALL-CAPS emphasis**: if the source uses uppercase or quotes for stress ("STOP IT"), mirror the emphasis in the target (use uppercase for Latin scripts, `「」` or quotes for CJK).
- **Interjections**: keep short emotive words (oh, wow, huh, no, hey, ah, ugh) and their target-language equivalents. These are prosody anchors — never drop them even under tight character budget.
- **Length constraint applies AFTER emotion preservation**: if the budget forces a cut, cut adjectives/filler words, NEVER cut punctuation or interjections.
"""

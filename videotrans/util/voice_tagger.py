# -*- coding: utf-8 -*-
"""
TTS 音色名 → 性别/年龄标签

纯字符串规则, 零外部依赖, 适用于音色名自带中文描述的引擎:
  doubao0/doubao2 (灿灿/擎苍/慈爱姥姥/...)
  azure (zh-CN-XiaoxiaoNeural 等命名含 female/male/youth 语义)
  edgetts (同 azure)
  qwen3local (Vivian/Uncle_fu/Sohee/...)

规则偏保守: 模糊项归 'any', 避免误分类导致男角色被迫分女声。
"""
from __future__ import annotations
import re
from typing import Optional, List, Tuple, Dict

# ------------------------------- 性别规则 (中文优先, 英文兜底)
# 顺序敏感: 先匹配更具体的 (如 "姥姥" 归 female 的同时不被 "大叔" 误判)
_FEMALE_PATTERNS = [
    r'女声|女生|小姐姐|姥姥|奶奶|淑女|少御|小美|公主|萝莉|大妈|大姐',
    r'灿灿|炀炀|梓梓|燃燃|甜美|甜宠|知性|温柔|亲切女|清新文艺女|鸡汤女|慈爱',
    r'童童|诚诚|懒小羊|(小|美)甜心|(小|美)女声',
    r'\bfemale\b|\bwoman\b|\bgirl\b|_f_|-f-|(?i)xiao[a-z]*neural',
    r'Vivian|Serena|Ono_anna|Sohee',
]
_MALE_PATTERNS = [
    r'男声|男生|大叔|青叔|小哥|小帅|少年|先生',
    r'擎苍|霸气|阳光青年|赘婿|古风|质朴|儒雅|开朗青年|译制片|活力解说男|解说小帅|智慧老者|说唱|道家|新闻',
    r'\bmale\b|\bman\b|\bboy\b|_m_|-m-',
    r'Uncle_fu|Dylan|Eric|Ryan|Aiden',
]

_GENDER_F_RE = re.compile('|'.join(_FEMALE_PATTERNS), re.IGNORECASE)
_GENDER_M_RE = re.compile('|'.join(_MALE_PATTERNS), re.IGNORECASE)

# ------------------------------- 年龄带
_AGE_RULES = [
    ('child',  r'童童|小朋友|萝莉|小公主|懒小羊|child|kid'),
    ('senior', r'姥姥|奶奶|老者|大爷|老先生|大妈|大叔|senior|elder'),
    ('adult',  r'青年|淑女|少御|小哥|小帅|小美|甜宠|淑女|青叔|先生|女声|男声|女生|男生|adult'),
    ('teen',   r'少年|少女|学生|teen'),
]
_AGE_RES = [(tag, re.compile(pat, re.IGNORECASE)) for tag, pat in _AGE_RULES]


def tag_voice(name: str) -> Dict[str, str]:
    """音色名 → {'gender': 'm'|'f'|'any', 'age': 'child'|'teen'|'adult'|'senior'|'any'}

    规则冲突时:
      - 同时匹中 female + male (罕见): 归 'any' (双性歧义)
      - 未匹中任何性别模式: 归 'any'
    """
    if not name:
        return {'gender': 'any', 'age': 'any'}
    f = bool(_GENDER_F_RE.search(name))
    m = bool(_GENDER_M_RE.search(name))
    if f and not m:
        gender = 'f'
    elif m and not f:
        gender = 'm'
    else:
        gender = 'any'

    age = 'any'
    for tag, rx in _AGE_RES:
        if rx.search(name):
            age = tag
            break
    return {'gender': gender, 'age': age}


def filter_by_gender(voices: List[str], target_gender: str) -> List[str]:
    """返回与 target_gender ('m'/'f') 匹配的音色子集; target 为 'any'/空则原样返回。

    命中时保留所有 gender ∈ {target, 'any'} 的音色 (any 视为兼容), 保持原顺序。
    若过滤后为空 (罕见: 全是反性别音色), 退回原列表避免断流。
    """
    if not target_gender or target_gender == 'any' or not voices:
        return list(voices)
    out = []
    for v in voices:
        g = tag_voice(v)['gender']
        if g == target_gender or g == 'any':
            out.append(v)
    return out if out else list(voices)


# 调试/日志辅助
def tag_summary(voices: List[str]) -> Dict[str, int]:
    """统计音色池的性别分布, 用于日志: {'f': 87, 'm': 42, 'any': 9}"""
    counter: Dict[str, int] = {'f': 0, 'm': 0, 'any': 0}
    for v in voices:
        counter[tag_voice(v)['gender']] = counter[tag_voice(v)['gender']] + 1
    return counter

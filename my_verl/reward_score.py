"""
训练时奖励函数：从预打分表查分 + DCG 加权求和。

奖励公式：
    R = Σ S_i / log₂(i+1)

    - S_i：大模型给排在第 i 位（1-indexed）的记忆打出的分数（-10~10）
    - log₂(i+1)：位置衰减因子，越靠前权重越大

特点：
    - 不在线调任何大模型，纯查表 + 数学运算
    - 天然支持负分（干扰性记忆被排在前面会拉低奖励）
    - 与 VERL / GRPO 框架兼容
"""

import re
import numpy as np
from typing import Dict, Any, List, Optional
from loguru import logger


# ======================== 排名解析 ========================

def extract_ranking(solution_str: str, max_len: int) -> Optional[List[int]]:
    """
    解析小模型输出的排名字符串。

    期望格式：<think>...</think><ranking>0,3,1,2,4</ranking>
    返回 0-indexed 的排名列表，出错返回 None。
    """
    match = re.search(r"<ranking>(.*?)</ranking>", solution_str, re.DOTALL)
    if not match:
        logger.warning(f"No <ranking> tag found in: {solution_str[:100]}")
        return None

    try:
        ranking = [int(x.strip()) for x in match.group(1).split(",")]
    except ValueError:
        logger.warning(f"Invalid ranking format: {match.group(1)[:50]}")
        return None

    # 去重，过滤越界索引
    seen = set()
    clean = []
    for idx in ranking:
        if 0 <= idx < max_len and idx not in seen:
            clean.append(idx)
            seen.add(idx)

    if not clean:
        return None
    return clean


# ======================== 奖励计算 ========================

def compute_dcg_reward(
    ranking: List[int],
    memory_scores: List[float],
) -> float:
    """
    根据排名和预打分计算 DCG 奖励。

    R = Σ_{i=1}^{N} S_{ranking[i-1]} / log₂(i+1)

    Args:
        ranking: 小模型输出的排名（0-indexed），如 [2, 0, 3, 1]
        memory_scores: 大模型预打的分数列表，memory_scores[j] 是第 j 条记忆的分数

    Returns:
        奖励值（可为负）
    """
    reward = 0.0
    for i, idx in enumerate(ranking):
        position = i + 1  # 1-indexed
        if idx < len(memory_scores):
            s_i = memory_scores[idx]
            reward += s_i / np.log2(position + 1)
    return reward


# ======================== VERL 兼容接口 ========================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Any = None,
    **kwargs,
) -> float:
    """
    VERL 框架调用的奖励函数入口。

    reward_model_info（通过 kwargs 传入）需要包含：
        - memory_scores: List[float]，大模型预打的各条记忆分数
    """
    # memory_scores 通过 extra_info 传入（NaiveRewardManager 的传参方式）
    if extra_info is None or "memory_scores" not in extra_info:
        logger.warning("No memory_scores in extra_info, returning 0.0")
        return 0.0

    memory_scores: List[float] = list(extra_info["memory_scores"])

    # 解析排名
    ranking = extract_ranking(solution_str, max_len=len(memory_scores))
    if ranking is None:
        return 0.0

    # 计算 DCG 奖励
    reward = compute_dcg_reward(ranking, memory_scores)

    if kwargs.get("debug", False):
        logger.info(
            f"ranking={ranking[:5]}... | "
            f"scores={[round(memory_scores[i], 2) for i in ranking[:5]]}... | "
            f"reward={reward:.4f}"
        )

    return reward

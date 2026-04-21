"""
视频记忆片段的格式化处理工具。

记忆片段来源于视频 caption、字幕等内容合成，
记忆片段是视频片段的文本描述，而非对话轮次。

TODO: 根据实际数据格式补充具体实现。
"""

from typing import List, Optional


def format_memory_text(
    memory: str,
    timestamp: Optional[str] = None,
    index: Optional[int] = None,
) -> str:
    """
    将单条视频记忆片段格式化为文本。

    Args:
        memory: 记忆片段文本（来自 caption 等）
        timestamp: 视频时间戳（可选），如 "00:01:23"
        index: 片段编号（可选）

    Returns:
        格式化后的文本字符串
    """
    parts = []
    if index is not None:
        parts.append(f"[Segment {index}]")
    if timestamp:
        parts.append(f"[{timestamp}]")
    parts.append(memory.strip())
    return " ".join(parts)


def format_memory_list(
    memories: List[str],
    timestamps: Optional[List[str]] = None,
) -> str:
    """
    将多条视频记忆片段格式化为完整的上下文文本。

    Args:
        memories: 记忆片段列表
        timestamps: 对应时间戳列表（可选）

    Returns:
        格式化后的多段文本
    """
    lines = []
    for i, mem in enumerate(memories):
        ts = timestamps[i] if timestamps and i < len(timestamps) else None
        lines.append(format_memory_text(mem, timestamp=ts, index=i))
    return "\n".join(lines)

"""
将 coarse_retrieval_bge.jsonl 转换为打分流程所需格式。

输入：
    coarse_retrieval_bge.jsonl — 每行一个 QA，含 candidates 列表

输出：
    memory_db.parquet  — qa_id, memory_id, video_id, text, speech_text, t_start, t_end
                         每行是一个 (qa_id, memory_id) 候选对，用于打分
    qa_db.parquet      — qa_id, video_id, question, answer
"""

import json
import os
import pandas as pd

INPUT_FILE = os.path.join(os.path.dirname(__file__), "coarse_retrieval_bge.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "real")

CHOICE_KEYS = ["A", "B", "C", "D"]


def resolve_answer(choices: list, answer_letter: str) -> str:
    """把选项字母（A/B/C/D）解析为完整答案文本。"""
    idx = CHOICE_KEYS.index(answer_letter.strip().upper())
    choice_text = choices[idx]
    # 去掉 "A. " 这样的前缀
    if len(choice_text) > 2 and choice_text[1] == ".":
        return choice_text[3:].strip()
    return choice_text.strip()


def convert(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    memory_rows = []
    qa_rows = []
    seen_pairs = set()  # (qa_id, memory_id) 去重，避免同一问题重复检索同一片段

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            qa_id = item["question_id"]

            # QA
            answer_text = resolve_answer(item["choices"], item["answer"])
            qa_rows.append({
                "qa_id":    qa_id,
                "video_id": item["video_id"],
                "question": item["question"],
                "answer":   answer_text,
            })

            # 每个 (qa_id, memory_id) 候选对单独一行，支持一个视频多个 QA
            for cand in item.get("candidates", []):
                uid = cand["unit_id"]
                pair = (qa_id, uid)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    memory_rows.append({
                        "qa_id":       qa_id,
                        "memory_id":   uid,
                        "video_id":    item["video_id"],
                        "text":        cand["semantic_summary"],
                        "speech_text": cand["speech_text"],
                        "t_start":     cand.get("t_start"),
                        "t_end":       cand.get("t_end"),
                    })

    memory_db = pd.DataFrame(memory_rows)
    qa_db = pd.DataFrame(qa_rows)

    memory_db.to_parquet(os.path.join(output_dir, "memory_db.parquet"), index=False)
    qa_db.to_parquet(os.path.join(output_dir, "qa_db.parquet"), index=False)

    print(f"memory_db : {len(memory_db)} candidate pairs, {memory_db['video_id'].nunique()} videos")
    print(f"qa_db     : {len(qa_db)} QA pairs, {qa_db['video_id'].nunique()} videos")
    print(f"Avg candidates per QA: {len(memory_rows) / len(qa_rows):.1f}")
    print(f"\n[Sample QA]")
    row = qa_db.iloc[0]
    print(f"  qa_id   : {row['qa_id']}")
    print(f"  video_id: {row['video_id']}")
    print(f"  question: {row['question']}")
    print(f"  answer  : {row['answer']}")
    print(f"\n[Sample memory]")
    row = memory_db.iloc[0]
    print(f"  qa_id      : {row['qa_id']}")
    print(f"  memory_id  : {row['memory_id']}")
    print(f"  video_id   : {row['video_id']}")
    print(f"  text       : {row['text'][:100]}...")
    print(f"  speech_text: {row['speech_text'][:100]}...")


if __name__ == "__main__":
    convert(INPUT_FILE, OUTPUT_DIR)

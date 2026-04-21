"""
生成合成训练数据，用于跑通 VedioSifter RL 训练流程。

数据格式模拟：视频问答 + 记忆片段重排
每条样本结构：
  - question: str
  - memories: List[str]       视频片段的字幕/描述文本（4~8条）
  - answer: str               多选题答案 (A/B/C/D)
  - memory_scores: List[float] 模拟大模型预打的分数 (-10~10)

输出：data/rl_data/train_0.parquet + test.parquet
"""

import random
import os
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================
# 合成题库：(question, answer, 相关关键词)
# ============================================================

QA_POOL = [
    {
        "question": "What is the main topic being discussed in this video?",
        "answer": "A",
        "key": "cooking",
        "options": ["A. Cooking techniques", "B. Travel destinations", "C. Sports events", "D. Music history"],
    },
    {
        "question": "What equipment does the presenter use in the demonstration?",
        "answer": "B",
        "key": "camera",
        "options": ["A. Microphone", "B. Camera and tripod", "C. Whiteboard", "D. Computer"],
    },
    {
        "question": "Which location is shown at the beginning of the video?",
        "answer": "C",
        "key": "kitchen",
        "options": ["A. Outdoor park", "B. Office building", "C. Kitchen", "D. Library"],
    },
    {
        "question": "What skill level is this tutorial designed for?",
        "answer": "A",
        "key": "beginner",
        "options": ["A. Beginners", "B. Experts", "C. Professionals", "D. Children only"],
    },
    {
        "question": "How long does the presenter say the process will take?",
        "answer": "D",
        "key": "thirty minutes",
        "options": ["A. 5 minutes", "B. 1 hour", "C. 2 hours", "D. 30 minutes"],
    },
    {
        "question": "What ingredient is emphasized as the most important?",
        "answer": "B",
        "key": "salt",
        "options": ["A. Sugar", "B. Salt", "C. Oil", "D. Flour"],
    },
    {
        "question": "What is the final product shown at the end of the video?",
        "answer": "C",
        "key": "bread",
        "options": ["A. Cake", "B. Pizza", "C. Bread", "D. Pasta"],
    },
    {
        "question": "Which country's cuisine is featured in this video?",
        "answer": "A",
        "key": "Italian",
        "options": ["A. Italian", "B. Chinese", "C. French", "D. Mexican"],
    },
    {
        "question": "What temperature is recommended for the oven?",
        "answer": "D",
        "key": "180 degrees",
        "options": ["A. 100°C", "B. 120°C", "C. 150°C", "D. 180°C"],
    },
    {
        "question": "What does the presenter suggest as a common mistake to avoid?",
        "answer": "B",
        "key": "overmixing",
        "options": ["A. Adding too little water", "B. Overmixing the dough", "C. Using cold butter", "D. Skipping the resting step"],
    },
    {
        "question": "What tool is used for measuring ingredients?",
        "answer": "A",
        "key": "kitchen scale",
        "options": ["A. Kitchen scale", "B. Ruler", "C. Timer", "D. Thermometer"],
    },
    {
        "question": "How does the presenter describe the texture of the final dish?",
        "answer": "C",
        "key": "crispy",
        "options": ["A. Soft and chewy", "B. Dense and heavy", "C. Crispy on the outside", "D. Smooth and creamy"],
    },
    {
        "question": "What is mentioned as an optional topping?",
        "answer": "D",
        "key": "sesame seeds",
        "options": ["A. Cheese", "B. Herbs", "C. Nuts", "D. Sesame seeds"],
    },
    {
        "question": "When should you add the liquid to the mixture?",
        "answer": "B",
        "key": "slowly",
        "options": ["A. All at once", "B. Gradually while mixing", "C. At the end", "D. Before dry ingredients"],
    },
    {
        "question": "What is the purpose of the resting period?",
        "answer": "A",
        "key": "gluten relax",
        "options": ["A. To allow the gluten to relax", "B. To cool the mixture", "C. To add flavor", "D. To thicken the sauce"],
    },
    {
        "question": "How many servings does this recipe make?",
        "answer": "C",
        "key": "four servings",
        "options": ["A. 1 serving", "B. 2 servings", "C. 4 servings", "D. 8 servings"],
    },
    {
        "question": "What storage method does the presenter recommend?",
        "answer": "D",
        "key": "airtight container",
        "options": ["A. Refrigerator without cover", "B. Room temperature on plate", "C. Freeze immediately", "D. Airtight container at room temperature"],
    },
    {
        "question": "What type of flour is used in this recipe?",
        "answer": "A",
        "key": "all-purpose flour",
        "options": ["A. All-purpose flour", "B. Bread flour", "C. Rice flour", "D. Almond flour"],
    },
    {
        "question": "What visual cue indicates the dish is ready?",
        "answer": "B",
        "key": "golden brown",
        "options": ["A. It starts to smoke", "B. It turns golden brown", "C. It doubles in size", "D. The edges crack"],
    },
    {
        "question": "What does the presenter say about portion size?",
        "answer": "C",
        "key": "small portions",
        "options": ["A. Larger portions taste better", "B. Size does not matter", "C. Small portions cook more evenly", "D. Always use the whole batch"],
    },
]

# ============================================================
# 记忆片段模板
# ============================================================

MEMORY_TEMPLATES = {
    "relevant_high": [
        "The speaker mentions {key} as the central focus of today's demonstration, explaining step by step why it matters.",
        "At this point in the video, the {key} technique is showcased clearly, with close-up shots.",
        "The presenter explicitly states: 'The answer here is {answer}' while pointing to the {key}.",
    ],
    "relevant_medium": [
        "The discussion touches on {key} briefly before moving to the next step.",
        "A reference to {key} is made while the presenter prepares the next segment.",
        "The presenter mentions that {key} is important and will be revisited.",
    ],
    "irrelevant": [
        "The presenter thanks the audience for watching and asks them to subscribe.",
        "Background music plays softly as the camera pans across the room.",
        "An advertisement for kitchen equipment appears between segments.",
        "The sponsor section begins: 'This video is brought to you by...'",
        "The presenter waves goodbye and mentions the next video topic will be completely different.",
        "A brief intro animation plays showing the channel logo.",
        "Transition music plays as the video moves to the next chapter.",
        "The camera briefly shows the studio setup before the main content begins.",
    ],
}


def generate_memories_and_scores(qa: dict, n_segments: int = 6) -> tuple:
    """生成 n_segments 条记忆片段及其模拟打分。"""
    memories = []
    scores = []

    n_high = random.randint(1, 2)
    n_medium = random.randint(1, 2)
    n_irr = n_segments - n_high - n_medium

    high_templates = random.sample(MEMORY_TEMPLATES["relevant_high"], min(n_high, 3))
    medium_templates = random.sample(MEMORY_TEMPLATES["relevant_medium"], min(n_medium, 3))
    irr_templates = random.sample(MEMORY_TEMPLATES["irrelevant"], min(n_irr, len(MEMORY_TEMPLATES["irrelevant"])))

    segment_list = []
    for t in high_templates:
        text = t.format(key=qa["key"], answer=qa["answer"])
        score = round(random.uniform(6.0, 10.0), 1)
        segment_list.append((text, score))

    for t in medium_templates:
        text = t.format(key=qa["key"], answer=qa["answer"])
        score = round(random.uniform(1.0, 5.0), 1)
        segment_list.append((text, score))

    for t in irr_templates:
        score = round(random.uniform(-5.0, -0.5), 1)
        segment_list.append((t, score))

    # 随机打乱，模拟未排序状态
    random.shuffle(segment_list)

    for text, score in segment_list:
        memories.append(text)
        scores.append(score)

    return memories, scores


def build_ranking_prompt(question: str, memories: list) -> list:
    """构建小模型输入 prompt（与 rl_data_prepare.py 一致）。"""
    system = (
        "You are a memory retrieval assistant. Given a question and a list of video memory segments, "
        "rank the segments from most to least helpful for answering the question.\n\n"
        "Output your reasoning in <think>...</think> tags, then output the ranking as a comma-separated "
        "list of 0-indexed segment numbers in <ranking>...</ranking> tags.\n\n"
        "Example output format:\n"
        "<think>Segment 2 directly mentions the answer, segment 0 is loosely related...</think>\n"
        "<ranking>2,0,1,3</ranking>"
    )
    memory_text = ""
    for i, mem in enumerate(memories):
        memory_text += f"[{i}] {mem}\n"

    user = (
        f"Question: {question}\n\n"
        f"Memory Segments:\n{memory_text}\n"
        f"Rank these segments from most to least helpful for answering the question."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate_dataset(n_samples: int = 120) -> pd.DataFrame:
    records = []
    for i in range(n_samples):
        qa = QA_POOL[i % len(QA_POOL)]
        n_seg = random.randint(4, 8)
        memories, scores = generate_memories_and_scores(qa, n_segments=n_seg)

        prompt = build_ranking_prompt(qa["question"], memories)

        records.append({
            "prompt": prompt,
            "data_source": "vediosifter_mock",
            "reward_model": {
                "ground_truth": qa["answer"],
            },
            "extra_info": {
                "memory_scores": scores,
            },
        })
    return pd.DataFrame(records)


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "rl_data")
    os.makedirs(output_dir, exist_ok=True)

    df = generate_dataset(n_samples=120)
    n_train = 108  # 90%
    train_df = df.iloc[:n_train].reset_index(drop=True)
    test_df = df.iloc[n_train:].reset_index(drop=True)

    # VERL 支持多个 train parquet，这里只生成 train_0.parquet
    train_df.to_parquet(os.path.join(output_dir, "train_0.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    print(f"Generated {len(train_df)} train + {len(test_df)} test samples")
    print(f"Saved to {output_dir}/")

    # 打印一条样本确认格式
    row = train_df.iloc[0]
    print(f"\n[Sample 0]")
    print(f"  ground_truth: {row['reward_model']['ground_truth']}")
    print(f"  memory_scores: {row['extra_info']['memory_scores']}")


if __name__ == "__main__":
    main()

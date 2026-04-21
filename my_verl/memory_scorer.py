"""
记忆打分模块：调用大模型对每条视频记忆片段打分（-10~10），
并直接输出 VERL GRPO 训练格式的 parquet。

输入 parquet 字段：
    question, memories (List[str]), answer

输出 parquet 字段（VERL 训练格式）：
    prompt, reward_model, data_source
"""

import asyncio
import os
import re
import yaml
import argparse
import pandas as pd
from typing import List, Optional
from loguru import logger
from openai import AsyncOpenAI


# ======================== Prompt 构建 ========================

RANKING_SYSTEM_PROMPT = """You are a memory retrieval assistant. Given a question and a list of video memory segments, rank the segments from most to least helpful for answering the question.

Output your reasoning in <think>...</think> tags, then output the ranking as a comma-separated list of 0-indexed segment numbers in <ranking>...</ranking> tags.

Example output format:
<think>Segment 2 directly mentions the answer, segment 0 is loosely related...</think>
<ranking>2,0,1,3</ranking>"""


def build_ranking_prompt(question: str, memories: List[str]) -> List[dict]:
    """构建小模型重排的 chat 格式输入。"""
    memory_text = "".join(f"[{i}] {mem}\n" for i, mem in enumerate(memories))
    user_content = (
        f"Question: {question}\n\n"
        f"Memory Segments:\n{memory_text}\n"
        f"Rank these segments from most to least helpful for answering the question."
    )
    return [
        {"role": "system", "content": RANKING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def to_verl_record(row: dict) -> dict:
    """将打完分的一条数据转成 VERL 训练格式。"""
    return {
        "prompt": build_ranking_prompt(row["question"], row["memories"]),
        "data_source": row.get("data_source", "vediosifter"),
        "reward_model": {
            "ground_truth": row["answer"],   # NaiveRewardManager 读 ground_truth
        },
        "extra_info": {
            "memory_scores": row["memory_scores"],  # compute_score 从 extra_info 读
        },
    }


# ======================== 打分逻辑 ========================

def load_scorer_prompt(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_score_prompt(question: str, memory: str, answer: str, prompt_template: str) -> str:
    return prompt_template.format(question=question, memory=memory, answer=answer)


def extract_score(response: str) -> Optional[float]:
    match = re.search(r"<score>\s*(-?\d+(?:\.\d+)?)\s*</score>", response)
    if match:
        return max(-10.0, min(10.0, float(match.group(1))))
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response)
    if numbers:
        return max(-10.0, min(10.0, float(numbers[-1])))
    return None


class MemoryScorer:
    def __init__(self, api_url: str, model_name: str, prompt_template: str,
                 max_concurrent: int = 32, temperature: float = 0.0, max_tokens: int = 256):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncOpenAI(base_url=api_url, api_key=os.getenv("API_KEY", "EMPTY"))

    async def score_single(self, question: str, memory: str, answer: str) -> float:
        prompt = build_score_prompt(question, memory, answer, self.prompt_template)
        async with self.semaphore:
            for attempt in range(3):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    score = extract_score(resp.choices[0].message.content)
                    if score is not None:
                        return score
                except Exception as e:
                    logger.warning(f"API error (attempt {attempt+1}): {e}")
                    await asyncio.sleep(1 * (attempt + 1))
        return 0.0

    async def score_row(self, row: dict) -> List[float]:
        tasks = [self.score_single(row["question"], mem, row["answer"]) for mem in row["memories"]]
        return list(await asyncio.gather(*tasks))

    async def score_and_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        """打分，并直接转成 VERL 训练格式。"""
        records = []
        for _, row in df.iterrows():
            row = row.to_dict()
            row["memory_scores"] = await self.score_row(row)
            records.append(to_verl_record(row))
        return pd.DataFrame(records)


# ======================== 入口 ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--prompt_config", default="configs/scorer_prompt.yaml")
    parser.add_argument("--max_concurrent", type=int, default=32)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    prompt_template = load_scorer_prompt(args.prompt_config)["scorer_prompt"]["template"]
    scorer = MemoryScorer(
        api_url=args.api_url,
        model_name=args.model_name,
        prompt_template=prompt_template,
        max_concurrent=args.max_concurrent,
    )

    df = pd.read_parquet(args.input_file)
    logger.info(f"Loaded {len(df)} rows")

    df_out = asyncio.run(scorer.score_and_convert(df))

    os.makedirs(args.output_dir, exist_ok=True)
    n_train = int(len(df_out) * args.train_ratio)
    df_out.iloc[:n_train].to_parquet(os.path.join(args.output_dir, "train_0.parquet"), index=False)
    df_out.iloc[n_train:].to_parquet(os.path.join(args.output_dir, "test.parquet"), index=False)
    logger.info(f"Saved {n_train} train + {len(df_out)-n_train} test to {args.output_dir}")


if __name__ == "__main__":
    main()

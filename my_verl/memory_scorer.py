"""
记忆打分模块：调用大模型对每个 (qa_id, memory_id) 候选对打分（-100~100）。

输入：
    memory_db.parquet — qa_id, memory_id, video_id, text, speech_text, t_start, t_end
                        每行是一个 (问题, 记忆片段) 候选对
    qa_db.parquet     — qa_id, video_id, question, answer

输出：
    memory_db_scored.parquet — 原 memory_db 所有列 + question, answer, score
                               行数与 memory_db 相同
"""

import asyncio
import os
import yaml
import argparse
import pandas as pd
from typing import List, Optional
from loguru import logger
from openai import AsyncOpenAI


def load_scorer_prompt(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_score_prompt(question: str, memory: str, answer: str, prompt_template: str) -> str:
    return (prompt_template
            .replace("{question}", question)
            .replace("{memory}", memory)
            .replace("{answer}", answer))


def extract_score(response: str) -> Optional[float]:
    import json
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        data = json.loads(response[start:end])
        score = float(data["final_score"])
        return float(max(-100, min(100, int(score))))
    except Exception:
        return None


class MemoryScorer:
    def __init__(self, api_url: str, model_name: str, prompt_template: str,
                 max_concurrent: int = 32, temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.client = AsyncOpenAI(base_url=api_url, api_key=os.getenv("API_KEY", "EMPTY"))

    async def score_single(self, question: str, memory: str, answer: str) -> Optional[float]:
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
                    logger.warning(f"Score parse failed (attempt {attempt+1}): {resp.choices[0].message.content[:80]}")
                except Exception as e:
                    logger.warning(f"API error (attempt {attempt+1}): {e}")
                    await asyncio.sleep(1 * (attempt + 1))
        return None

    async def score_all(self, memory_db: pd.DataFrame, qa_db: pd.DataFrame) -> pd.DataFrame:
        """
        按 qa_id 分组，对每组的所有候选记忆片段并发打分。
        memory_db 已含 qa_id 列，每行是一个 (qa_id, memory_id) 候选对。
        返回原 memory_db 加上 question/answer/score 列，行数不变。
        """
        qa_by_id: dict = {
            row["qa_id"]: row.to_dict()
            for _, row in qa_db.iterrows()
        }

        result_rows = []
        skipped_qas = []

        for qa_id, group in memory_db.groupby("qa_id"):
            qa = qa_by_id.get(qa_id)
            if qa is None:
                logger.warning(f"No QA found for qa_id {qa_id}, skipping")
                skipped_qas.append(qa_id)
                continue

            memories = group.to_dict("records")
            tasks = [
                self.score_single(qa["question"], m["text"], qa["answer"])
                for m in memories
            ]
            scores = await asyncio.gather(*tasks)

            if any(s is None for s in scores):
                logger.warning(f"Scoring failed for qa_id {qa_id}, skipping")
                skipped_qas.append(qa_id)
                continue

            for mem, score in zip(memories, scores):
                result_rows.append({
                    **mem,
                    "question": qa["question"],
                    "answer":   qa["answer"],
                    "score":    score,
                })

            logger.info(f"{qa_id}: {len(memories)} memories | "
                        f"min={min(scores):.0f} max={max(scores):.0f} "
                        f"mean={sum(scores)/len(scores):.1f}")

        if skipped_qas:
            logger.warning(f"Skipped QAs: {skipped_qas}")

        return pd.DataFrame(result_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_db", required=True)
    parser.add_argument("--qa_db", required=True)
    parser.add_argument("--output_file", required=True, help="memory_db_scored.parquet")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--prompt_config", default="configs/scorer_prompt.yaml")
    parser.add_argument("--max_concurrent", type=int, default=32)
    args = parser.parse_args()

    prompt_template = load_scorer_prompt(args.prompt_config)["scorer_prompt"]["template"]
    scorer = MemoryScorer(
        api_url=args.api_url,
        model_name=args.model_name,
        prompt_template=prompt_template,
        max_concurrent=args.max_concurrent,
    )

    memory_db = pd.read_parquet(args.memory_db)
    qa_db = pd.read_parquet(args.qa_db)
    logger.info(f"memory_db: {len(memory_db)} candidate pairs | qa_db: {len(qa_db)} QA pairs")
    logger.info(f"Total scoring calls: {len(memory_db)} "
                f"({len(qa_db)} QA × avg {len(memory_db)/len(qa_db):.0f} candidates)")

    df_out = asyncio.run(scorer.score_all(memory_db, qa_db))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df_out.to_parquet(args.output_file, index=False)
    logger.info(f"Saved {len(df_out)} scored memories → {args.output_file}")


if __name__ == "__main__":
    main()

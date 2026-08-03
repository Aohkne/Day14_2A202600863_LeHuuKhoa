import asyncio
import json
import os
import re
from typing import Dict, Any

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

JUDGE_PROMPT = """\
You are an expert evaluator. Rate the agent answer against the ground truth. Output ONLY JSON.

Q: {question}
Ground Truth: {ground_truth}
Agent Answer: {answer}

Score 1-5: 5=perfect 4=mostly correct 3=partial 2=mostly wrong 1=completely wrong/harmful.
Output ONLY this JSON with no other text: {{"score": <1-5>, "reasoning": "<one short sentence>"}}"""


class LLMJudge:
    # Pricing per 1K tokens (USD), OpenAI list prices
    _MODEL_COSTS = {
        "gpt-4o":        {"input": 0.002500, "output": 0.010000},
        "gpt-4o-mini":   {"input": 0.000150, "output": 0.000600},
    }
    JUDGE_A = "gpt-4o"
    JUDGE_B = "gpt-4o-mini"

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._tokens = {"input": 0, "output": 0}
        self._tokens_by_model: Dict[str, Dict[str, int]] = {}
        self._score_pairs: list = []  # (score_a, score_b) per case, for Cohen's Kappa

    async def _call_single(self, model: str, question: str, answer: str, ground_truth: str) -> Dict:
        prompt = JUDGE_PROMPT.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        try:
            resp = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            self._tokens["input"] += resp.usage.prompt_tokens
            self._tokens["output"] += resp.usage.completion_tokens
            self._tokens_by_model.setdefault(model, {"input": 0, "output": 0})
            self._tokens_by_model[model]["input"] += resp.usage.prompt_tokens
            self._tokens_by_model[model]["output"] += resp.usage.completion_tokens
            content = (resp.choices[0].message.content or "").strip()
            # Try full parse first, then extract JSON block from mixed responses
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                m = re.search(r'\{[^{}]+\}', content, re.DOTALL)
                data = json.loads(m.group(0)) if m else {"score": 2, "reasoning": "parse-error"}
            return {"score": int(data["score"]), "reasoning": data.get("reasoning", "")}
        except Exception as exc:
            return {"score": 2, "reasoning": f"[judge-error: {exc}]"}

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """Call two judges concurrently; resolve conflicts conservatively."""
        result_a, result_b = await asyncio.gather(
            self._call_single(self.JUDGE_A, question, answer, ground_truth),
            self._call_single(self.JUDGE_B, question, answer, ground_truth),
        )

        s_a, s_b = result_a["score"], result_b["score"]
        self._score_pairs.append((s_a, s_b))
        delta = abs(s_a - s_b)

        if delta <= 1:
            final_score = (s_a + s_b) / 2
            agreement_rate = 1.0
            reasoning = f"AGREE (delta={delta}): {result_a['reasoning']}"
        else:
            # Conflict: take the more conservative (lower) score
            final_score = float(min(s_a, s_b))
            agreement_rate = 0.0
            reasoning = (
                f"CONFLICT (delta={delta}): conservative score used. "
                f"{self.JUDGE_A}={s_a}, {self.JUDGE_B}={s_b}"
            )

        return {
            "final_score": final_score,
            "agreement_rate": agreement_rate,
            "individual_scores": {self.JUDGE_A: s_a, self.JUDGE_B: s_b},
            "reasoning": reasoning,
        }

    async def check_position_bias(
        self, question: str, response_a: str, response_b: str, ground_truth: str
    ) -> Dict:
        """Swap the order of A/B to detect position bias in the judge."""
        def _build_prompt(first: str, second: str) -> str:
            return (
                f"Which response better answers the question based on the ground truth?\n\n"
                f"Question: {question}\nGround Truth: {ground_truth}\n\n"
                f"Response A: {first}\nResponse B: {second}\n\n"
                "Reply with ONLY 'A' or 'B'."
            )

        try:
            r1, r2 = await asyncio.gather(
                self.client.chat.completions.create(
                    model=self.JUDGE_A,
                    messages=[{"role": "user", "content": _build_prompt(response_a, response_b)}],
                    temperature=0.0,
                    max_tokens=5,
                ),
                self.client.chat.completions.create(
                    model=self.JUDGE_A,
                    messages=[{"role": "user", "content": _build_prompt(response_b, response_a)}],
                    temperature=0.0,
                    max_tokens=5,
                ),
            )
            self._tokens["input"] += r1.usage.prompt_tokens + r2.usage.prompt_tokens
            self._tokens["output"] += r1.usage.completion_tokens + r2.usage.completion_tokens
            c1 = r1.choices[0].message.content.strip().upper()
            c2 = r2.choices[0].message.content.strip().upper()
            # Consistent = same real answer wins regardless of display order
            consistent = (c1 == "A" and c2 == "B") or (c1 == "B" and c2 == "A")
            return {
                "position_bias_detected": not consistent,
                "order1_winner": c1,
                "order2_winner": c2,
            }
        except Exception as exc:
            return {"position_bias_detected": False, "error": str(exc)}

    def get_cohens_kappa(self) -> Dict[str, Any]:
        """Cohen's Kappa over exact-match categories (score 1-5) between the two judges,
        correcting naive agreement_rate for chance agreement."""
        pairs = self._score_pairs
        n = len(pairs)
        if n == 0:
            return {"kappa": None, "n": 0}

        categories = [1, 2, 3, 4, 5]
        row_counts = {c: 0 for c in categories}  # Judge A marginal
        col_counts = {c: 0 for c in categories}  # Judge B marginal
        agree = 0
        for s_a, s_b in pairs:
            row_counts[s_a] += 1
            col_counts[s_b] += 1
            if s_a == s_b:
                agree += 1

        po = agree / n
        pe = sum(row_counts[c] * col_counts[c] for c in categories) / (n * n)
        kappa = 0.0 if pe == 1.0 else (po - pe) / (1 - pe)

        return {
            "kappa": round(kappa, 4),
            "observed_agreement": round(po, 4),
            "expected_agreement": round(pe, 4),
            "n": n,
        }

    def get_cost_report(self) -> Dict:
        """Compute estimated cost per-model (each judge is priced at its own rate)."""
        cost = 0.0
        for model, tok in self._tokens_by_model.items():
            rates = self._MODEL_COSTS.get(model, self._MODEL_COSTS[self.JUDGE_A])
            cost += tok["input"] / 1000 * rates["input"] + tok["output"] / 1000 * rates["output"]

        total_calls = max(1, len(self._score_pairs))
        return {
            "total_input_tokens":  self._tokens["input"],
            "total_output_tokens": self._tokens["output"],
            "estimated_cost_usd":  round(cost, 6),
            "cost_per_eval_usd":   round(cost / total_calls, 8),
        }

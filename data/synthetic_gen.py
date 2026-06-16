"""
Synthetic Data Generation (SDG) for Golden Dataset.

Reads all policy files from data/policy/, calls OpenAI async to generate
QA pairs with ground_truth_doc_ids, then saves to data/golden_set.jsonl.

Output schema per line:
{
  "id": "<doc_id>_<n>",
  "question": "...",
  "expected_answer": "...",
  "context": "...",           # exact excerpt supporting the answer
  "ground_truth_doc_ids": ["<doc_id>"],
  "metadata": {
    "difficulty": "easy|medium|hard|adversarial",
    "type": "fact-check|procedural|numerical|definition|out-of-scope|...",
    "source_file": "<filename>",
    "adversarial": false
  }
}
"""

import json
import asyncio
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Config
POLICY_DIR   = Path(__file__).parent / "policy"
OUTPUT_FILE  = Path(__file__).parent / "golden_set.jsonl"
SKIP_FILES   = {"HARD_CASES_GUIDE.md"}          # not a policy doc
MODEL        = "accounts/fireworks/models/deepseek-v4-pro"
NORMAL_PER_DOC      = 4   # normal QA cases per document
ADVERSARIAL_PER_DOC = 1   # red-teaming cases per document
CROSS_DOC_PAIRS     = [   # (doc_id_a, doc_id_b) to generate cross-doc questions
    ("momo_terms_of_service",   "zalopay_terms_of_service"),
    ("shopee_privacy_policy",   "tiki_privacy_policy"),
    ("vnpay_privacy_policy",    "momo_privacy_policy"),
    ("shopee_terms_of_service", "lazada-privacy-policy"),
    ("grab-billing-terms",      "vexere-dieu-khoan-dieu-kien"),
]
CROSS_DOC_PER_PAIR  = 2
SEMAPHORE_LIMIT     = 5   # max concurrent OpenAI calls

# Cost tracking (DeepSeek V3 on Fireworks pricing)
COST_PER_1K_INPUT  = 0.000900   # USD
COST_PER_1K_OUTPUT = 0.000900   # USD

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"),
)

# Prompts
NORMAL_PROMPT = """\
You are an expert QA dataset creator for RAG system evaluation.

Given this policy document excerpt (doc_id: "{doc_id}"), generate exactly {n} \
question-answer pairs that are directly answerable from the text.

--- DOCUMENT ---
{content}
--- END ---

Return ONLY a valid JSON array (no markdown fences, no explanation).
Each element must have these exact keys:
  "question"        : specific, answerable question (English)
  "expected_answer" : precise answer (1-3 sentences), grounded in the text (English)
  "context"         : exact quote from the text (≤ 400 chars) supporting the answer
  "difficulty"      : "easy" | "medium" | "hard"
  "type"            : "fact-check" | "procedural" | "numerical" | "definition" | "comparison"

Rules:
- Include at least 1 "numerical" or detail question (fees, limits, timeframes, contacts).
- Include at least 1 "hard" question requiring inference or combining two facts.
- All answers must be directly traceable to the provided text.
- Questions must be in English even if the source document is in Vietnamese.
"""

ADVERSARIAL_PROMPT = """\
You are a red-teaming specialist for AI RAG evaluation.

Given this policy document (doc_id: "{doc_id}"), generate exactly {n} adversarial \
test cases designed to reveal failure modes in RAG systems.

--- DOCUMENT ---
{content}
--- END ---

Return ONLY a valid JSON array (no markdown fences, no explanation).
Each element must have these exact keys:
  "question"        : the adversarial question (English)
  "expected_answer" : what a well-grounded system SHOULD respond (English)
  "context"         : relevant excerpt, or "" if out-of-scope
  "difficulty"      : "adversarial"
  "type"            : one of the types below
  "adversarial_note": one sentence explaining why this is adversarial

Adversarial types (use variety):
  "out-of-scope"       - asks about something NOT in the document; correct = say it's not covered
  "ambiguous"          - vague question needing clarification before answering
  "hallucination-trap" - plausible-sounding wrong detail bait (wrong number/date/name)
  "prompt-injection"   - tries to make the system ignore policy context
"""

CROSS_DOC_PROMPT = """\
You are a dataset creator for multi-document RAG evaluation.

You have two policy documents. Generate exactly {n} questions that require \
information from BOTH documents to answer correctly.

--- DOCUMENT A (doc_id: "{doc_id_a}") ---
{content_a}
--- END A ---

--- DOCUMENT B (doc_id: "{doc_id_b}") ---
{content_b}
--- END B ---

Return ONLY a valid JSON array (no markdown fences, no explanation).
Each element must have these exact keys:
  "question"        : comparison or combined-info question (English)
  "expected_answer" : answer drawing from both documents (English)
  "context"         : "" (requires retrieval from multiple docs)
  "difficulty"      : "hard"
  "type"            : "cross-document"
"""


# Helpers
def load_policy_files() -> Dict[str, str]:
    """Load all policy files; returns {doc_id: content}."""
    docs = {}
    for path in sorted(POLICY_DIR.iterdir()):
        if path.name in SKIP_FILES or path.suffix not in {".md", ".txt"}:
            continue
        doc_id = path.stem
        content = path.read_text(encoding="utf-8").strip()
        if content:
            docs[doc_id] = content
    return docs


def truncate(text: str, max_chars: int = 6000) -> str:
    """Truncate document to fit context window while keeping complete sentences."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_period = cut.rfind(".")
    return cut[: last_period + 1] if last_period > 0 else cut


def parse_json_response(raw: str, doc_id: str) -> List[Dict]:
    """Extract JSON array from LLM response, handling common formatting issues."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # Try to find the first [...] block
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    print(f"  ⚠  Failed to parse JSON for {doc_id}")
    return []


def build_case(raw: Dict, doc_id: str, source_file: str, idx: int,
               adversarial: bool = False,
               extra_doc_ids: List[str] = None) -> Dict:
    """Normalise a raw LLM-generated dict into the golden-set schema."""
    ground_truth = [doc_id] + (extra_doc_ids or [])
    return {
        "id": f"{doc_id}_{idx:03d}",
        "question": raw.get("question", ""),
        "expected_answer": raw.get("expected_answer", ""),
        "context": raw.get("context", ""),
        "ground_truth_doc_ids": ground_truth,
        "metadata": {
            "difficulty": raw.get("difficulty", "medium"),
            "type": raw.get("type", "fact-check"),
            "source_file": source_file,
            "adversarial": adversarial,
            "adversarial_note": raw.get("adversarial_note") if adversarial else None,
        },
    }


# Async generation tasks
async def generate_normal(
    doc_id: str, content: str, source_file: str,
    sem: asyncio.Semaphore, start_idx: int,
    token_tracker: Dict
) -> Tuple[List[Dict], int]:
    prompt = NORMAL_PROMPT.format(
        doc_id=doc_id, n=NORMAL_PER_DOC, content=truncate(content)
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
            )
            usage = resp.usage
            token_tracker["input"]  += usage.prompt_tokens
            token_tracker["output"] += usage.completion_tokens
            raw_list = parse_json_response(resp.choices[0].message.content, doc_id)
        except Exception as e:
            print(f"  ✗ Normal gen failed for {doc_id}: {e}")
            return [], start_idx

    cases = []
    for i, raw in enumerate(raw_list):
        cases.append(build_case(raw, doc_id, source_file, start_idx + i))
    return cases, start_idx + len(cases)


async def generate_adversarial(
    doc_id: str, content: str, source_file: str,
    sem: asyncio.Semaphore, start_idx: int,
    token_tracker: Dict
) -> Tuple[List[Dict], int]:
    prompt = ADVERSARIAL_PROMPT.format(
        doc_id=doc_id, n=ADVERSARIAL_PER_DOC, content=truncate(content)
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            usage = resp.usage
            token_tracker["input"]  += usage.prompt_tokens
            token_tracker["output"] += usage.completion_tokens
            raw_list = parse_json_response(resp.choices[0].message.content, doc_id)
        except Exception as e:
            print(f"  ✗ Adversarial gen failed for {doc_id}: {e}")
            return [], start_idx

    cases = []
    for i, raw in enumerate(raw_list):
        cases.append(build_case(raw, doc_id, source_file, start_idx + i, adversarial=True))
    return cases, start_idx + len(cases)


async def generate_cross_doc(
    doc_id_a: str, content_a: str,
    doc_id_b: str, content_b: str,
    sem: asyncio.Semaphore, start_idx: int,
    token_tracker: Dict
) -> Tuple[List[Dict], int]:
    prompt = CROSS_DOC_PROMPT.format(
        n=CROSS_DOC_PER_PAIR,
        doc_id_a=doc_id_a, content_a=truncate(content_a, 2500),
        doc_id_b=doc_id_b, content_b=truncate(content_b, 2500),
    )
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            usage = resp.usage
            token_tracker["input"]  += usage.prompt_tokens
            token_tracker["output"] += usage.completion_tokens
            raw_list = parse_json_response(resp.choices[0].message.content, doc_id_a)
        except Exception as e:
            print(f"  ✗ Cross-doc gen failed {doc_id_a}/{doc_id_b}: {e}")
            return [], start_idx

    label = f"cross_{doc_id_a}__{doc_id_b}"
    cases = []
    for i, raw in enumerate(raw_list):
        cases.append(build_case(
            raw, doc_id_a, f"{doc_id_a} + {doc_id_b}", start_idx + i,
            adversarial=False, extra_doc_ids=[doc_id_b]
        ))
    return cases, start_idx + len(cases)


# Main
async def main():
    t0 = time.time()
    print("📂  Loading policy files...")
    docs = load_policy_files()
    print(f"    Found {len(docs)} documents: {', '.join(docs.keys())}\n")

    sem           = asyncio.Semaphore(SEMAPHORE_LIMIT)
    token_tracker = {"input": 0, "output": 0}
    all_cases: List[Dict] = []

    # --- Per-document tasks (normal + adversarial) ---
    print("⚙️   Generating per-document QA pairs (async)...")
    idx = 0
    tasks = []
    for doc_id, content in docs.items():
        source_file = next(
            (p.name for p in POLICY_DIR.iterdir() if p.stem == doc_id), doc_id
        )
        tasks.append(("normal",      doc_id, content, source_file))
        tasks.append(("adversarial", doc_id, content, source_file))

    # Run all per-doc tasks concurrently
    per_doc_coros = []
    for kind, doc_id, content, source_file in tasks:
        if kind == "normal":
            per_doc_coros.append(
                generate_normal(doc_id, content, source_file, sem, 0, token_tracker)
            )
        else:
            per_doc_coros.append(
                generate_adversarial(doc_id, content, source_file, sem, 0, token_tracker)
            )

    results = await asyncio.gather(*per_doc_coros, return_exceptions=True)

    # Re-index all cases sequentially
    idx = 0
    doc_counters: Dict[str, int] = {}
    for (kind, doc_id, _, source_file), result in zip(tasks, results):
        if isinstance(result, Exception):
            print(f"  ✗ Exception for {doc_id}/{kind}: {result}")
            continue
        cases, _ = result
        for case in cases:
            doc_counters.setdefault(doc_id, 0)
            case["id"] = f"{doc_id}_{doc_counters[doc_id]:03d}"
            doc_counters[doc_id] += 1
            all_cases.append(case)
        print(f"  ✓ {doc_id} [{kind}]: {len(cases)} cases")

    # --- Cross-document tasks ---
    print("\n⚙️   Generating cross-document QA pairs...")
    cross_coros = []
    for doc_id_a, doc_id_b in CROSS_DOC_PAIRS:
        if doc_id_a in docs and doc_id_b in docs:
            cross_coros.append(
                generate_cross_doc(
                    doc_id_a, docs[doc_id_a],
                    doc_id_b, docs[doc_id_b],
                    sem, 0, token_tracker,
                )
            )
        else:
            missing = doc_id_a if doc_id_a not in docs else doc_id_b
            print(f"  ⚠  Skipping cross-doc pair: {missing} not found")

    cross_results = await asyncio.gather(*cross_coros, return_exceptions=True)
    cross_idx = 0
    for (doc_id_a, doc_id_b), result in zip(CROSS_DOC_PAIRS, cross_results):
        if isinstance(result, Exception):
            print(f"  ✗ Exception cross {doc_id_a}/{doc_id_b}: {result}")
            continue
        cases, _ = result
        for case in cases:
            case["id"] = f"cross_{cross_idx:03d}"
            cross_idx += 1
            all_cases.append(case)
        print(f"  ✓ cross {doc_id_a} × {doc_id_b}: {len(cases)} cases")

    # --- Write output ---
    print(f"\n💾  Writing {len(all_cases)} cases to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # --- Summary ---
    elapsed   = time.time() - t0
    cost_usd  = (
        token_tracker["input"]  / 1000 * COST_PER_1K_INPUT +
        token_tracker["output"] / 1000 * COST_PER_1K_OUTPUT
    )
    n_adv     = sum(1 for c in all_cases if c["metadata"]["adversarial"])
    n_cross   = sum(1 for c in all_cases if c["metadata"]["type"] == "cross-document")
    n_normal  = len(all_cases) - n_adv - n_cross

    print("\n" + "=" * 50)
    print(f"✅  Done in {elapsed:.1f}s")
    print(f"    Total cases      : {len(all_cases)}")
    print(f"    Normal QA        : {n_normal}")
    print(f"    Adversarial      : {n_adv}")
    print(f"    Cross-document   : {n_cross}")
    print(f"    Tokens used      : {token_tracker['input']:,} in / {token_tracker['output']:,} out")
    print(f"    Estimated cost   : ${cost_usd:.4f} USD")
    print(f"    Output           : {OUTPUT_FILE}")
    print("=" * 50)

    if len(all_cases) < 50:
        print(f"\n⚠️  WARNING: Only {len(all_cases)} cases generated (need 50+).")
        print("    Check API key, network, or increase NORMAL_PER_DOC.")
    else:
        print(f"\n🎯  Target met: {len(all_cases)} ≥ 50 cases.")


if __name__ == "__main__":
    asyncio.run(main())

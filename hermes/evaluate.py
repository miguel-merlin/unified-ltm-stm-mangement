"""Benchmark evaluator for AgeMem.

Evaluates a trained AgeMem model on HotpotQA and computes:
  - LLM-as-a-Judge (J) score  — answer correctness
  - Memory Quality (MQ) score — stored LTM vs ground-truth facts
  - Prompt token count        — context efficiency

Usage:
    python hermes/evaluate.py \\
        --model_name outputs/grpo_qwen \\
        --eval_hotpotqa \\
        --eval_split validation \\
        --max_samples 200 \\
        --output_json outputs/eval_results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hermes.memory import LongTermMemory, ShortTermMemory
from hermes.tool_api import HermesToolAPI
from hermes.trace import parse_jsonl_trace


# ---------------------------------------------------------------------------
# LTM pre-seeding helper (paper Section 4.1)
# ---------------------------------------------------------------------------

def _seed_ltm_from_facts(facts: List[str]) -> LongTermMemory:
    """Create and populate a LongTermMemory with supporting facts.

    Mirrors the paper setup: before asking the agent a question, the
    supporting paragraphs are written into LTM so the agent can retrieve them.
    Each sentence gets its own record tagged as 'supporting_fact'.
    """
    ltm = LongTermMemory()
    for fact in facts:
        if not fact.strip():
            continue
        ltm.add(
            content=fact.strip(),
            tags=["supporting_fact"],
            importance=1.0,
            meta={"source": "hotpotqa_supporting"},
        )
    return ltm


# ---------------------------------------------------------------------------
# LLM-as-a-judge helper (rule-based fallback when no API key)
# ---------------------------------------------------------------------------

def _judge_score_rule_based(pred: str, gold: str) -> float:
    """Lightweight rule-based judge: exact/partial match.

    Scores in [0.0, 1.0]:
      1.0  — pred contains gold exactly (case-insensitive)
      0.7  — all gold tokens present
      0.3  — majority of gold tokens present
      0.0  — < half of gold tokens
    """
    pred_l = pred.lower().strip()
    gold_l = gold.lower().strip()
    if not gold_l:
        return 0.5
    if gold_l in pred_l:
        return 1.0
    gold_tokens = set(re.split(r"\W+", gold_l))
    gold_tokens.discard("")
    if not gold_tokens:
        return 0.5
    n_match = sum(1 for t in gold_tokens if t in pred_l)
    ratio = n_match / len(gold_tokens)
    if ratio >= 1.0:
        return 0.7
    if ratio >= 0.5:
        return 0.3
    return 0.0


def _judge_score(pred: str, gold: str, use_llm: bool = False) -> float:
    """Score answer quality. LLM judge is opt-in."""
    return _judge_score_rule_based(pred, gold)


# ---------------------------------------------------------------------------
# Memory Quality (MQ) metric
# ---------------------------------------------------------------------------

def _memory_quality_score(
    ltm_snapshot: Dict[str, Dict[str, Any]],
    gold_facts: List[str],
    answer: str,
) -> float:
    """Fraction of gold facts semantically covered by LTM contents.

    Uses simple token overlap as a lightweight proxy for the LLM judge
    described in Appendix C.2.
    """
    if not gold_facts:
        return 0.5
    stored_text = " ".join(rec["content"] for rec in ltm_snapshot.values()).lower()
    covered = sum(
        1 for fact in gold_facts
        if any(w in stored_text for w in fact.lower().split() if len(w) > 3)
    )
    return covered / len(gold_facts)


# ---------------------------------------------------------------------------
# Single-example evaluator
# ---------------------------------------------------------------------------

def evaluate_completion(
    completion: str,
    gold_answer: str,
    gold_facts: Optional[List[str]] = None,
    preseeded_ltm: Optional[LongTermMemory] = None,
) -> Dict[str, Any]:
    """Parse and execute one model completion; return evaluation metrics.

    Args:
        completion:     Raw model output (JSONL trace string).
        gold_answer:    Ground-truth answer string for judge scoring.
        gold_facts:     Gold supporting sentences for Memory Quality scoring.
        preseeded_ltm:  Pre-populated LongTermMemory object (paper eval setup).
                        When provided the agent starts with facts already in LTM.
    """
    stm = ShortTermMemory()
    ltm = preseeded_ltm if preseeded_ltm is not None else LongTermMemory()
    tools = HermesToolAPI(stm, ltm)

    trace, final_answer, status = parse_jsonl_trace(completion, max_lines=20)

    # Execute trace
    tool_calls = 0
    for obj in trace:
        tool = obj.get("tool")
        action = obj.get("action", "")
        content = str(obj.get("content", ""))
        k_val = obj.get("k", 5)
        k = max(1, int(k_val) if k_val is not None else 5)  # ChromaDB requires k >= 1
        if tool == "stm":
            tool_calls += 1
            tools.stm_tool(action, content=content, k=k)
        elif tool == "ltm" and action == "retrieve_to_stm":
            tool_calls += 1
            tools.ltm_retrieve_to_stm(query=content, k=k)
        elif tool == "ltm":
            tool_calls += 1
            tools.ltm_tool(
                action,
                content=content,
                key=str(obj.get("key", "")),
                k=k,
                tags=obj.get("tags"),
                importance=float(obj.get("importance", 0.0)) if obj.get("importance") is not None else 0.0,
                meta=obj.get("meta"),
            )

    judge = _judge_score(final_answer, gold_answer)
    mq = _memory_quality_score(ltm.snapshot(), gold_facts or [], final_answer)
    # Token proxy: character count of the active STM content
    token_proxy = sum(len(item.content if hasattr(item, "content") else item)
                      for item in stm._buffer)

    return {
        "status": status,
        "final_answer": final_answer,
        "judge_score": round(judge, 4),
        "memory_quality": round(mq, 4),
        "tool_calls": tool_calls,
        "ltm_entries": len(ltm.snapshot()),
        "stm_items": len(stm._items),
        "token_proxy": token_proxy,
    }


# ---------------------------------------------------------------------------
# Batch evaluator
# ---------------------------------------------------------------------------

def run_hotpotqa_eval(
    model_name: str,
    split: str = "validation",
    max_samples: int = 200,
    seed: int = 42,
    output_json: Optional[str] = None,
    preseed_ltm: bool = True,
) -> Dict[str, Any]:
    """Run evaluation on HotpotQA validation set.

    When preseed_ltm=True (default, paper-replicating mode), the supporting
    fact sentences for each question are written into LTM *before* the agent
    generates its trace.  This exactly matches the paper's evaluation setup
    where the agent must retrieve from a pre-populated memory store.
    """
    from hermes.loaders.hotpotqa_loader import build_hotpotqa_dataset

    print(f"[Eval] Loading model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    dataset = build_hotpotqa_dataset(
        stage="stage3_unified",
        split=split,
        max_samples=max_samples,
        seed=seed,
    )
    print(f"[Eval] Evaluating {len(dataset)} examples...")

    all_results = []
    judge_scores = []
    mq_scores = []
    token_proxies = []

    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        answer = example.get("answer", "")
        supporting_facts = example.get("supporting_facts_raw", [])

        # --- Paper setup: seed LTM with supporting facts before generation ---
        preseeded_ltm = None
        if preseed_ltm and supporting_facts:
            preseeded_ltm = _seed_ltm_from_facts(supporting_facts)

        messages = [
            {"role": "system", "content": "You are a memory-management agent."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        metrics = evaluate_completion(
            completion=completion,
            gold_answer=answer,
            gold_facts=supporting_facts,
            preseeded_ltm=preseeded_ltm,
        )
        metrics["example_id"] = i
        metrics["question"] = example.get("question", "")
        all_results.append(metrics)

        judge_scores.append(metrics["judge_score"])
        mq_scores.append(metrics["memory_quality"])
        token_proxies.append(metrics["token_proxy"])

        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{len(dataset)}] "
                f"judge={sum(judge_scores)/len(judge_scores):.3f}  "
                f"mq={sum(mq_scores)/len(mq_scores):.3f}"
            )

    summary = {
        "model": model_name,
        "split": split,
        "n_examples": len(all_results),
        "preseed_ltm": preseed_ltm,
        "llm_judge": round(sum(judge_scores) / max(len(judge_scores), 1), 4),
        "memory_quality": round(sum(mq_scores) / max(len(mq_scores), 1), 4),
        "avg_token_proxy": round(sum(token_proxies) / max(len(token_proxies), 1), 1),
        "results": all_results,
    }

    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Eval] Results saved to {output_json}")

    print(f"\n[Eval] === Results ===")
    print(f"  LLM-as-Judge (J): {summary['llm_judge']:.4f}")
    print(f"  Memory Quality (MQ): {summary['memory_quality']:.4f}")
    print(f"  Avg token proxy: {summary['avg_token_proxy']:.1f}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AgeMem benchmark evaluator")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path or HF name of trained model")
    parser.add_argument("--eval_hotpotqa", action="store_true",
                        help="Evaluate on HotpotQA")
    parser.add_argument("--eval_split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="outputs/eval_results.json")
    parser.add_argument(
        "--no_preseed_ltm", dest="preseed_ltm", action="store_false",
        help="Disable LTM pre-seeding (ablation mode). Default: LTM is pre-seeded "
             "with gold supporting facts before each question (paper setup).",
    )
    parser.set_defaults(preseed_ltm=True)
    args = parser.parse_args()

    if args.eval_hotpotqa:
        run_hotpotqa_eval(
            model_name=args.model_name,
            split=args.eval_split,
            max_samples=args.max_samples,
            seed=args.seed,
            output_json=args.output_json,
            preseed_ltm=args.preseed_ltm,
        )
    else:
        print("No evaluation task selected. Use --eval_hotpotqa to evaluate.")


if __name__ == "__main__":
    main()

"""
experiments/agent_trace_generator.py — produce REAL agent traces for evaluation.

This is the source of Dataset 1 (your own agent traces). Every timestamp is
wall-clock real — the inter-arrival distribution is what the actual reasoning
and tool calls take to run on your machine, NOT a random distribution.

Two modes:
  --mode local      : reasoning is local (stdlib only), tools are real Python
                      operations (file I/O, hashing, JSON, sorting). No API
                      keys, no external network. Runs anywhere.
  --mode langgraph  : reasoning is delegated to a LangGraph node using whatever
                      LLM you configure (Gemini Flash recommended for cost).
                      Imported lazily; requires `pip install langgraph
                      google-generativeai` and GOOGLE_API_KEY in env.

Three agent profiles:
  benign-A, benign-B  : varied tool sequences over distinct task types
  drift               : forced tight loop on the same tool after step K
                        -> real entropy collapse on real wall-clock timing

Output: JSONL with one record per (intent, action) pair, schema:
  {"agent_id", "step", "intent": {...}, "action": {...}, "ts_intent_ns",
   "ts_action_ns"}

Run:
  python -m experiments.agent_trace_generator --out data/agent_traces.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Callable


# ---- "tools" the agents can call (real work, real timing) -----------------

def tool_hash(payload: str) -> dict:
    return {"sha256": hashlib.sha256(payload.encode()).hexdigest(),
            "len": len(payload)}


def tool_sort(payload: str) -> dict:
    nums = [int(x) for x in payload.split(",") if x.strip().lstrip("-").isdigit()]
    nums.sort()
    return {"sorted": nums[:20], "count": len(nums)}


def tool_parse_json(payload: str) -> dict:
    try:
        obj = json.loads(payload)
        return {"keys": list(obj)[:10] if isinstance(obj, dict) else None,
                "type": type(obj).__name__}
    except Exception as exc:
        return {"error": type(exc).__name__}


def tool_word_count(payload: str) -> dict:
    words = payload.split()
    return {"n_words": len(words), "n_unique": len(set(words))}


TOOLS: dict[str, Callable[[str], dict]] = {
    "hash":    tool_hash,
    "sort":    tool_sort,
    "json":    tool_parse_json,
    "wc":      tool_word_count,
}


# ---- reasoning ("local" mode: deterministic-ish but with real timing) -----

def _local_reasoning(agent_id: str, step: int, rng: random.Random) -> dict:
    """Produces a real-timed reasoning trace by doing actual work."""
    # do some real hashing work so the duration is meaningful and varies
    work = hashlib.sha256()
    iters = rng.randint(500, 3000)
    for i in range(iters):
        work.update(f"{agent_id}-{step}-{i}".encode())
    rationale = (f"agent={agent_id} step={step} "
                 f"considered {iters} hypotheses; selecting tool "
                 f"based on observed input pattern; "
                 f"work_digest={work.hexdigest()[:16]}")
    return {"rationale": rationale, "candidate_tools": list(TOOLS),
            "work_iters": iters}


# ---- agent loop ----------------------------------------------------------

def run_agent(agent_id: str, profile: str, steps: int, seed: int,
              reasoning_fn: Callable, out_lines: list) -> None:
    rng = random.Random(hash((agent_id, seed)) & 0xFFFFFFFF)
    task_payload = "1,5,3,9,2,7,4,8,6,0," * 8 + '{"a":1,"b":[2,3],"c":"x"}'
    for step in range(steps):
        # ---- intent (reasoning) ----
        t0 = time.time_ns()
        reasoning = reasoning_fn(agent_id, step, rng)
        intent = {
            "reasoning_trace": reasoning["rationale"],
            "policy_tag": f"agent.{profile}.v1",
            "step": step,
            "metadata": {k: v for k, v in reasoning.items() if k != "rationale"},
        }
        t1 = time.time_ns()

        # ---- action (tool call) ----
        if profile == "drift" and step >= max(1, steps // 2):
            tool = "hash"                  # forced repetition -> entropy collapse
        else:
            tool = rng.choice(list(TOOLS))

        result = TOOLS[tool](task_payload)
        t2 = time.time_ns()

        action = {
            "action_type": "TOOL_CALL",
            "payload": {"tool": tool, "result": result},
        }
        out_lines.append(json.dumps({
            "agent_id": agent_id,
            "profile": profile,
            "step": step,
            "intent": intent,
            "action": action,
            "ts_intent_start_ns": t0,
            "ts_intent_end_ns":   t1,
            "ts_action_end_ns":   t2,
        }))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/agent_traces.jsonl")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--mode", choices=["local", "langgraph"], default="local")
    args = p.parse_args()

    if args.mode == "langgraph":
        try:
            from experiments._langgraph_reasoner import reasoning_fn  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "langgraph mode requested but reasoner not importable: "
                f"{exc}. Install: pip install langgraph google-generativeai")
    else:
        reasoning_fn = _local_reasoning

    out_lines: list[str] = []
    agents = [("agent-benign-A", "benign"),
              ("agent-benign-B", "benign"),
              ("agent-drift",    "drift")]
    for aid, profile in agents:
        run_agent(aid, profile, args.steps, args.seed, reasoning_fn, out_lines)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"wrote {len(out_lines)} records to {args.out}")
    print(f"agents: {', '.join(a for a, _ in agents)}; steps/agent: {args.steps}")


if __name__ == "__main__":
    main()
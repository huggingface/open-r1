"""
Reward functions for Grounded-R1: hallucination-free information extraction via GRPO.

Signal stack (designed for additive composition in GRPOTrainer):
  grounded_format_reward      — structural JSON compliance (max 1.0)
  quote_grounding_reward      — exact substring matching against source context (max 1.0)
  chunk_routing_reward        — v1: quote must come from the correct gold chunk (max 1.0)
  answer_faithfulness_reward  — lexical faithfulness of final_answer to quotes (max 1.0)
  reasoning_quality_reward    — non-trivial CoT encouragement (max 1.0)

Every function matches the GRPOTrainer reward signature:
  fn(completions: list[list[dict]], **kwargs) -> list[float | None]

Dataset columns forwarded automatically by GRPOTrainer into kwargs:
  context_raw      str  — concatenated passage text (all rewards)
  context_chunks   str  — JSON {chunk_id: text}       (chunk_routing_reward only)
  gold_chunk_ids   str  — JSON [chunk_id, ...]         (chunk_routing_reward only)
"""

import json
import re
from typing import Optional


REQUIRED_KEYS = frozenset({"reasoning_path", "is_context_sufficient", "final_answer", "extracted_quotes"})
REQUIRED_QUOTE_KEYS = frozenset({"chunk_id", "exact_quote"})

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "we", "they",
})

_INSUFFICIENT_PHRASES = frozenset({
    "not contain", "insufficient", "cannot answer", "no information",
    "not enough", "does not provide", "not mentioned", "not found",
    "unable to answer", "no relevant", "context does not",
    "does not contain", "no sufficient",
})

_SUFFICIENCY_VOCAB = frozenset({
    "sufficient", "insufficient", "enough", "context", "contains",
    "mentions", "found", "relevant", "available", "provided", "answer",
    "information", "passage", "supports",
})


def _parse_grounded_response(content: str) -> Optional[dict]:
    """Extract and parse the JSON payload from a model completion.

    Tries (in order): ```json block, bare JSON object anywhere in content.
    Returns None if no valid JSON is found.
    """
    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _is_grounded(quote: str, context: str) -> bool:
    """Check if quote is an exact (or whitespace-normalised) substring of context."""
    if not quote:
        return False
    if quote in context:
        return True
    # Normalise consecutive whitespace — handles tokeniser artefacts
    normalised_quote = " ".join(quote.split())
    normalised_ctx = " ".join(context.split())
    return normalised_quote in normalised_ctx


# ---------------------------------------------------------------------------
# Reward 1 — Structural compliance
# ---------------------------------------------------------------------------

def grounded_format_reward(completions, **kwargs) -> list[float]:
    """
    Verifies the JSON schema of the grounded response.

    Additive scoring (max 1.0):
      +0.25  response is valid JSON
      +0.25  all four required top-level keys are present
      +0.25  correct field types (bool, list, str)
      +0.25  every item in extracted_quotes has non-empty chunk_id + exact_quote strings
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]

        parsed = _parse_grounded_response(content)
        if parsed is None:
            rewards.append(0.0)
            continue
        score = 0.25

        if not REQUIRED_KEYS.issubset(parsed.keys()):
            rewards.append(score)
            continue
        score += 0.25

        type_ok = (
            isinstance(parsed["reasoning_path"], str)
            and isinstance(parsed["is_context_sufficient"], bool)
            and isinstance(parsed["final_answer"], str)
            and isinstance(parsed["extracted_quotes"], list)
        )
        if not type_ok:
            rewards.append(score)
            continue
        score += 0.25

        quotes = parsed["extracted_quotes"]
        quotes_ok = not quotes or all(
            isinstance(q, dict)
            and REQUIRED_QUOTE_KEYS.issubset(q.keys())
            and isinstance(q["chunk_id"], str)
            and isinstance(q["exact_quote"], str)
            and len(q["exact_quote"].strip()) > 0
            for q in quotes
        )
        if quotes_ok:
            score += 0.25

        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# Reward 2 — Quote grounding (core anti-hallucination signal)
# ---------------------------------------------------------------------------

def quote_grounding_reward(completions, context_raw: list[str], **kwargs) -> list[float]:
    """
    Verifies that every extracted_quote is a verbatim substring of the source context.

    This is the primary reward signal for eradicating factual hallucinations.

    Scoring logic:
      is_context_sufficient=True,  quotes non-empty  →  mean(exact_match per quote)  [0, 1]
      is_context_sufficient=True,  quotes empty      →  0.0   (claimed sufficient, cited nothing)
      is_context_sufficient=False, quotes empty      →  1.0   (correctly abstained)
      is_context_sufficient=False, quotes non-empty  →  scored normally (partial credit)
    """
    rewards = []
    for completion, ctx in zip(completions, context_raw):
        content = completion[0]["content"]
        parsed = _parse_grounded_response(content)

        if parsed is None or not REQUIRED_KEYS.issubset(parsed.keys()):
            rewards.append(0.0)
            continue

        quotes = parsed["extracted_quotes"]
        is_sufficient = parsed["is_context_sufficient"]

        if not is_sufficient and not quotes:
            rewards.append(1.0)
            continue

        if is_sufficient and not quotes:
            rewards.append(0.0)
            continue

        valid_quotes = [
            q for q in quotes
            if isinstance(q, dict) and q.get("exact_quote", "").strip()
        ]
        if not valid_quotes:
            rewards.append(0.0)
            continue

        grounded_count = sum(1 for q in valid_quotes if _is_grounded(q["exact_quote"], ctx))
        rewards.append(grounded_count / len(valid_quotes))

    return rewards


# ---------------------------------------------------------------------------
# Reward 3 — Answer faithfulness
# ---------------------------------------------------------------------------

def answer_faithfulness_reward(completions, **kwargs) -> list[float]:
    """
    Checks that the final_answer is lexically supported by the extracted_quotes,
    preventing the model from using grounded quotes as a cover while hallucinating
    the synthesised answer.

    Score = |content_tokens(answer) ∩ content_tokens(all_quotes)| / |content_tokens(answer)|

    For is_context_sufficient=False: checks that the answer contains an abstention phrase.
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        parsed = _parse_grounded_response(content)

        if parsed is None or not REQUIRED_KEYS.issubset(parsed.keys()):
            rewards.append(0.0)
            continue

        final_answer = parsed["final_answer"]
        quotes = parsed["extracted_quotes"]
        is_sufficient = parsed["is_context_sufficient"]

        if not is_sufficient:
            answer_lower = final_answer.lower()
            reward = 1.0 if any(p in answer_lower for p in _INSUFFICIENT_PHRASES) else 0.2
            rewards.append(reward)
            continue

        if not quotes or not final_answer.strip():
            rewards.append(0.0)
            continue

        quote_text = " ".join(
            q.get("exact_quote", "") for q in quotes if isinstance(q, dict)
        )
        quote_tokens = {
            t.lower()
            for t in re.findall(r"\b\w+\b", quote_text)
            if t.lower() not in _STOPWORDS
        }
        answer_tokens = {
            t.lower()
            for t in re.findall(r"\b\w+\b", final_answer)
            if t.lower() not in _STOPWORDS
        }

        if not answer_tokens:
            rewards.append(0.0)
            continue

        overlap = len(answer_tokens & quote_tokens) / len(answer_tokens)
        rewards.append(float(overlap))

    return rewards


# ---------------------------------------------------------------------------
# Reward 4 — Reasoning quality (soft, encourages CoT)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Reward 3b — Chunk routing (v1, multi-chunk contexts)
# ---------------------------------------------------------------------------

def chunk_routing_reward(
    completions,
    context_chunks: list[str],
    gold_chunk_ids: list[str],
    **kwargs,
) -> list[float]:
    """
    Verifies that each extracted_quote comes from a gold (supporting) chunk,
    not from a distractor chunk.

    This is the key v1 reward for multi-passage contexts (e.g. HotpotQA).
    It sits on top of quote_grounding_reward: a quote must both exist verbatim
    in the source AND originate from the correct passage.

    Args:
        context_chunks:  list of JSON strings, each encoding {chunk_id: text}.
        gold_chunk_ids:  list of JSON strings, each encoding [chunk_id, ...].
                         Empty list means the question is unanswerable.

    Scoring:
      is_context_sufficient=True, gold non-empty, quotes present:
          mean(1.0 if chunk_id ∈ gold_ids AND exact_quote ∈ chunk_text else 0.0)
      is_context_sufficient=False, gold empty (truly unanswerable):
          1.0 — correct abstention
      is_context_sufficient=True,  gold empty:
          0.0 — claimed sufficient when nothing is answerable
      is_context_sufficient=False, gold non-empty:
          0.0 — wrongly abstained when context is sufficient
      No quotes when sufficient:
          0.0
    """
    rewards = []
    for completion, chunks_json, gold_json in zip(completions, context_chunks, gold_chunk_ids):
        content = completion[0]["content"]
        parsed = _parse_grounded_response(content)

        if parsed is None or not REQUIRED_KEYS.issubset(parsed.keys()):
            rewards.append(0.0)
            continue

        try:
            chunk_dict: dict[str, str] = json.loads(chunks_json)
            gold_ids: list[str] = json.loads(gold_json)
        except (json.JSONDecodeError, TypeError):
            rewards.append(0.0)
            continue

        is_sufficient = parsed["is_context_sufficient"]
        quotes = parsed["extracted_quotes"]
        is_truly_unanswerable = len(gold_ids) == 0

        # Correct abstention
        if is_truly_unanswerable and not is_sufficient and not quotes:
            rewards.append(1.0)
            continue

        # Wrong claim of sufficiency for unanswerable
        if is_truly_unanswerable and is_sufficient:
            rewards.append(0.0)
            continue

        # Wrong abstention for answerable
        if not is_truly_unanswerable and not is_sufficient:
            rewards.append(0.0)
            continue

        # No quotes when claimed sufficient
        valid_quotes = [q for q in quotes if isinstance(q, dict) and q.get("exact_quote", "").strip()]
        if not valid_quotes:
            rewards.append(0.0)
            continue

        # Score each quote: gold chunk + grounded in that chunk
        scores = []
        for q in valid_quotes:
            cid = q.get("chunk_id", "")
            eq = q.get("exact_quote", "")
            chunk_text = chunk_dict.get(cid, "")
            routed_correctly = cid in gold_ids
            grounded_in_chunk = _is_grounded(eq, chunk_text)
            scores.append(1.0 if (routed_correctly and grounded_in_chunk) else 0.0)

        rewards.append(sum(scores) / len(scores))

    return rewards


# ---------------------------------------------------------------------------
# Reward 4 — Reasoning quality (soft, encourages CoT)
# ---------------------------------------------------------------------------

def reasoning_quality_reward(completions, **kwargs) -> list[float]:
    """
    Soft reward encouraging a substantive chain-of-thought in reasoning_path.

    +0.5   reasoning_path is non-empty
    +0.3   mentions sufficiency-assessment vocabulary
    +0.2   at least 20 words (penalises trivial one-liners)
    """
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        parsed = _parse_grounded_response(content)

        if parsed is None or "reasoning_path" not in parsed:
            rewards.append(0.0)
            continue

        reasoning = parsed["reasoning_path"]
        score = 0.0

        if reasoning.strip():
            score += 0.5
            if any(kw in reasoning.lower() for kw in _SUFFICIENCY_VOCAB):
                score += 0.3
            if len(reasoning.split()) >= 20:
                score += 0.2

        rewards.append(score)
    return rewards

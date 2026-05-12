"""Unit tests for Grounded-R1 reward functions."""

import json

from open_r1.grounded_rewards import (
    _is_grounded,
    _parse_grounded_response,
    answer_faithfulness_reward,
    chunk_routing_reward,
    grounded_format_reward,
    quote_grounding_reward,
    reasoning_quality_reward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content: str) -> list[list[dict]]:
    """Wrap raw content into the GRPOTrainer completion format."""
    return [[{"role": "assistant", "content": content}]]


def _json_completion(payload: dict) -> list[list[dict]]:
    return _make_completion(json.dumps(payload))


GOOD_RESPONSE = {
    "reasoning_path": (
        "I will check whether the context contains information about the capital of France. "
        "The context states 'Paris is the capital of France', which is sufficient to answer the question. "
        "I will extract the relevant quote directly from the provided passage."
    ),
    "is_context_sufficient": True,
    "final_answer": "Paris is the capital of France.",
    "extracted_quotes": [
        {"chunk_id": "doc_0", "exact_quote": "Paris is the capital of France"}
    ],
}

CONTEXT = "Paris is the capital of France and a major European city."


# ---------------------------------------------------------------------------
# _parse_grounded_response
# ---------------------------------------------------------------------------

class TestParseGroundedResponse:
    def test_json_block(self):
        content = f"```json\n{json.dumps(GOOD_RESPONSE)}\n```"
        assert _parse_grounded_response(content) == GOOD_RESPONSE

    def test_bare_json(self):
        assert _parse_grounded_response(json.dumps(GOOD_RESPONSE)) == GOOD_RESPONSE

    def test_embedded_in_text(self):
        content = f"Here is my answer: {json.dumps(GOOD_RESPONSE)} Done."
        result = _parse_grounded_response(content)
        assert result == GOOD_RESPONSE

    def test_invalid_returns_none(self):
        assert _parse_grounded_response("not json at all") is None
        assert _parse_grounded_response("") is None


# ---------------------------------------------------------------------------
# _is_grounded
# ---------------------------------------------------------------------------

class TestIsGrounded:
    def test_exact_match(self):
        assert _is_grounded("Paris is the capital", "Paris is the capital of France")

    def test_no_match(self):
        assert not _is_grounded("London is the capital", "Paris is the capital of France")

    def test_whitespace_normalisation(self):
        assert _is_grounded("Paris  is  the  capital", "Paris is the capital of France")

    def test_empty_quote_is_false(self):
        assert not _is_grounded("", "Paris is the capital of France")


# ---------------------------------------------------------------------------
# grounded_format_reward
# ---------------------------------------------------------------------------

class TestGroundedFormatReward:
    def test_perfect_response_scores_1(self):
        rewards = grounded_format_reward(_json_completion(GOOD_RESPONSE))
        assert rewards == [1.0]

    def test_invalid_json_scores_0(self):
        rewards = grounded_format_reward(_make_completion("this is not json"))
        assert rewards == [0.0]

    def test_missing_key_partial_score(self):
        incomplete = {k: v for k, v in GOOD_RESPONSE.items() if k != "extracted_quotes"}
        rewards = grounded_format_reward(_json_completion(incomplete))
        assert rewards == [0.25]  # JSON parseable (+0.25), but missing key → stops there

    def test_wrong_type_for_is_context_sufficient(self):
        bad = dict(GOOD_RESPONSE, is_context_sufficient="yes")
        rewards = grounded_format_reward(_json_completion(bad))
        assert rewards == [0.5]  # 0.25 JSON + 0.25 keys, type check fails

    def test_empty_quotes_list_is_valid(self):
        """Empty extracted_quotes is structurally fine (unanswerable case)."""
        payload = dict(GOOD_RESPONSE, is_context_sufficient=False, extracted_quotes=[])
        rewards = grounded_format_reward(_json_completion(payload))
        assert rewards == [1.0]

    def test_batch_processing(self):
        completions = _json_completion(GOOD_RESPONSE) + _make_completion("garbage")
        rewards = grounded_format_reward(completions)
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0

    def test_empty_exact_quote_string_penalised(self):
        bad_quotes = dict(GOOD_RESPONSE, extracted_quotes=[{"chunk_id": "a", "exact_quote": ""}])
        rewards = grounded_format_reward(_json_completion(bad_quotes))
        assert rewards == [0.75]  # fails last +0.25


# ---------------------------------------------------------------------------
# quote_grounding_reward
# ---------------------------------------------------------------------------

class TestQuoteGroundingReward:
    def test_exact_grounded_quote_scores_1(self):
        rewards = quote_grounding_reward(_json_completion(GOOD_RESPONSE), context_raw=[CONTEXT])
        assert rewards == [1.0]

    def test_hallucinated_quote_scores_0(self):
        hallucinated = dict(
            GOOD_RESPONSE,
            extracted_quotes=[{"chunk_id": "doc_0", "exact_quote": "London is the capital of England"}],
        )
        rewards = quote_grounding_reward(_json_completion(hallucinated), context_raw=[CONTEXT])
        assert rewards == [0.0]

    def test_partial_grounding(self):
        mixed = dict(
            GOOD_RESPONSE,
            extracted_quotes=[
                {"chunk_id": "doc_0", "exact_quote": "Paris is the capital of France"},
                {"chunk_id": "doc_0", "exact_quote": "This sentence does not exist anywhere"},
            ],
        )
        rewards = quote_grounding_reward(_json_completion(mixed), context_raw=[CONTEXT])
        assert rewards == [0.5]

    def test_correct_abstention_scores_1(self):
        abstain = dict(
            GOOD_RESPONSE,
            is_context_sufficient=False,
            extracted_quotes=[],
            final_answer="The provided context does not contain sufficient information.",
        )
        rewards = quote_grounding_reward(_json_completion(abstain), context_raw=[CONTEXT])
        assert rewards == [1.0]

    def test_false_sufficiency_with_no_quotes_scores_0(self):
        bad = dict(GOOD_RESPONSE, is_context_sufficient=True, extracted_quotes=[])
        rewards = quote_grounding_reward(_json_completion(bad), context_raw=[CONTEXT])
        assert rewards == [0.0]

    def test_invalid_completion_scores_0(self):
        rewards = quote_grounding_reward(_make_completion("not json"), context_raw=[CONTEXT])
        assert rewards == [0.0]


# ---------------------------------------------------------------------------
# answer_faithfulness_reward
# ---------------------------------------------------------------------------

class TestAnswerFaithfulnessReward:
    def test_faithful_answer_scores_high(self):
        rewards = answer_faithfulness_reward(_json_completion(GOOD_RESPONSE))
        assert rewards[0] > 0.5

    def test_abstention_answer_for_insufficient_context(self):
        abstain = dict(
            GOOD_RESPONSE,
            is_context_sufficient=False,
            extracted_quotes=[],
            final_answer="The provided context does not contain sufficient information to answer this question.",
        )
        rewards = answer_faithfulness_reward(_json_completion(abstain))
        assert rewards == [1.0]

    def test_hallucinated_answer_with_good_quotes_penalised(self):
        hallucinated_answer = dict(
            GOOD_RESPONSE,
            final_answer="Jupiter is the largest planet in the solar system.",
        )
        rewards = answer_faithfulness_reward(_json_completion(hallucinated_answer))
        assert rewards[0] < 0.2

    def test_invalid_completion_scores_0(self):
        rewards = answer_faithfulness_reward(_make_completion("garbage"))
        assert rewards == [0.0]


# ---------------------------------------------------------------------------
# reasoning_quality_reward
# ---------------------------------------------------------------------------

class TestReasoningQualityReward:
    def test_good_reasoning_scores_1(self):
        rewards = reasoning_quality_reward(_json_completion(GOOD_RESPONSE))
        assert rewards == [1.0]

    def test_empty_reasoning_scores_0(self):
        empty_reasoning = dict(GOOD_RESPONSE, reasoning_path="")
        rewards = reasoning_quality_reward(_json_completion(empty_reasoning))
        assert rewards == [0.0]

    def test_short_reasoning_without_vocab_scores_half(self):
        # No sufficiency vocab words, under 20 words → only +0.5 for non-empty
        short = dict(GOOD_RESPONSE, reasoning_path="Checking the text now.")
        rewards = reasoning_quality_reward(_json_completion(short))
        assert rewards == [0.5]

    def test_invalid_completion_scores_0(self):
        rewards = reasoning_quality_reward(_make_completion("not json"))
        assert rewards == [0.0]


# ---------------------------------------------------------------------------
# End-to-end: reward interaction
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_perfect_response_all_rewards_max(self):
        completions = _json_completion(GOOD_RESPONSE)
        ctx = [CONTEXT]
        fmt = grounded_format_reward(completions)
        grnd = quote_grounding_reward(completions, context_raw=ctx)
        faith = answer_faithfulness_reward(completions)
        rsn = reasoning_quality_reward(completions)
        assert fmt == [1.0]
        assert grnd == [1.0]
        assert faith[0] > 0.5
        assert rsn == [1.0]

    def test_hallucinated_response_penalised_across_signals(self):
        hallucinated = {
            "reasoning_path": "The context mentions that London is the capital.",
            "is_context_sufficient": True,
            "final_answer": "London is the capital of the UK.",
            "extracted_quotes": [
                {"chunk_id": "doc_0", "exact_quote": "London is the capital of the United Kingdom"}
            ],
        }
        completions = _json_completion(hallucinated)
        ctx = [CONTEXT]
        grnd = quote_grounding_reward(completions, context_raw=ctx)
        faith = answer_faithfulness_reward(completions)
        assert grnd == [0.0]   # quote not in context → grounding = 0
        assert faith[0] > 0.5  # answer IS consistent with the (hallucinated) quote — faithfulness is internal


# ---------------------------------------------------------------------------
# chunk_routing_reward (v1 — multi-chunk contexts)
# ---------------------------------------------------------------------------

# Multi-chunk test fixtures
CHUNK_A = "Paris is the capital of France and a major European city."
CHUNK_B = "Berlin is the capital of Germany, located in central Europe."
CHUNK_C = "London is the capital of the United Kingdom and a financial hub."

MULTI_CHUNKS = json.dumps({
    "France_0": CHUNK_A,
    "Germany_0": CHUNK_B,
    "UK_0": CHUNK_C,
})
GOLD_IDS = json.dumps(["France_0"])  # only France chunk is relevant


def _routing_completion(is_sufficient: bool, answer: str, quotes: list[dict],
                        reasoning: str = "Checked all chunks for relevant information about France.") -> list:
    payload = {
        "reasoning_path": reasoning,
        "is_context_sufficient": is_sufficient,
        "final_answer": answer,
        "extracted_quotes": quotes,
    }
    return _json_completion(payload)


class TestChunkRoutingReward:
    def test_perfect_routing_scores_1(self):
        """Quote from gold chunk, verbatim substring → 1.0."""
        completions = _routing_completion(
            True, "Paris",
            [{"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"}],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [1.0]

    def test_distractor_chunk_scores_0(self):
        """Quote from a distractor chunk, even if verbatim → 0.0."""
        completions = _routing_completion(
            True, "Berlin",
            [{"chunk_id": "Germany_0", "exact_quote": "Berlin is the capital of Germany"}],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [0.0]

    def test_gold_chunk_but_hallucinated_quote_scores_0(self):
        """Correct chunk id, but exact_quote not in that chunk → 0.0."""
        completions = _routing_completion(
            True, "Paris",
            [{"chunk_id": "France_0", "exact_quote": "Paris has always been the greatest city"}],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [0.0]

    def test_partial_routing_half_score(self):
        """One routed correctly, one from distractor → 0.5."""
        completions = _routing_completion(
            True, "Paris and Berlin",
            [
                {"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"},
                {"chunk_id": "Germany_0", "exact_quote": "Berlin is the capital of Germany"},
            ],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [0.5]

    def test_correct_abstention_unanswerable(self):
        """gold_chunk_ids=[], is_context_sufficient=False, no quotes → 1.0."""
        completions = _routing_completion(
            False,
            "The provided context does not contain sufficient information.",
            [],
            reasoning="None of the provided chunks mention the capital of Mars, context is insufficient.",
        )
        empty_gold = json.dumps([])
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[empty_gold])
        assert rewards == [1.0]

    def test_wrong_sufficiency_claim_for_unanswerable(self):
        """gold_chunk_ids=[], is_context_sufficient=True → 0.0 (should have abstained)."""
        completions = _routing_completion(
            True, "Paris",
            [{"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"}],
        )
        empty_gold = json.dumps([])
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[empty_gold])
        assert rewards == [0.0]

    def test_wrong_abstention_for_answerable(self):
        """gold_chunk_ids non-empty, is_context_sufficient=False → 0.0 (missed the answer)."""
        completions = _routing_completion(
            False,
            "The provided context does not contain sufficient information.",
            [],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [0.0]

    def test_invalid_json_scores_0(self):
        rewards = chunk_routing_reward(
            _make_completion("not json"),
            context_chunks=[MULTI_CHUNKS],
            gold_chunk_ids=[GOLD_IDS],
        )
        assert rewards == [0.0]

    def test_no_quotes_when_sufficient_scores_0(self):
        completions = _routing_completion(True, "Paris", [])
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[GOLD_IDS])
        assert rewards == [0.0]

    def test_multi_gold_chunks_hotpotqa_style(self):
        """Two gold chunks (HotpotQA bridge questions), both cited correctly → 1.0."""
        two_gold = json.dumps(["France_0", "Germany_0"])
        completions = _routing_completion(
            True, "Both Paris and Berlin are capitals in Europe.",
            [
                {"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"},
                {"chunk_id": "Germany_0", "exact_quote": "Berlin is the capital of Germany"},
            ],
        )
        rewards = chunk_routing_reward(completions, context_chunks=[MULTI_CHUNKS], gold_chunk_ids=[two_gold])
        assert rewards == [1.0]

    def test_batch_processing(self):
        """Two examples processed in one call."""
        comp_good = _routing_completion(
            True, "Paris",
            [{"chunk_id": "France_0", "exact_quote": "Paris is the capital of France"}],
        )
        comp_bad = _routing_completion(
            True, "Berlin",
            [{"chunk_id": "Germany_0", "exact_quote": "Berlin is the capital of Germany"}],
        )
        completions = comp_good + comp_bad
        rewards = chunk_routing_reward(
            completions,
            context_chunks=[MULTI_CHUNKS, MULTI_CHUNKS],
            gold_chunk_ids=[GOLD_IDS, GOLD_IDS],
        )
        assert rewards == [1.0, 0.0]

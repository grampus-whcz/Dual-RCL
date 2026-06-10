"""
LLM-based voting system for Win Rate (W-rate) computation.

Replaces the paper's "three experienced operations engineers" with three
LLM voters that independently evaluate and vote for the best method per
test case.  The three voters use the same model but different temperatures
to simulate independent expert judgments:

  - Voter 1: temperature=0.1 (conservative / precise)
  - Voter 2: temperature=0.3 (balanced)
  - Voter 3: temperature=0.5 (exploratory)

Tie-breaking mimics the paper's "discussion + second round": when no
method receives a majority (e.g., 1-1-1 split), an arbitration LLM call
resolves the tie based on all three voters' reasoning.

Usage:
    voter = LLMVoter(model_name="glm-4.7")
    result = voter.vote_case(case_id, methods_output, ground_truth)
    win_rates = voter.compute_win_rates(batch_results, method_names)
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from evaluation.llm_client import create_client

logger = logging.getLogger(__name__)


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class VoteResult:
    """Result of voting for a single test case."""
    case_id: str
    votes: List[dict]              # Each voter's {vote, ranking, reasoning}
    winner: str                    # Majority-vote winner (or tie-break winner)
    is_tie: bool = False           # Whether tie-breaking was needed
    tie_break_reasoning: str = ''  # Only set if is_tie=True


# =====================================================================
# Prompt Templates
# =====================================================================

VOTER_SYSTEM_PROMPT = """You are an experienced operations engineer with \
deep expertise in microservice failure diagnosis. You are evaluating the \
quality of reasoning outputs from different diagnosis methods. You must \
select the SINGLE BEST method for this specific test case based on the \
reasoning quality of each method's output."""

TIEBREAK_SYSTEM_PROMPT = """You are a senior operations architect serving as \
the final arbitrator in a disagreement between three engineers. Each engineer \
has evaluated the same set of diagnosis methods but could not reach consensus. \
Review their reasoning and make the final decision."""


def _build_voter_prompt(
    methods_output: Dict[str, Dict[str, str]],
    ground_truth: str,
    case_context: str = '',
) -> str:
    """Build the voting prompt for a single test case.

    Args:
        methods_output: {method_name: {expert_type: reasoning_text, ...}, ...}
        ground_truth: Ground truth root cause service name.
        case_context: Optional context string (date, time, etc.).
    """
    methods_section = []
    for method_name, expert_outputs in methods_output.items():
        parts = [f"### Method: {method_name}\n"]
        for expert, text in expert_outputs.items():
            # Truncate very long texts to keep prompt manageable
            display = text[:2000] + ('...' if len(text) > 2000 else '')
            parts.append(f"**{expert} output:**\n{display}\n")
        methods_section.append('\n'.join(parts))

    return f"""Evaluate the following microservice failure diagnosis methods \
and vote for the BEST one.

## Test Case Context
- Ground truth root cause: **{ground_truth}**
{f"- {case_context}" if case_context else ""}

## Method Outputs
{chr(10).join(methods_section)}

## Evaluation Criteria
1. **Accuracy**: Does the reasoning correctly identify the root cause as {ground_truth}?
2. **Completeness**: Are all relevant observations captured?
3. **Logical coherence**: Is the reasoning chain sound and well-structured?
4. **Specificity**: Does it reference specific data (service names, metrics, timestamps)?
5. **Clarity**: Is the explanation clear and professional?

## Instructions
Evaluate ALL methods above and vote for the SINGLE BEST method.
Output a JSON object in this exact format:
```json
{{
    "vote": "method_name",
    "ranking": ["best_method", "second_best", "..."],
    "reasoning": "Brief justification for your vote (2-3 sentences)"
}}
```"""


def _build_tiebreak_prompt(
    votes: List[dict],
    methods_output: Dict[str, Dict[str, str]],
    ground_truth: str,
) -> str:
    """Build the arbitration prompt for tie-breaking."""
    voter_reasoning = []
    for i, v in enumerate(votes):
        voter_reasoning.append(
            f"Engineer {i+1} voted for **{v.get('vote', 'unknown')}**: "
            f"{v.get('reasoning', 'No reasoning provided.')}"
        )

    return f"""Three engineers could not reach agreement on the best \
diagnosis method for this case. Review their reasoning and make the final call.

## Ground Truth Root Cause: {ground_truth}

## Engineer Opinions:
{chr(10).join(voter_reasoning)}

## Available Methods: {', '.join(methods_output.keys())}

Review each engineer's reasoning, then select the SINGLE BEST method.
Output a JSON object:
```json
{{
    "final_vote": "method_name",
    "reasoning": "Your justification for breaking the tie"
}}
```"""


# =====================================================================
# Helpers
# =====================================================================

def _extract_vote_from_text(
    text: str,
    valid_methods: List[str],
) -> Optional[dict]:
    """Try to extract a vote from raw LLM text when JSON parsing fails.

    Looks for method names mentioned in the text and returns the first
    one found as the vote.

    Args:
        text: Raw LLM response text.
        valid_methods: List of valid method names to look for.

    Returns:
        dict with 'vote', 'ranking', 'reasoning' or None.
    """
    import re as _re

    if not text or not text.strip():
        return None

    # Look for explicit "vote": "Method" patterns (even in broken JSON)
    vote_match = _re.search(
        r'"vote"\s*:\s*"([^"]+)"', text, _re.IGNORECASE
    )
    if vote_match:
        voted = vote_match.group(1)
        # Find the closest valid method name
        for m in valid_methods:
            if m.lower() == voted.lower().strip():
                return {
                    'vote': m,
                    'ranking': [m] + [x for x in valid_methods if x != m],
                    'reasoning': 'Extracted from partial response.',
                }

    # Look for method name mentioned as the best/preferred/winner
    for m in valid_methods:
        patterns = [
            rf'\bvotes?\s+(?:for\s+)?["\']?{ _re.escape(m)}["\']?',
            rf'\bbest\s+(?:method\s+)?(?:is\s+)?["\']?{_re.escape(m)}["\']?',
            rf'\bprefer\s+["\']?{_re.escape(m)}["\']?',
            rf'\bselect\s+(?:method\s+)?["\']?{_re.escape(m)}["\']?',
            rf'\bchoose\s+["\']?{_re.escape(m)}["\']?',
            rf'\bwinner\s*(?:is\s+)?["\']?{_re.escape(m)}["\']?',
        ]
        for pat in patterns:
            if _re.search(pat, text, _re.IGNORECASE):
                return {
                    'vote': m,
                    'ranking': [m] + [x for x in valid_methods if x != m],
                    'reasoning': 'Extracted from text pattern.',
                }

    # Last resort: find first valid method name mentioned in the text
    # that appears in a "voting" context (after "vote", "best", "choose")
    text_lower = text.lower()
    first_mentions = []
    for m in valid_methods:
        idx = text_lower.find(m.lower())
        if idx >= 0:
            first_mentions.append((idx, m))
    if first_mentions:
        first_mentions.sort()
        best = first_mentions[0][1]
        return {
            'vote': best,
            'ranking': [best] + [x for x in valid_methods if x != best],
            'reasoning': 'Extracted as first-mentioned method.',
        }

    return None


# =====================================================================
# LLM Voter
# =====================================================================

class LLMVoter:
    """LLM-based voting system that replaces human engineer evaluation.

    Three LLM voters with different temperatures independently evaluate
    each test case.  Majority vote determines the winner.  Ties are
    resolved through an arbitration call.

    Usage:
        voter = LLMVoter(model_name="glm-4.7")
        result = voter.vote_case(
            case_id="0701_11-50",
            methods_output={
                "LocaleXpert": {"TraceExpert": "...", ...},
                "DualChannel": {"TraceExpert": "...", ...},
            },
            ground_truth="mobservice1",
        )
    """

    # Three voters with increasing temperature to simulate independence
    VOTER_TEMPERATURES = [0.1, 0.3, 0.5]

    def __init__(
        self,
        model_name: str = "glm-4.7",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.client = create_client(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    def _single_vote(
        self,
        methods_output: Dict[str, Dict[str, str]],
        ground_truth: str,
        temperature: float,
        case_context: str = '',
    ) -> dict:
        """Get one voter's vote."""
        prompt = _build_voter_prompt(methods_output, ground_truth, case_context)
        result = self.client.call_json(
            VOTER_SYSTEM_PROMPT, prompt,
            temperature=temperature,
        )

        if result is None:
            # Fallback: try to extract vote from raw text
            raw_text = self.client.call(
                VOTER_SYSTEM_PROMPT, prompt,
                temperature=temperature,
            )
            if raw_text:
                extracted = _extract_vote_from_text(raw_text, list(methods_output.keys()))
                if extracted:
                    logger.info(f"[LLMVoter] Recovered vote from raw text: "
                                f"{extracted['vote']}")
                    return extracted

            logger.warning("[LLMVoter] Voter returned unparseable response; "
                           "defaulting to first method.")
            return {
                'vote': list(methods_output.keys())[0] if methods_output else 'unknown',
                'ranking': list(methods_output.keys()),
                'reasoning': 'Failed to parse LLM response.',
            }

        # Ensure 'vote' key exists
        if 'vote' not in result:
            result['vote'] = list(methods_output.keys())[0] if methods_output else 'unknown'
        return result

    def _break_tie(
        self,
        votes: List[dict],
        methods_output: Dict[str, Dict[str, str]],
        ground_truth: str,
    ) -> str:
        """Resolve a tie through arbitration."""
        prompt = _build_tiebreak_prompt(votes, methods_output, ground_truth)
        result = self.client.call_json(TIEBREAK_SYSTEM_PROMPT, prompt)

        if result and 'final_vote' in result:
            reasoning = result.get('reasoning', '')
            logger.info(f"[LLMVoter] Tie broken: {result['final_vote']} "
                        f"({reasoning})")
            return result['final_vote'], reasoning

        # Fallback: pick the first method that appears most in rankings
        logger.warning("[LLMVoter] Tie-break failed; using fallback.")
        all_rankings = [v.get('ranking', []) for v in votes]
        first_choices = [r[0] for r in all_rankings if r]
        if first_choices:
            winner = Counter(first_choices).most_common(1)[0][0]
            return winner, 'Fallback: most common first-choice ranking.'
        return list(methods_output.keys())[0], 'Fallback: first method listed.'

    def vote_case(
        self,
        case_id: str,
        methods_output: Dict[str, Dict[str, str]],
        ground_truth: str,
        case_context: str = '',
    ) -> VoteResult:
        """Run the full voting process for one test case.

        Args:
            case_id: Case identifier (e.g., "0701_11-50").
            methods_output: {method_name: {expert_type: reasoning_text}}.
            ground_truth: Ground truth root cause service.
            case_context: Optional context (date, time, etc.).

        Returns:
            VoteResult with votes, winner, and tie information.
        """
        if not methods_output:
            return VoteResult(case_id=case_id, votes=[], winner='')

        # Only one method → automatic winner
        if len(methods_output) == 1:
            only_method = list(methods_output.keys())[0]
            return VoteResult(
                case_id=case_id,
                votes=[{'vote': only_method, 'ranking': [only_method],
                        'reasoning': 'Only method available.'}],
                winner=only_method,
                is_tie=False,
            )

        # Stage 1: Three independent votes
        votes = []
        for temp in self.VOTER_TEMPERATURES:
            vote = self._single_vote(
                methods_output, ground_truth, temp, case_context
            )
            votes.append(vote)
            logger.info(f"[LLMVoter] case={case_id} temp={temp} "
                        f"vote={vote.get('vote', '?')}")

        # Stage 2: Determine majority
        vote_counts = Counter(v.get('vote', '') for v in votes)
        total_voters = len(votes)
        majority_threshold = (total_voters + 1) // 2  # ceil(N/2)

        # Check for majority
        winner = None
        for method, count in vote_counts.most_common():
            if count >= majority_threshold:
                winner = method
                break

        is_tie = winner is None

        # Stage 3: Tie-breaking if needed
        tie_reasoning = ''
        if is_tie:
            logger.info(f"[LLMVoter] case={case_id}: Tie detected "
                        f"({dict(vote_counts)}), running arbitration.")
            winner, tie_reasoning = self._break_tie(
                votes, methods_output, ground_truth
            )

        return VoteResult(
            case_id=case_id,
            votes=votes,
            winner=winner,
            is_tie=is_tie,
            tie_break_reasoning=tie_reasoning,
        )

    def vote_batch(
        self,
        cases: List[Dict[str, Any]],
    ) -> List[VoteResult]:
        """Vote on a batch of test cases.

        Args:
            cases: List of dicts, each with:
                - 'case_id': str
                - 'methods_output': {method: {expert: text}}
                - 'ground_truth': str
                - 'case_context': str (optional)

        Returns:
            List of VoteResult, one per case.
        """
        results = []
        for i, case in enumerate(cases):
            logger.info(f"[LLMVoter] Processing case {i+1}/{len(cases)}: "
                        f"{case.get('case_id', '?')}")
            result = self.vote_case(
                case_id=case['case_id'],
                methods_output=case['methods_output'],
                ground_truth=case['ground_truth'],
                case_context=case.get('case_context', ''),
            )
            results.append(result)
        return results

    @staticmethod
    def compute_win_rates(
        results: List[VoteResult],
        method_names: List[str],
    ) -> Dict[str, float]:
        """Compute W-rate for each method from VoteResult list.

        W-rate = proportion of cases where the method won (majority vote
        or tie-break).

        Args:
            results: List of VoteResult from vote_batch().
            method_names: List of method names to compute rates for.

        Returns:
            Dict mapping method_name -> win_rate (0.0 to 1.0).
        """
        if not results:
            return {m: 0.0 for m in method_names}

        win_counts = Counter(r.winner for r in results)
        total = len(results)

        return {
            method: round(win_counts.get(method, 0) / total, 4)
            for method in method_names
        }

    @staticmethod
    def compute_tie_rate(results: List[VoteResult]) -> float:
        """Compute the proportion of cases that required tie-breaking."""
        if not results:
            return 0.0
        ties = sum(1 for r in results if r.is_tie)
        return round(ties / len(results), 4)

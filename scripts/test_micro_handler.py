#!/usr/bin/env python3
"""Local end-to-end proof of the v3.0 MicroSessionDetectionSynapse handler.

Builds a valid schema-v4.1 MicroSessionDetectionSynapse (exactly as a validator
sends it), runs the shared contract validation, and scores it with the miner's
MicroSessionScorer — proving the synapse round-trips and the miner returns one
risk score per item, without needing a live validator query.

Run on the miner box (bittensor must be importable):
    python scripts/test_micro_handler.py
"""
from poker44.protocol import (
    MicroSessionDetectionSynapse,
    validate_micro_session_request,
)
from super_poker.micro_session_scorer import MicroSessionScorer


def make_item(item_id, actions, sizes, pressures):
    phases = ["preflop", "flop", "turn", "river"]
    return {
        "schema_version": "4.1",
        "item_id": item_id,
        "window_id": "test-window",
        "decisions": [
            {
                "decision_number": i + 1,
                "phase": phases[i],
                "position_group": "late",
                "pressure": pressures[i],
                "action_type": actions[i],
                "size_bucket": sizes[i],
                "is_all_in": actions[i] == "all_in",
            }
            for i in range(4)
        ],
    }


items = [
    make_item("aggr", ["raise", "bet", "raise", "all_in"],
              ["pot", "overbet", "pot", "all_in"],
              ["facing_bet", "no_call", "facing_bet", "no_call"]),
    make_item("pass", ["call", "check", "call", "check"],
              ["half_pot", "not_applicable", "half_pot", "not_applicable"],
              ["facing_bet", "no_call", "facing_bet", "no_call"]),
    make_item("fold", ["fold", "fold", "check", "call"],
              ["not_applicable", "not_applicable", "not_applicable", "half_pot"],
              ["facing_bet", "facing_bet", "no_call", "facing_bet"]),
]

# Build the synapse exactly as a validator would send it.
syn = MicroSessionDetectionSynapse(
    window_id="test-window",
    dataset_hash="a" * 64,
    query_id="q-test-1",
    items=items,
)

# 1. Shared contract validation (miners + validators use the same rules).
validate_micro_session_request(syn)
print("[1] validation: PASS")

# 2. Score exactly as forward_micro_sessions does.
scorer = MicroSessionScorer()
scores = scorer.score_items(syn.items)
syn.risk_scores = scores
syn.predictions = [s >= 0.5 for s in scores]
syn.model_version = scorer.model_version

# 3. Report + assert the response is well-formed.
print(f"[2] scorer version: {scorer.model_version}")
for item, score in zip(items, scores):
    print(f"      {item['item_id']:5s} -> {score:.3f}")
assert len(scores) == len(items), "score count != item count!"
assert all(0.0 <= s <= 1.0 for s in scores), "score out of [0,1]!"
assert any(s >= 0.5 for s in scores), "no item scored >= 0.5 (zero-gate risk)!"
print("[3] MICRO HANDLER OK — synapse round-trips, one valid score per item.")

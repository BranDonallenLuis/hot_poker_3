import math

from super_poker.dataset import Example
from super_poker.features import VISIBLE_BB_BUCKETS, _amount_bucket, chunk_features, hand_features
from super_poker.train import augment_live_size_chunks, matrix


def test_empty_features_are_stable_and_finite():
    assert set(chunk_features([])) == set(chunk_features([{}]))
    assert all(math.isfinite(value) for value in chunk_features([]).values())


def test_hero_behavior_is_separate_from_opponents():
    hand = {
        "metadata": {"hero_seat": 2, "max_seats": 6},
        "players": [{"seat": 1, "starting_stack": 2}, {"seat": 2, "starting_stack": 4}],
        "actions": [
            {"street": "preflop", "actor_seat": 1, "action_type": "raise", "amount": 1, "pot_before": 1},
            {"street": "preflop", "actor_seat": 2, "action_type": "raise", "amount": 3, "pot_before": 2},
            {"street": "flop", "actor_seat": 1, "action_type": "bet", "amount": 1, "pot_before": 4},
            {"street": "flop", "actor_seat": 2, "action_type": "fold", "amount": 0, "pot_before": 5},
        ],
    }
    values = hand_features(hand)
    assert values["three_bet"] == 1.0
    assert values["pfr"] == 1.0
    assert values["fold_to_aggression"] == 0.5
    assert values["hero_raise_count"] == 1.0
    assert values["hero_stack_vs_table"] > 1.0


def test_labels_and_ids_do_not_change_features():
    base = {"metadata": {"hero_seat": 1}, "actions": []}
    leaked = {**base, "hand_id": "special", "label": "bot", "is_bot": True}
    assert hand_features(base) == hand_features(leaked)


def test_training_matrix_uses_validator_visible_payload():
    hand = {
        "metadata": {"hero_seat": 2, "max_seats": 6, "bb": 5.0},
        "players": [{"seat": 2, "starting_stack": 500.0, "hole_cards": ["As", "Ah"]}],
        "streets": [{"street": "river", "board_cards": ["2c", "3d", "4h"]}],
        "actions": [],
        "outcome": {"showdown": True, "total_pot": 100.0},
    }
    example = Example([hand], 1, "2026-07-16", "train", "hash")

    frame, _ = matrix([example])

    assert frame.loc[0, "showdown_mean"] == 0.0
    assert frame.loc[0, "hero_stack_mean"] != 500.0


def test_amount_buckets_match_validator_grid():
    assert VISIBLE_BB_BUCKETS[_amount_bucket(1.47)] == 1.5
    assert VISIBLE_BB_BUCKETS[_amount_bucket(5.9)] == 6.0
    assert VISIBLE_BB_BUCKETS[_amount_bucket(55.0)] == 56.0


def test_chunk_features_include_hero_independent_signatures():
    chunk = [{"metadata": {"hero_seat": 6}, "actions": [
        {"actor_seat": 1, "action_type": "raise", "street": "preflop", "normalized_amount_bb": 3.1},
        {"actor_seat": 2, "action_type": "call", "street": "preflop", "normalized_amount_bb": 3.0},
    ]}] * 2
    values = chunk_features(chunk)
    assert values["actor_signature_top_share"] == 1.0
    assert values["street_signature_top_share"] == 1.0
    assert values["joint_signature_top_share"] == 1.0


def test_live_size_augmentation_stays_within_date_and_label():
    examples = [
        Example([{"i": i}] * 40, label, "2026-07-18", "train", f"{label}-{i}")
        for label in (0, 1) for i in range(3)
    ]
    augmented = augment_live_size_chunks(examples)
    assert len(augmented) == 6
    assert all(90 <= len(example.hands) <= 105 for example in augmented)
    assert all(example.source_date == "2026-07-18" for example in augmented)

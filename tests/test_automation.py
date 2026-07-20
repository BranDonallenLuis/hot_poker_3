import json
from datetime import datetime, timezone

from super_poker.automation import (
    CycleSchedule,
    Gates,
    assess_candidate,
    cycle_status,
    distribute_approved,
    promote,
)


def metadata(reward=0.88, ap=0.93, fpr=0.04, hard_fpr=0.04, folds=(0.85, 0.9)):
    return {
        "walk_forward_overall": {
            "reward": reward,
            "average_precision": ap,
            "fpr": fpr,
            "hard_fpr": hard_fpr,
        },
        "walk_forward": [{"reward": value} for value in folds],
    }


def test_candidate_must_pass_absolute_and_relative_gates():
    incumbent = metadata(reward=0.88, ap=0.93)
    assert assess_candidate(metadata(reward=0.881, ap=0.931), incumbent)["approved"]
    rejected = assess_candidate(metadata(reward=0.86, ap=0.90, fpr=0.08), incumbent)
    assert not rejected["approved"]
    assert "fpr_above_limit" in rejected["reasons"]
    assert "reward_regressed" in rejected["reasons"]


def test_unstable_fold_is_rejected():
    decision = assess_candidate(metadata(folds=(0.9, 0.2)), None, Gates())
    assert not decision["approved"]
    assert "unstable_or_missing_walk_forward_fold" in decision["reasons"]


def test_promotion_backs_up_model_and_metrics(tmp_path):
    candidate = tmp_path / "candidate.joblib"
    candidate.write_bytes(b"new")
    candidate.with_suffix(".metrics.json").write_text(json.dumps(metadata()), encoding="utf-8")
    incumbent = tmp_path / "hot_poker_3.joblib"
    incumbent.write_bytes(b"old")
    incumbent.with_suffix(".metrics.json").write_text(json.dumps(metadata(0.82)), encoding="utf-8")

    backup = promote(candidate, incumbent, tmp_path / "backups")

    assert backup is not None and backup.read_bytes() == b"old"
    assert incumbent.read_bytes() == b"new"
    assert json.loads(incumbent.with_suffix(".metrics.json").read_text())["walk_forward_overall"]["reward"] == 0.88


def test_only_approved_candidate_is_distributed(tmp_path):
    candidate = tmp_path / "candidate.joblib"
    candidate.write_bytes(b"excellent-model")
    payload = metadata()
    payload["model_version"] = "test-version"
    candidate.with_suffix(".metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    decision = assess_candidate(payload, None)

    approved = distribute_approved(candidate, tmp_path / "approved", decision)

    assert decision["approved"]
    assert approved.name == "hot_poker_3-test-version.joblib"
    assert approved.read_bytes() == b"excellent-model"
    assert approved.with_suffix(".metrics.json").is_file()
    assert approved.with_suffix(".approval.json").is_file()


def test_cycle_is_due_only_in_six_hour_pre_start_window():
    schedule = CycleSchedule(
        anchor_utc=datetime(2026, 7, 16, 12, tzinfo=timezone.utc),
        duration_hours=120,
        lead_hours=6,
    )
    assert not cycle_status(datetime(2026, 7, 16, 5, 59, tzinfo=timezone.utc), schedule)["due"]
    assert cycle_status(datetime(2026, 7, 16, 6, 0, tzinfo=timezone.utc), schedule)["due"]
    assert cycle_status(datetime(2026, 7, 16, 11, 59, tzinfo=timezone.utc), schedule)["due"]


def test_cycle_rolls_forward_exactly_120_hours():
    schedule = CycleSchedule(
        anchor_utc=datetime(2026, 7, 16, 12, tzinfo=timezone.utc),
        duration_hours=120,
        lead_hours=6,
    )
    status = cycle_status(datetime(2026, 7, 16, 12, tzinfo=timezone.utc), schedule)
    assert status["competition_start_utc"] == "2026-07-21T12:00:00+00:00"
    assert status["deployment_window_start_utc"] == "2026-07-21T06:00:00+00:00"

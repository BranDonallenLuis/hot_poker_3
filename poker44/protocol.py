"""Bittensor protocol shared by Poker44 validators and miner axons."""

from __future__ import annotations

from typing import Any, ClassVar

import bittensor as bt
from pydantic import ConfigDict, Field

SUPPORTED_PROTOCOL_VERSIONS = frozenset({"1", "2"})


class SessionDetectionSynapse(bt.Synapse):
    """Classify miner-visible subject sessions as human or bot.

    Validators populate ``sessions`` and retain ground truth locally. Miners
    return one probability-like risk score for every session. A score close to
    zero means human and a score close to one means bot.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    protocol_version: str = "2"
    window_id: str = ""
    dataset_hash: str = ""
    sessions: list[dict[str, Any]] = Field(default_factory=list)

    risk_scores: list[float] | None = None
    predictions: list[bool] | None = None
    model_version: str | None = None

    required_hash_fields: ClassVar[list[str]] = [
        "protocol_version",
        "window_id",
        "dataset_hash",
        "sessions",
    ]

    def deserialize(self) -> "SessionDetectionSynapse":
        return self


def validate_session_request(synapse: SessionDetectionSynapse) -> None:
    """Reject unsupported or incomplete requests before miner inference."""

    if synapse.protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(
            f"Unsupported Poker44 protocol version: {synapse.protocol_version}"
        )
    if not synapse.window_id.strip():
        raise ValueError("window_id is required")
    if synapse.protocol_version == "2":
        dataset_hash = synapse.dataset_hash.strip().lower()
        if len(dataset_hash) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_hash
        ):
            raise ValueError("dataset_hash must be a 64-character SHA-256 hex digest")


# Transitional import name for validators/miners that have not yet upgraded.
DetectionSynapse = SessionDetectionSynapse

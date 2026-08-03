"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import hashlib
import os
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse
from poker44.protocol import SessionDetectionSynapse, MicroSessionDetectionSynapse
from super_poker import live_capture
from super_poker.inference import SuperPokerModel
from super_poker.session_scorer import SessionScorer
from super_poker.micro_session_scorer import MicroSessionScorer


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self.model_path = Path(os.getenv("HOT_POKER_MODEL_PATH", repo_root / "artifacts" / "hot_poker_3.joblib"))
        self.predictor = None
        if self.model_path.is_file():
            try:
                self.predictor = SuperPokerModel(self.model_path)
            except Exception as exc:
                bt.logging.error(f"Unable to load {self.model_path}: {exc}")
        backend = "xgboost" if self.predictor is not None else "heuristic-fallback"
        artifact_sha256 = self._sha256_file(self.model_path) if self.predictor else ""
        bt.logging.info(f"Hot Poker 3 miner started | backend={backend}")
        metadata = self.predictor.metadata if self.predictor is not None else {}
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                repo_root / "super_poker" / "features.py",
                repo_root / "super_poker" / "inference.py",
                repo_root / "super_poker" / "live_capture.py",
                repo_root / "super_poker" / "sequence_model.py",
                repo_root / "super_poker" / "session_features_v3.py",
                repo_root / "super_poker" / "session_scorer.py",
                repo_root / "super_poker" / "micro_session_scorer.py",
                repo_root / "poker44" / "protocol.py",
                repo_root / "poker44" / "contracts" / "microsession.py",
                repo_root / "poker44" / "utils" / "encrypted_endpoints.py",
                repo_root / "super_poker" / "blend.py",
            ],
            defaults={
                "model_name": metadata.get("model_name", "hot-poker-3-fallback"),
                "model_version": metadata.get("model_version", "1"),
                "framework": metadata.get("framework", backend),
                "license": "MIT",
                "repo_url": os.getenv(
                    "HOT_POKER_REPO_URL",
                    "https://github.com/BranDonallenLuis/hot_poker_3",
                ),
                # Set POKER44_MODEL_ARTIFACT_URL after publishing this exact
                # artifact as a release asset.
                "artifact_url": "",
                "artifact_sha256": artifact_sha256,
                "model_card_url": (
                    "https://github.com/BranDonallenLuis/hot_poker_3/blob/main/MODEL_CARD.md"
                ),
                "notes": "Chronologically validated XGBoost behavioral detector with independent chunk scoring.",
                "open_source": True,
                "inference_mode": "local-joblib" if self.predictor else "heuristic",
                "training_data_statement": (
                    metadata.get("training_data", "Fallback uses runtime chunk features only.")
                ),
                "training_data_sources": ["poker44_public_benchmark"] if self.predictor else ["none"],
                "private_data_attestation": (
                    "No validator-private data or labels are used."
                ),
            },
        )
        if self.predictor is not None:
            self.model_manifest.update(
                {
                    "feature_schema_sha256": metadata.get("feature_schema_sha256", ""),
                    "validation_metrics": metadata.get("walk_forward_overall", {}),
                }
            )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

        # Poker44 v3.0: also serve SessionDetectionSynapse (sessions + telemetry).
        # Trained telemetry model if POKER44_V3_MODEL_PATH is set, else launch
        # heuristic. Attached alongside DetectionSynapse for the v2.0->v3.0 switch.
        self.session_scorer = SessionScorer()
        try:
            self.axon.attach(
                forward_fn=self.forward_session,
                blacklist_fn=self.blacklist_session,
                priority_fn=self.priority_session,
            )
            bt.logging.info(
                f"v3 SessionDetectionSynapse handler attached (scorer={self.session_scorer.model_version})"
            )
        except Exception as exc:
            bt.logging.warning(f"Could not attach v3 session handler: {exc}")

        # Poker44 v3.0 tournaments use MicroSessionDetectionSynapse (schema v4.1:
        # 4 strategic decisions, NO telemetry) — this is the LIVE v3.0 contract.
        self.micro_scorer = MicroSessionScorer()
        try:
            self.axon.attach(
                forward_fn=self.forward_micro_sessions,
                blacklist_fn=self.blacklist_micro_sessions,
                priority_fn=self.priority_micro_sessions,
            )
            bt.logging.info(
                f"v3 MicroSessionDetectionSynapse handler attached (scorer={self.micro_scorer.model_version})"
            )
        except Exception as exc:
            bt.logging.warning(f"Could not attach micro-session handler: {exc}")

        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        scores = (
            self.predictor.predict_chunk_scores(chunks)
            if self.predictor is not None
            else [self.score_chunk(chunk) for chunk in chunks]
        )
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Scored {len(chunks)} chunks with valid bot probabilities.")
        # Diagnostic capture of the live (unlabeled) input distribution. Disabled unless
        # POKER44_CAPTURE / POKER44_CAPTURE_BATCH are set; helpers swallow every error and
        # this call is wrapped again so capture can never affect the scored response.
        try:
            validator_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
            live_capture.capture(chunks, scores, self.uid, validator_hotkey)
            live_capture.capture_batch(chunks, scores, self.uid, validator_hotkey)
        except Exception:
            pass
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)

    # --- Poker44 v3.0 session handlers ---------------------------------------
    async def forward_session(self, synapse: SessionDetectionSynapse) -> SessionDetectionSynapse:
        """Return one bot-risk score per subject session (v3.0 tournaments)."""
        sessions = synapse.sessions or []
        scores = self.session_scorer.score_sessions(sessions)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_version = self.session_scorer.model_version
        bt.logging.info(
            f"v3 scored {len(sessions)} sessions "
            f"(above_0.5={sum(1 for s in scores if s >= 0.5) / len(scores):.3f})"
            if scores else "v3 received 0 sessions"
        )
        return synapse

    async def blacklist_session(self, synapse: SessionDetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority_session(self, synapse: SessionDetectionSynapse) -> float:
        return self.caller_priority(synapse)

    # --- Poker44 v3.0 micro-session handlers (schema v4.1, LIVE contract) -----
    async def forward_micro_sessions(
        self, synapse: MicroSessionDetectionSynapse
    ) -> MicroSessionDetectionSynapse:
        """Return one bot-risk score per v4.1 micro-session item. Robust: always
        emits exactly len(items) scores, never rejects a real query."""
        items = synapse.items or []
        try:
            scores = self.micro_scorer.score_items(items)
            if len(scores) != len(items):
                scores = [0.5] * len(items)
        except Exception as exc:
            bt.logging.error(f"micro-session scoring error: {exc}")
            scores = [0.5] * len(items)
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_version = self.micro_scorer.model_version
        bt.logging.info(
            f"v3 scored {len(items)} micro-session items window={synapse.window_id} "
            f"(above_0.5={sum(1 for s in scores if s >= 0.5) / len(scores):.3f})"
            if scores else "v3 micro received 0 items"
        )
        # Diagnostic capture of the live (unlabeled) v4.1 micro-session inputs.
        # Gated by POKER44_CAPTURE; fail-safe so it can never affect scoring.
        try:
            validator_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
            live_capture.capture_micro(
                items, scores, self.uid, validator_hotkey, synapse.window_id
            )
        except Exception:
            pass
        return synapse

    async def blacklist_micro_sessions(
        self, synapse: MicroSessionDetectionSynapse
    ) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority_micro_sessions(self, synapse: MicroSessionDetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)

# Hot Poker 3 Model Card

## Model identity

- Model: `hot-poker-3`
- Version: `20260720-161318`
- Framework: XGBoost with validator-stable behavioral signatures
- Artifact: `hot_poker_3.joblib`
- Artifact SHA-256: `d516935f0f05b5c99ca7168c5306f78ef593e2e105898c664bef5f6b366df44b`
- Feature schema: `super-poker-3.v4-r3-live-gap` (567 features)
- Inference: independent per-chunk probabilities

The initial artifact is derived from the approved Super Poker 3 implementation, but Hot
Poker 3 has separate model identity, deployment state, round history, hotkey, UID, port,
PM2 process, and future training lineage.

## Training data

The initial model uses 2,488 labeled examples from 54 Poker44 public benchmark releases
through 2026-07-18, plus 324 same-date, same-label live-size training augmentations. No
validator-private data or labels are used. Augmented examples are excluded from validation.

## Chronological validation

| Metric | Value |
| --- | ---: |
| Reward | 0.912191 |
| Average precision | 0.948505 |
| ROC AUC | 0.940054 |
| Bot recall | 0.767380 |
| Hard bot recall | 0.732620 |
| Hard false-positive rate | 0.032086 |
| Worst-fold reward | 0.902354 |

These public-data results are model-selection evidence, not a guarantee of live competition
performance. Each live score must be tied to the exact Hot Poker 3 artifact and deployment.

## Deployment safety

Run `python scripts/model/preflight.py` before every deployment. Production must not start
unless the Git identity, artifact metadata, runtime feature schema, and prediction smoke test
all pass. Keep this UID frozen for a complete evaluation round after deployment.

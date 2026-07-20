# Hot Poker 3

An open-source Poker44 SN126 miner using a regularized XGBoost bot detector.

## Design

- miner-visible, sanitization-aware behavior features only;
- hero-relative features plus robust all-table context;
- entropy, variability, quantiles, response behavior, and cross-hand signatures;
- chronological walk-forward evaluation on unseen release dates;
- deployment threshold learned from prior-date human out-of-fold scores;
- independent per-chunk probabilities with no top-k or prevalence forcing;
- current `poker44.score.scoring.reward` used for evaluation.

No performance on private live data is guaranteed. The saved metrics describe public
benchmark validation only.

## Initial Result

The initial artifact uses five chronological walk-forward folds through 2026-07-18.
Each fold trains only on earlier dates and learns its threshold from an earlier release.

| Metric | Result |
|---|---:|
| Poker44 reward | 0.9122 |
| Average precision | 0.9485 |
| ROC AUC | 0.9401 |
| Bot recall at constrained FPR | 0.7674 |
| Observed FPR | 0.0481 |
| Hard bot recall at 0.5 | 0.7326 |
| Hard human FPR at 0.5 | 0.0321 |

These results are model-selection evidence on public releases. They are not evidence
that the final all-data artifact has seen or will win on private validator batches.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

The trainer reads the public cache created by `Poker44-subnet/bot_detector/download.py`.
The default location is `../Poker44-subnet/data/raw`.

## Train

```bash
python -m super_poker.train \
  --data-dir ../Poker44-subnet/data/raw \
  --artifact artifacts/hot_poker_3.joblib
```

Training writes:

- `artifacts/hot_poker_3.joblib`: model, feature schema, threshold, metadata;
- `artifacts/hot_poker_3.metrics.json`: readable validation and provenance data.

## Evaluate

```bash
python -m super_poker.evaluate --dates 2026-07-12,2026-07-13
```

Do not treat evaluation on releases used for final training as an unseen test result. The
walk-forward metrics embedded during training are the honest model-selection signal.

## Run Miner

After training, use the standard Poker44 miner command or scripts. Override the artifact with:

```bash
export HOT_POKER_MODEL_PATH=/absolute/path/to/hot_poker_3.joblib
```

Before publishing, set the real public repository URL in `neurons/miner.py` so the model
manifest can meet transparent-miner policy.

Use a separate registered hotkey and axon port for this miner. Confirm with the Poker44
operator that one owner may run multiple competition UIDs before registering or deploying it.

Current deployment identity (public addresses only):

- wallet name: `hot_poker`
- coldkey SS58: `5DWwZ1DdMVqtQv3BN5Z7SFSnkvRpTprauS6VKiQJAHCDENhf`
- hotkey name: `hot-poker-3`
- hotkey SS58: `5EJLe5vs1uX1yxYk8Qs9J5p21VyurtukusCTrxeF73TdcQSB`
- default axon port: `7028`
- default PM2 process: `hot_poker_3`

## Automatic Learning

Automation uses `../Poker44-subnet/data/raw` by default, matching the training command.
That cache should contain the historical backfill created during initial setup. To initialize
an empty cache directly through this project, run once with an explicit directory:

```bash
python -m super_poker.automation daily --data-dir data/raw --backfill
```

Daily data-only update:

```bash
python -m super_poker.automation daily
```

Daily update plus a non-deployed candidate:

```bash
python -m super_poker.automation daily --train-candidate
```

Competition-cycle retraining and guarded deployment:

```bash
python -m super_poker.automation cycle
```

The cycle checker is anchored at `2026-07-16 12:00 UTC`, repeats every 120 hours, and
starts the guarded workflow six hours before the upcoming competition. It can be configured:

```bash
export HOT_POKER_CYCLE_ANCHOR_UTC=2026-07-16T12:00:00Z
export HOT_POKER_CYCLE_HOURS=120
export HOT_POKER_CYCLE_LEAD_HOURS=6
```

The cycle candidate is approved and promoted only when all excellent-performance checks pass:

- reward >= 0.85 and no more than 0.002 below the incumbent;
- average precision >= 0.92 and no more than 0.002 below the incumbent;
- pooled FPR <= 0.05;
- hard-threshold FPR <= 0.06;
- every walk-forward fold reward >= 0.80.

Rejected candidates remain under `artifacts/candidates/`. Approved candidates are copied to
`artifacts/approved/` with metrics and the gate decision. Successful deployment then copies
the incumbent to `artifacts/backups/` and atomically replaces the serving artifact.
The latest decision is recorded in `artifacts/automation-state.json`.

To schedule with cron while using the existing Poker44 environment:

```cron
15 2 * * * HOT_POKER_PYTHON=/home/achilles/Projects/Poker44-subnet/.venv/bin/python /home/achilles/Projects/hot-poker-3/scripts/model/daily_learning.sh --train-candidate
35 * * * * HOT_POKER_PYTHON=/home/achilles/Projects/Poker44-subnet/.venv/bin/python /home/achilles/Projects/hot-poker-3/scripts/model/cycle_learning.sh
```

The plain daily CLI only downloads and inspects by default. The installed cron command includes
`--train-candidate`, so it also trains and records a candidate without deploying it. Do not
schedule other training jobs during the six-hour pre-competition window. The hourly cycle check
is inexpensive outside that window and records each cycle completion so it cannot deploy twice.

## License

MIT. Poker44 reference code remains under its original MIT license.

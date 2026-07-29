#!/bin/bash

# Poker44 Miner Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-hot_poker}"
HOTKEY="${HOTKEY:-hot-poker-3}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-hot_poker_3}"
AXON_PORT="${AXON_PORT:-7028}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
MODEL_PATH="${HOT_POKER_MODEL_PATH:-$(pwd)/artifacts/hot_poker_3.joblib}"
PYTHON_BIN="${HOT_POKER_PYTHON:-python}"
REPO_COMMIT="${POKER44_MODEL_REPO_COMMIT:-$(git rev-parse HEAD 2>/dev/null)}"

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

if [[ ! "$REPO_COMMIT" =~ ^[0-9a-f]{7,40}$ ]]; then
    echo "Error: unable to resolve a valid Git commit for the model manifest"
    exit 1
fi

"$PYTHON_BIN" scripts/model/preflight.py \
  --artifact "$MODEL_PATH" \
  --expected-model "hot-poker-3-blend" \
  || {
    echo "Error: model preflight failed; existing PM2 process was not changed"
    exit 1
  }

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
export HOT_POKER_MODEL_PATH="$MODEL_PATH"
export POKER44_MODEL_REPO_COMMIT="$REPO_COMMIT"

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi

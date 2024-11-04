#!/usr/bin/env bash
set -euxo pipefail

# Run subnet
./scripts/localnet.sh &

(
  timeout 300s python3.11 neurons/validator.py \
    --netuid 1 \
    --subtensor.chain_endpoint ws://127.0.0.1:9946 \
    --wallet.name validator \
    --wallet.hotkey default \
    --logging.debug &

  timeout 300s python3.11 neurons/miner.py \
    --netuid 1 \
    --subtensor.chain_endpoint ws://127.0.0.1:9946 \
    --wallet.name miner \
    --wallet.hotkey default \
    --logging.error &

  # Wait for both background processes to finish or timeout
  wait
)

# todo: how to verify end result
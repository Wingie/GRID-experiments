#!/usr/bin/env bash
# Run a single self-play episode with the current model (needs a running game + RIMAPI).
#   scripts/run_episode.sh [experiment]
set -euo pipefail

EXPERIMENT="${1:-self_play}"
python -m rimworld_agent.training.self_play experiment="$EXPERIMENT" \
  self_play.iterations=1 self_play.episodes_per_iter=1

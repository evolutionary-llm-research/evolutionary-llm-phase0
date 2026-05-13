#!/bin/bash
set -e
PYTHONPATH='/mnt/e/github/Evolutionary LLM Research'
CONFIG="config/phase1b.yaml"
MANIFEST="data/v2/corpus_manifest_v3.json"
GENS=35
AGENTS=12
DOCS=30

for BIOME in savanna desert plain; do
  for SEED in 42 123 456; do
    OUTDIR="experiments/phase1b/${BIOME}_seed${SEED}"
    echo "=== Starting ${BIOME} seed=${SEED} ==="
    PYTHONPATH="$PYTHONPATH" python3 -m src.evolution.cli run \
      --biome $BIOME \
      --generations $GENS \
      --agents $AGENTS \
      --docs-per-agent $DOCS \
      --config $CONFIG \
      --output-dir $OUTDIR \
      --corpus-manifest $MANIFEST \
      --seed $SEED
    echo "=== Done ${BIOME} seed=${SEED} ==="
  done
done
echo "=== ALL PHASE 1b COMPLETE ==="

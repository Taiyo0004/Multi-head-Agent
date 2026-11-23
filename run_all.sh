#!/usr/bin/env bash
# Full pipeline: correctness -> training -> evaluation -> efficiency.
set -euo pipefail
python -m pytest tests -q
python -m mha_torch.verify
python -m mha_torch.train --train-size 2000 --batch-size 2 --grad-accum 4
python -m mha_torch.train_latent --train-size 1500
python -m mha_torch.evaluate --limit 200
python -m mha_torch.bench --tasks 8
echo "Done. Numbers are in results/torch/ and adapters/training_report.json"

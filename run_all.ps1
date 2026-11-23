# Full pipeline on Windows: correctness -> training -> evaluation -> efficiency.
# Usage:  powershell -ExecutionPolicy Bypass -File run_all.ps1
$ErrorActionPreference = "Stop"

python -m pytest tests -q
python -m mha_torch.verify
python -m mha_torch.train --train-size 2000 --batch-size 2 --grad-accum 4
python -m mha_torch.train_latent --train-size 1500
python -m mha_torch.evaluate --limit 200
python -m mha_torch.bench --tasks 8

Write-Host "`nDone. Numbers are in results\torch\ and adapters\training_report.json" -ForegroundColor Green

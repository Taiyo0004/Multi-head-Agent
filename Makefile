.PHONY: help install verify train train-latent evaluate bench test all clean

help:
	@echo "make install   - install dependencies"
	@echo "make test      - run the test suite (tiny random model, no download of weights)"
	@echo "make verify    - correctness checks against a real model"
	@echo "make train     - train one LoRA adapter per role on GSM8K"
	@echo "make train-latent - stage two: train the latent bridge"
	@echo "make evaluate  - GSM8K accuracy + cost for four configurations"
	@echo "make bench     - efficiency across the four execution modes"
	@echo "make all       - verify, train, evaluate, bench"

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest tests -q

verify:
	python -m mha_torch.verify

train:
	python -m mha_torch.train --train-size 2000

train-latent:
	python -m mha_torch.train_latent --train-size 1500

evaluate:
	python -m mha_torch.evaluate --limit 200

bench:
	python -m mha_torch.bench --tasks 8

all: verify train train-latent evaluate bench

clean:
	rm -rf results adapters
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

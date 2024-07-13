<div align="center">
  <h2><b> Trading-LLM: Forex Series Forecasting by Reprogramming Large Language Models </b></h2>
</div>

## Introduction

Trading-LLM is a reprogramming framework to repurpose LLMs for general time series forecasting with the backbone language models kept intact.
Notably, we show that time series analysis (e.g., forecasting) can be cast as yet another "language task" that can be effectively tackled by an off-the-shelf LLM.

## Requirements

- accelerate>=0.20.3
- einops>=0.7.0
- matplotlib>=3.7.0
- numpy>=1.23.5
- pandas>=1.5.3
- pandas_ta>=0.3.14b0
- scikit_learn>=1.2.2
- scipy>=1.5.4
- torch>=2.0.1
- tqdm>=4.65.0
- peft>=0.4.0
- transformers>=4.31.0
- deepspeed>=0.13.0
- bitsandbytes>=0.43.1
- datasets>=2.19.0
- sentencepiece>=0.2.0
- mpi4py>=3.1.6
- matplotlib
- torchmetrics
- sklearn

To install all dependencies:

```
pip install -r requirements.txt
```

## Quick Demos

1. Place datasets them under `./dataset`
2. If not done yet, go to `./data_provider` to add dataloader, look at `DatasetCustom`.
3. Tune the model. Write script to run the model in `./scripts`.
4. Run the script. For example, to run the model on AAPL dataset:

```bash
bash ./scripts/TradingLLM_AAPL.sh
```

5. The results will be saved in `./checkpoints`.

## Detailed usage

Please refer to `run_main.py` for the detailed description of each hyperparameter.

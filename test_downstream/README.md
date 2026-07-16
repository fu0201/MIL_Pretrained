# MIL Benchmark Code

This repository contains the downstream MIL evaluation code used for blind-review experiments.

Two evaluation modes are provided:

- `benchmark.py`: standard MIL fine-tuning from random initialization.
- `benchmark_pretrain.py`: MIL fine-tuning after loading a locally supplied distilled checkpoint through `--pretrained_weights`.

Distilled checkpoints and dataset files are intentionally not included in this repository. The scripts expect users to provide local paths through environment variables.

## Repository Layout

```text
benchmark.py                 # standard MIL benchmark
benchmark_pretrain.py        # benchmark with optional distilled initialization
benchmark_scratch.sh         # batch runner without distilled weights
benchmark_pretrain.sh        # batch runner with locally supplied distilled weights
downstream_task_jsons/       # example task split file with <PFM_NAME> placeholders
requirements.txt             # Python package requirements
```

## Requirements

```bash
pip install -r requirements.txt
```

Before running the benchmark, download and prepare the official MIL backbone implementations and any required official pretrained model weights for the selected `SLIDE_NAMES`. The evaluation scripts in this repository only provide the downstream benchmark workflow and do not redistribute third-party model weights.

## Running Standard MIL

```bash
DATA_ROOT=/path/to/datasets \
JOB_DIR=/path/to/results \
PFM_NAMES="conch_v1_5" \
SLIDE_NAMES="transmil" \
GPU_ID=0 \
bash benchmark_scratch.sh
```

## Running With Distilled Initialization

Weights are not distributed. To run the distilled-initialization benchmark, set either a shared checkpoint directory:

```bash
PRETRAINED_WEIGHTS_DIR=/path/to/distilled_checkpoints \
DATA_ROOT=/path/to/datasets \
JOB_DIR=/path/to/results_distill \
SLIDE_NAMES="dagmil" \
GPU_ID=0 \
bash benchmark_pretrain.sh
```

or provide a per-model checkpoint path:

```bash
PRETRAINED_WEIGHTS_DAGMIL=/path/to/local/dagmil.pt \
bash benchmark_pretrain.sh
```

The checkpoint should contain either a model state dict or a `student_state_dict` entry compatible with the selected slide encoder.

## Notes

- The included `bcnb_er.json` is an example task split file. Add additional task JSONs under `downstream_task_jsons/` and extend the shell scripts if needed.
- Dataset JSON entries use `<PFM_NAME>` as a placeholder and are resolved at runtime with `--pfm_name`.
- Generated results, local data, and model checkpoints should remain outside version control.
- The code depends on PyTorch, scikit-learn, NumPy, and tqdm.

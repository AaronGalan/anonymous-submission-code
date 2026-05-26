# Code for the Anonymous Submission

This repository contains the training and evaluation code for an anonymous submission.

The release documents the main training and testing pipeline used in our experiments, including the reinforcement-learning training entry point, the custom reward interface, and the accuracy evaluation script.

## Repository Structure

```text
kgrlvr/
|-- kgrl_reward.py
|-- compute_generation_acc.py
|-- utils/
|-- scripts/
`-- chat/
```

- `kgrl_reward.py`: custom reward implementation used by the RL trainer.
- `compute_generation_acc.py`: utility for computing answer accuracy from generated outputs.
- `utils/`: helper functions used by the training and evaluation pipeline.
- `scripts/`: shell scripts for training and checkpoint evaluation.
- `chat/`: prompt or chat-format related utilities.

## Environment

The experiments were run with the following environment:

```text
CUDA 12.4
PyTorch 2.6.0
verl 0.4.1
vLLM 0.8.4
```

Please install `verl` following the official installation instructions for the corresponding version. The scripts in this repository were tested with `verl==0.4.1` and `vllm==0.8.4`.

A typical environment setup may look as follows:

```bash
conda create -n kgrlvr python=3.10 -y
conda activate kgrlvr

# Install PyTorch and vLLM according to the local CUDA environment.
# The experiments used:
#   torch==2.6.0
#   vllm==0.8.4
#
# Please install verl following its official instructions.
```

Additional packages may be required depending on the local environment and data format.

## Expected Directory Layout

The provided scripts assume the following local directory structure:

```text
kgrlvr/
|-- data/
|   `-- MMedBench/
|       `-- <subset>/
|           |-- train-<language>.parquet
|           `-- test-<language>.parquet
|-- models/
|   `-- <model_name>/
|-- checkpoint/
|-- scripts/
|   |-- train_mmed_kgrl.sh
|   `-- test_checkpoint_mmedbench.sh
|-- kgrl_reward.py
`-- compute_generation_acc.py
```

The training and evaluation scripts expect prepared `.parquet` files under `data/MMedBench/<subset>/` and a local HuggingFace-style model directory under `models/<model_name>/`.

## Data Format

The scripts assume that each training or evaluation example has been converted into a record compatible with the `verl` data interface. A representative schema is shown below:

```python
{
    "data_source": str,
    "prompt": [
        {
            "role": str,
            "content": str
        }
    ],
    "ability": str,
    "reward_model": {
        "style": str,
        "ground_truth": str
    },
    "extra_info": {
        "split": str,
        "index": int,
        "answer": str,
        "question": str,
        "options": dict[str, str],
        "meta_info": str,
        "answer_idx": str,
        "rationale": str,
        "metamap_phrases": list[str],
        "human_checked": bool,
        "human_check_passed": bool,
        "extend_entities": list[str],
        "gold_entities": list[str]
    }
}
```

In this format, `gold_entities` corresponds to the high-relevance evidence group, while `extend_entities` corresponds to the low-relevance evidence group. These fields are used by the reward function during training.

## Training

To train a model on MMedBench, run:

```bash
bash scripts/train_mmed_kgrl.sh \
  --language=English \
  --subdir=2hop \
  --model=Qwen2.5-7B-Instruct \
  --kgrl_reward_mode=light_weighted_eweight0_alpha2_beta3 \
  --device=0,1,2,3 \
  --now=<experiment_timestamp>
```

Main arguments:

- `--language`: language subset used for training and validation.
- `--subdir`: data subset directory under `data/MMedBench/`.
- `--model`: local model directory name under `models/`.
- `--kgrl_reward_mode`: reward configuration used by `kgrl_reward.py`.
- `--device`: visible GPU IDs.
- `--now`: experiment identifier used for checkpoint and logging paths.

By default, checkpoints are saved under:

```text
checkpoint/MMedBench/<model_name>-<reward_mode>/<experiment_name>/
```

The training script uses `verl.trainer.main_ppo` with GRPO-style optimization and calls the custom reward function through:

```text
kgrl_reward.py::compute_score
```

Logging is configured in offline mode by default.

## Evaluation

To evaluate a trained checkpoint on MMedBench, run:

```bash
bash scripts/test_checkpoint_mmedbench.sh \
  --subset=2hop \
  --language=English \
  --model=Qwen2.5-7B-Instruct \
  --kgrl_reward_mode=light_weighted_eweight0_alpha2_beta3 \
  --ckpt=last \
  --now=<experiment_timestamp> \
  --devices=0,1,2,3
```

Main arguments:

- `--subset`: data subset directory under `data/MMedBench/`.
- `--language`: language subset used for evaluation.
- `--model`: local model directory name under `models/`.
- `--kgrl_reward_mode`: reward configuration associated with the evaluated checkpoint.
- `--ckpt`: checkpoint name. Use `last` to automatically resolve the latest checkpoint.
- `--now`: experiment identifier used to locate the checkpoint and output directory.
- `--devices`: visible GPU IDs.

The argument `--ckpt=last` automatically resolves the latest checkpoint recorded in the experiment directory. A specific checkpoint can also be provided, for example:

```bash
--ckpt=global_step_600
```

The script first runs generation with `verl.trainer.main_generation`, then computes accuracy using:

```bash
python compute_generation_acc.py \
  --subset=<subset> \
  --filename=<generation_file>.parquet
```

Generated outputs are saved under:

```text
data/MMedBench/<subset>/checkpoint/<model_name>-<reward_mode>/
```

## Notes

This anonymous release focuses on the training, reward, and evaluation components of the submitted work. It includes the reinforcement-learning training entry point, the custom KG-derived reward implementation, checkpoint evaluation scripts, and accuracy computation utilities.

Due to institutional restrictions and data redistribution constraints, the raw data preprocessing pipeline, original benchmark files, and intermediate KG retrieval artifacts are not included. The scripts assume that the required benchmark examples have already been converted into the data format described above.

The required model files should be prepared as local HuggingFace-style directories under `models/<model_name>/`. Logging is configured in offline mode by default.

This repository is provided solely for anonymous review.

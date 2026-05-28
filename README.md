# Robust LLM Unlearning Against Relearning Attacks

Official implementation for:

**Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter**

Zeguan Xiao, Xuanzhe Xu, Yun Chen, Yong Wang, Jian Yang, Yanqing Hu, Guanhua Chen

arXiv: [2605.11685](https://arxiv.org/abs/2605.11685)

## Overview

Large language model unlearning aims to remove the influence of selected data without retraining a model from scratch. A practical difficulty is that many unlearned models remain fragile: after a small relearning attack, the model can rapidly recover the supposedly forgotten knowledge.

This repository studies that failure mode from a representation-geometry perspective. The paper shows that existing unlearning methods mostly change dominant representation components, while minor components are left comparatively untouched. During relearning, dominant-component changes are easier to reverse; minor-component changes are more persistent. Based on this observation, we propose **Minor Component Unlearning (MCU)**, which explicitly pushes unlearning effects into minor representation components to improve robustness against relearning attacks.

The code supports experiments on:

- **WMDP-Bio**
- **WMDP-Cyber**
- **Years**

and reports:

- forget-set accuracy before and after unlearning,
- WikiText loss ratio,
- MMLU accuracy through `lm-evaluation-harness`,
- relearning robustness metrics after fine-tuning on the relearn split.

## Repository Structure

```text
.
+-- configs/                 # Hydra configs for the main experiments
+-- data/                    # Processed local evaluation/unlearning data
+-- src/
|   +-- main_runner.py       # Main unlearning + evaluation + relearning entry point
|   +-- relearn.py           # Relearning-only utilities
|   +-- data_processing/     # Dataset construction / preprocessing scripts
|   +-- utils/               # Losses, CIR, data loading, evaluation, training helpers
+-- build_env.sh             # Environment setup commands
+-- run_exps.sh              # Example batch launcher
+-- pyproject.toml
+-- requirements.txt
```

## Installation

The experiments are GPU-oriented and assume CUDA. The default configs use `meta-llama/Llama-3.1-8B` in bfloat16, so a modern NVIDIA GPU with enough memory is recommended.

```bash
git clone <this-repo-url>
cd unlearning-1

conda create -n unlearn python=3.11 -y
conda activate unlearn

pip install -r requirements.txt
pip install -e . --no-deps
pip install lm_eval==0.4.8
pip install tiktoken
```

You also need Hugging Face access for the base model used in the configs:

```bash
export HF_TOKEN=<your_huggingface_token>
```

The training script logs to Weights & Biases by default. Either configure W&B:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

or run locally/offline:

```bash
export WANDB_MODE=offline
```

## Data

The processed WMDP-Bio, WMDP-Cyber, Years, and WikiText files used by the main scripts are included under `data/`.

During execution, the code also downloads small retention-corpus slices from Hugging Face:

- `HuggingFaceFW/fineweb-edu`
- `m-a-p/FineFineWeb`

Make sure your environment can access the Hugging Face Hub.

## Reproducing Experiments

All main experiments use:

```bash
python src/main_runner.py --config-name=<config_name> --exp-num=<experiment_id>
```

Available configs:

| Config | Dataset | Default model |
| --- | --- | --- |
| `main_llama_bio` | WMDP-Bio | `meta-llama/Llama-3.1-8B` |
| `main_llama_cyber` | WMDP-Cyber | `meta-llama/Llama-3.1-8B` |
| `main_llama_years` | Years | `meta-llama/Llama-3.1-8B` |

Experiment IDs:

| ID | Method |
| --- | --- |
| `0` | MLP Breaking + CIR |
| `1` | MLP Breaking + CIR + MCU (`mlp_confuse_mcu` in code) |
| `2` | GradDiff / gradient ascent baseline |
| `3` | RMU + CIR |
| `4` | RMU + CIR + MCU (`rmu_mcu` in code) |

For example, to reproduce the WMDP-Cyber MCU run:

```bash
python src/main_runner.py --config-name=main_llama_cyber --exp-num=1
```

To run all five WMDP-Cyber experiments:

```bash
for i in 0 1 2 3 4; do
    python src/main_runner.py --config-name=main_llama_cyber --exp-num=$i
done
```

The included `run_exps.sh` is a template launcher for this pattern:

```bash
bash run_exps.sh
```

### Useful Overrides

Hydra overrides can be appended after the script arguments. Common examples:

```bash
# Use another compatible base model.
python src/main_runner.py --config-name=main_llama_bio --exp-num=1 \
    model_id=meta-llama/Llama-3.2-1B

# Skip the relearning stage.
python src/main_runner.py --config-name=main_llama_bio --exp-num=1 \
    skip_relearn=true

# Skip MMLU evaluation for a faster run.
python src/main_runner.py --config-name=main_llama_bio --exp-num=1 \
    skip_mmlu_eval=true

# Save the unlearned model.
python src/main_runner.py --config-name=main_llama_bio --exp-num=1 \
    save_model=true save_path=saved_models/bio_mcu
```


## Outputs

Each run logs metrics to W&B and appends a compact result row to:

```text
main_runner_results.csv
```

The most relevant fields are:

- `end_forget_acc_t1`: final forget-set accuracy after unlearning,
- `max_ret_acc_t1`: maximum relearned forget-set accuracy during relearning,
- `delta_acc_t1`: relearning gap, where lower is more robust,
- `wikitext_loss_ratio`: utility proxy relative to the original model,
- `mmlu_acc`: MMLU accuracy from `lm-evaluation-harness`.

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{xiao2026robustllmunlearningrelearning,
      title={Robust LLM Unlearning Against Relearning Attacks: The Minor Components in Representations Matter},
      author={Zeguan Xiao and Xuanzhe Xu and Yun Chen and Yong Wang and Jian Yang and Yanqing Hu and Guanhua Chen},
      year={2026},
      eprint={2605.11685},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.11685},
}
```

## Acknowledgements

This codebase builds on the public [CIR repository](https://github.com/filyp/unlearning) for **Collapse of Irrelevant Representations (CIR) Ensures Robust and Non-Disruptive LLM Unlearning** by Filip Sondej and Yushi Yang. We thank the authors for releasing their implementation.

## License

This project is released under the MIT License.

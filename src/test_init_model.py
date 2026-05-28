# %%
# usage:
# python src/main_runner.py --config-name=CONFIG_NAME --exp-num=NUM
# for example:
# python src/main_runner.py --config-name=cb --exp-num=3
#
# adapted from explorations/2_proj_training.py which also has some code for grad_proj,
#     dm grad and dm act, and the traditional dm,
#     mmlu evals, and per-token loss increase visualizations,
#     and context undisruption (a more fancy retaining technique)
# but here, we aim for more simlicity and dataset generality

# see main_runner_2025.08.18_configurable_control_set.py for:
#     jigsaw_threats implementation,
#     use_wikitext_as_retain option,
#     recording some batches and their per-token losses, to later see how they are disrupted
#    inspect per question acc

import argparse
import logging
import time
from collections import Counter
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch as pt
from datasets import Dataset, concatenate_datasets, load_dataset
from IPython import get_ipython
from omegaconf import DictConfig, OmegaConf, open_dict
from transformers import AutoTokenizer

import wandb
from utils import loss_fns
from utils.common_cir import *
from utils.common_cir import _get_projections
from utils.data_loading import *
from utils.evals import eval_on, lm_eval
from utils.git_and_reproducibility import get_conf_hash
from utils.loss_fns import cross_entropy, kl_loss
from utils.training import get_update_norm, scale_grads_, set_seeds, trainable_modules

# plt dark theme
plt.style.use("dark_background")

logging.basicConfig(level=logging.INFO)

# Parse just the config-name, let Hydra handle the rest
parser = argparse.ArgumentParser()
parser.add_argument("--config-name")
parser.add_argument("--exp-num", type=int)
parser.add_argument("--group-name", type=str, default=None)
args, remaining_args = parser.parse_known_args()

if get_ipython() is not None:
    args.config_name = "main_comparison_llama_bio"
    args.exp_num = 0
    remaining_args = ["model_id=meta-llama/Llama-3.2-1B"]  # locally we use only 1B

with hydra.initialize(config_path="../configs", version_base="1.2"):
    # Load base config without overrides first
    base_cfg = hydra.compose(config_name=args.config_name)
    cfg = OmegaConf.merge(
        base_cfg,
        base_cfg.experiment_list[args.exp_num],
        OmegaConf.from_dotlist(remaining_args),
    )
    tags = cfg.tags

# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
logging.info(f"Number of GPUs available: {num_gpus}")
device_main = pt.device("cuda")
device_storage = pt.device("cuda")

# * load model
if "gemma" in cfg.model_id:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main, attn_implementation='eager'
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main
    )
model.config.use_cache = False

lm_eval_results = lm_eval(
    model=model,
    tasks=["mmlu"],
    batch_size=2,
    num_fewshot=5,
)

logging.info(f"Initial lm eval results: {lm_eval_results['mmlu_acc']}")

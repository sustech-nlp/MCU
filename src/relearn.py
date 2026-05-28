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
    )
    # Use open_dict to allow adding new keys from command line
    with open_dict(cfg):
        cfg.merge_with(OmegaConf.from_dotlist(remaining_args))
    tags = cfg.tags

# ! setup
set_seeds(42)

num_gpus = pt.cuda.device_count()
logging.info(f"Number of GPUs available: {num_gpus}")
device_main = pt.device("cuda")
device_storage = pt.device("cuda")
# if num_gpus == 1:
#     pt.set_default_device("cuda")
#     device_main = pt.device("cuda")
#     device_storage = pt.device("cuda")
# elif num_gpus == 2:
#     pt.set_default_device("cuda:0")
#     device_main = pt.device("cuda:0")
#     device_storage = pt.device("cuda:1")

tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
tokenizer.pad_token = tokenizer.eos_token

# ! load wikitext batches
wikitext = load_local("wikitext_16k.jsonl")
wikitext_batches = [
    tokenizer(x["text"], **cfg.tokenizer)
    for x in wikitext.shuffle(seed=42).batch(cfg.wikitext_batch_size)
]


_corpus_version = "corpus_simple" if "simple" in cfg.dataset else "corpus"
is_dev = "dev_" if cfg.use_dev_split else ""
if "bio" in cfg.dataset:
    retain_set = load_fineweb_bio_corpus()
    T = load_local(f"wmdp_deduped_bio/{is_dev}T_{_corpus_version}.jsonl")
    V = load_local(f"wmdp_deduped_bio/{is_dev}V_{_corpus_version}.jsonl")

elif "cyber" in cfg.dataset:
    retain_set = load_fineweb_tech_corpus()
    T = load_local(f"wmdp_deduped_cyber/{is_dev}T_{_corpus_version}.jsonl")
    V = load_local(f"wmdp_deduped_cyber/{is_dev}V_{_corpus_version}.jsonl")

elif "years" in cfg.dataset:
    retain_set = load_fineweb_edu_corpus()
    T = load_local(f"dates_years/T.jsonl")
    V = load_local(f"dates_years/V.jsonl")

elif "mmlu" in cfg.dataset:
    retain_set = load_fineweb_edu_corpus()
    T = load_local(f"mmlu/T.jsonl")
    V = load_local(f"mmlu/V.jsonl")

else:
    raise ValueError(f"Unknown dataset: {cfg.dataset}")


T = T.filter(lambda x: x[cfg.model_id.split("/")[-1]] > 0.25)
V = V.filter(lambda x: x[cfg.model_id.split("/")[-1]] > 0.25)
T_and_V = concatenate_datasets([T, V])
eval_qs = T_and_V if cfg.get("eval_on_all_questions", False) else V
logging.info(f"{len(T)=}, {len(V)=}, {len(eval_qs)=}")

if "pairs" in cfg.dataset:
    only_ans = "only_ans" in cfg.dataset
    training_batches = load_batches_from_pairs_set(T_and_V, cfg, only_ans)
    retraining_batches = load_batches_from_pairs_set(T, cfg, only_ans)

elif "simple" in cfg.dataset:
    training_batches = load_batches_from_simple_set(T_and_V, cfg, cfg.train_batch_size)
    retraining_batches = load_batches_from_simple_set(T, cfg, cfg.train_batch_size)

elif "deebs" in cfg.dataset:
    deebs_corpus = load_local("wmdp_deduped_deebs_corpus.jsonl")
    t_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(T["question"]))
    v_txts = deebs_corpus.filter(lambda x: x["original_question"] in set(V["question"]))
    t_and_v_txts = concatenate_datasets([t_txts, v_txts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]

elif "years" in cfg.dataset:
    t_txts = load_local("dates_years/corpus_T.jsonl")
    v_txts = load_local("dates_years/corpus_V.jsonl")
    t_and_v_txts = concatenate_datasets([t_txts, v_txts])

    training_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_and_v_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]
    retraining_batches = [
        tokenizer(texts, **cfg.tokenizer)
        for texts in t_txts.shuffle(seed=42).batch(cfg.train_batch_size)["text"]
    ]

elif "mmlu" in cfg.dataset:
    training_batches = load_batches_from_simple_set(T_and_V, cfg, cfg.train_batch_size)
    retraining_batches = load_batches_from_simple_set(T, cfg, cfg.train_batch_size)

retain_batches = [
    tokenizer(x["text"], **cfg.tokenizer)
    for x in retain_set.shuffle(seed=42).batch(cfg.retain_batch_size)
    # .select(range(max(len(training_batches), cfg.num_eval_batches)))
]

recall_batches = load_recall_batches(eval_qs, cfg, batch_size=1)

print(f"Number of training batches: {len(training_batches)}")
print(f"Number of retain batches: {len(retain_batches)}")
print(f"Number of retraining batches: {len(retraining_batches)}")


def _get_loss(model, batches, use_answer_mask=False):
    loss_acc = 0
    for batch in batches:
        with pt.no_grad():
            output = model(**batch)
            if use_answer_mask:
                answer_mask = batch["answer_mask"]
                loss_acc += cross_entropy(output, batch, answer_mask=answer_mask).item()
            else:
                loss_acc += cross_entropy(output, batch).item()
    return loss_acc / len(batches)


def get_metrics(model):
    res = {}
    model.eval()

    # * eval forget acc
    res["forget_acc_t0"], res["forget_acc_t1"] = eval_on(eval_qs, model)

    nb = cfg.num_eval_batches
    res["wikitext_loss"] = _get_loss(model, wikitext_batches[:nb])
    res["retain_loss"] = _get_loss(model, retain_batches[:nb])
    res["recall_loss"] = _get_loss(model, recall_batches, use_answer_mask=True)

    logging.info(res)
    return res


# %% setup

# * load model
model = AutoModelForCausalLM.from_pretrained(
    cfg.unlearn_model_id, torch_dtype=pt.bfloat16, device_map=device_main
)
model.config.use_cache = False

retraining_optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retraining_rate)

# * set trainable params
mlp_modules = ["gate_proj", "up_proj", "down_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in mlp_modules)
    if p.requires_grad:
        logging.info(f"training {n}")

# * get metrics
res = get_metrics(model)

retraining_acc_t0, retraining_acc_t1 = [], []

for epoch in range(cfg.retraining_epochs):
    model.train()
    for batch in retraining_batches:
        pt.cuda.empty_cache()
        model.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = cross_entropy(output, batch)
        loss.backward()
        retraining_optimizer.step()

    # * get metrics
    res = get_metrics(model)
    retraining_acc_t0.append(res["forget_acc_t0"])
    retraining_acc_t1.append(res["forget_acc_t1"])

save_path = cfg.save_path
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
logging.info(f"Model saved to {save_path}")


# usage:
# python src/main_runner.py --config-name=CONFIG_NAME --exp-num=NUM
# for example:
# python src/main_runner.py --config-name=main_llama_bio --exp-num=0

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
    args.config_name = "main_llama_bio"
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

# * mask out the most common tokens
if cfg.mask_n_most_common_tokens is not None:
    # count the most common tokens in the retain set
    counter = Counter()
    for b in retain_batches:
        counter.update(b["input_ids"].flatten().tolist())
    nt = cfg.mask_n_most_common_tokens
    most_common_tokens = pt.tensor([t for t, _ in counter.most_common(nt)])
    # mask out the common tokens
    for b in training_batches:
        b["answer_mask"] = ~pt.isin(b["input_ids"], most_common_tokens)
        # AND with the attention mask, just in case
        b["answer_mask"] = b["answer_mask"] & b["attention_mask"].bool()

    coverage = sum(c for _, c in counter.most_common(nt)) / sum(counter.values())
    logging.info(f"coverage: {coverage:.2f}")


if cfg.get("retain_on_dev", False):
    if "bio" in cfg.dataset:
        dev_T = load_local(f"wmdp_deduped_bio/dev_T_{_corpus_version}.jsonl")
        dev_V = load_local(f"wmdp_deduped_bio/dev_V_{_corpus_version}.jsonl")
    elif "cyber" in cfg.dataset:
        dev_T = load_local(f"wmdp_deduped_cyber/dev_T_{_corpus_version}.jsonl")
        dev_V = load_local(f"wmdp_deduped_cyber/dev_V_{_corpus_version}.jsonl")
    dev_T_and_V = concatenate_datasets([dev_T, dev_V])

    retain_batches = load_batches_from_simple_set(dev_T_and_V, cfg, cfg.retain_batch_size)


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
if "gemma" in cfg.model_id:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=pt.bfloat16, device_map=device_main, attn_implementation='eager'
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=pt.bfloat16, device_map="auto"
    )
model.config.use_cache = False
all_layers = model.model.layers  # for trimmed model


# inspect per question acc
acc_list = []
for ex in eval_qs:
    acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
    acc_list.append(acc)
    logging.info(f"{acc=}, {ex['question']}")


# ! watch out - placing this optimization before retraining_optimizer is defined,
# causes it to bind only to the early layers!
# it's an unintended bug, but quite benign, since we only modified th early layers, retraining only them is not that bad
# but beware this anyway
if cfg.loss_fn_name in ["mlp_confuse", "mlp_confuse_mcu", "rmu", "rmu_mcu"]:  # trim the model
    max_layer = max(cfg.layer_range)
    model.model.layers = model.model.layers[: max_layer + 1]
    if cfg.get("cb_retaining_layers"):
        assert max(cfg.cb_retaining_layers) <= max_layer


# * set trainable params
logging.info(f"target_modules: {cfg.target_modules}")
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in cfg.target_modules)
    if p.requires_grad:
        logging.info(f"training {n}")

install_hooks(model)

unit_optimizer = pt.optim.SGD(model.parameters(), lr=1.0)
retraining_optimizer = pt.optim.SGD(model.parameters(), lr=cfg.retraining_rate)

# * cache the activations for circuit breaker retaining
retain_batches = retain_batches[: len(training_batches)]
if cfg.get("retaining_rate", 0) > 0 and "cb_retain" in cfg.retaining_loss_fns:
    for batch in retain_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        batch["retain_acts"] = {
            l_num: output.hidden_states[l_num].detach().to("cpu")
            for l_num in cfg.cb_retaining_layers
        }

# cache the activations for rmu retaining
if (cfg.get("retaining_rate", 0) > 0) and ("rmu_retain" in cfg.retaining_loss_fns):
    for batch in retain_batches:
        with pt.no_grad():
            output = model(**batch, output_hidden_states=True)
        batch["retain_acts"] = {
            l_num: output.hidden_states[l_num].detach().to("cpu")
            for l_num in cfg.rmu_retaining_layers
        }

if cfg.loss_fn_name in  ["rmu", "rmu_mcu"]:
    hidden_size = model.config.hidden_size
    # same random vector for all batches
    random_vector = pt.rand(1, hidden_size, dtype=pt.bfloat16, device=device_main)
    control_vec = random_vector / pt.norm(random_vector) * cfg.rmu_steering_coeff
    for batch in training_batches:
        batch["rmu_control_vec"] = control_vec.cpu()

    # * compute PCA
    if cfg.loss_fn_name == "rmu_mcu":
        all_layer_acts = {layer_id: [] for layer_id in range(*cfg.layer_range)}
        for batch in training_batches:
            with pt.no_grad():
                output = model(**batch, output_hidden_states=True)
            _mask = batch.get("answer_mask", batch["attention_mask"])
            _mask = _mask.bool().clone()
            _mask[:, : cfg.cut_off_tokens] = False
            for layer_id in range(*cfg.layer_range):
                full_act = output.hidden_states[layer_id].detach()
                _act = full_act[_mask]
                all_layer_acts[layer_id].append(_act.cpu())

        rmu_control_pcs = {}
        for layer_id, acts in all_layer_acts.items():
            all_acts = pt.cat(acts)
            rmu_control_pcs[layer_id] = _get_projections(all_acts, cfg.org_out_proj_num, cfg.cir_niter)
            del all_acts
        del all_layer_acts


if cfg.loss_fn_name in ["mlp_confuse", "mlp_confuse_mcu"]:
    # * install hooks for MLPs
    def save_output_hook(module, args, output):
        module.cached_out = output

    def stop_grad_hook(module, grad_input, grad_output):
        if grad_input[0] is None:
            # this happens on layer 0, with requires_grad=False on 1st MLP layer
            return
        # return [pt.zeros_like(grad_input[0])] + list(grad_input[1:])
        return [None] + list(grad_input[1:])

    for layer_id in range(*cfg.layer_range):
        model.model.layers[layer_id].mlp.register_forward_hook(save_output_hook)
        # if cfg.mlp_stop_grad:
        #     model.model.layers[layer_id].mlp.gate_proj.register_full_backward_hook(stop_grad_hook)
        #     model.model.layers[layer_id].mlp.up_proj.register_full_backward_hook(stop_grad_hook)

    # * cache the activations for MLP confusion
    for batch in training_batches:
        with pt.no_grad():
            output = model(**batch)
        _mask = batch.get("answer_mask", batch["attention_mask"])
        _mask = _mask.bool().clone()
        _mask[:, : cfg.cut_off_tokens] = False
        batch["org_mlp_out"] = {}
        batch["org_mlp_out_norm"] = {}
        for layer_id in range(*cfg.layer_range):
            mlp = model.model.layers[layer_id].mlp
            out = mlp.cached_out.detach()[_mask]
            batch["org_mlp_out"][layer_id] = out.cpu()
            batch["org_mlp_out_norm"][layer_id] = out.float().norm(dim=-1).mean().cpu()

    # * compute PCA
    if cfg.loss_fn_name == "mlp_confuse_mcu":
        org_mlp_out_pcs = {}
        
        for layer_id in range(*cfg.layer_range):
            mlp_outputs = [batch["org_mlp_out"][layer_id] for batch in training_batches]
            all_outputs = pt.cat(mlp_outputs)
            org_mlp_out_pcs[layer_id] = _get_projections(all_outputs, cfg.org_out_proj_num, cfg.cir_niter)
            del all_outputs

        for batch in training_batches:
            batch["org_mlp_out_projected"] = {}
            batch["org_mlp_out_projected_norm"] = {}
            for layer_id in range(*cfg.layer_range):
                out_projected = batch["org_mlp_out"][layer_id].clone().to(device_main)
                for comp in org_mlp_out_pcs[layer_id]:
                    out_projected -= project_out(out_projected, comp)
                batch["org_mlp_out_projected"][layer_id] = out_projected.cpu()
                batch["org_mlp_out_projected_norm"][layer_id] = out_projected.float().norm(dim=-1).mean().cpu()


if cfg.loss_fn_name == "npo":
    for batch in training_batches:
        with pt.no_grad():
            output = model(**batch)
            org_logits = output.logits.detach()
        batch["org_logits"] = org_logits.cpu().float()


# %%
# script name -> project
# config name & hash -> group
# experiment number -> name
project_name = "unlearning/" + Path(__file__).relative_to(repo_root()).as_posix()
project_name = project_name.replace("/", "|")
group = (
    args.group_name
    if args.group_name is not None
    else f"{args.config_name}_{get_conf_hash(args.config_name)}"
)
_args = "_".join(str(v) for v in cfg.experiment_list[args.exp_num].values())
run_name = f"{args.exp_num}|{_args}|{'_'.join(remaining_args)}"
wandb.init(
    project=project_name,
    group=group,
    name=run_name,
    tags=tags,
    config=OmegaConf.to_container(cfg),
)


model.model.layers = all_layers  # for trimmed model
init_res = get_metrics(model)
if cfg.loss_fn_name in ["mlp_confuse", "mlp_confuse_mcu", "rmu", "rmu_mcu"]:  # trim the model
    max_layer = max(cfg.layer_range)
    model.model.layers = model.model.layers[: max_layer + 1]

wandb.log(init_res)
assert cfg.algorithm in ["CIR", "GA"]

# % full training loop
start_time = time.time()
act_to_collapse = None
_retain_iter = 0
global_step = 0
acts_list = {n: [] for n, _ in trainable_modules(model)}
grads_list = {n: [] for n, _ in trainable_modules(model)}

for epoch in range(cfg.max_num_epochs):
    pt.cuda.empty_cache()

    # ! one epoch
    model.train()
    for b_num, batch in enumerate(training_batches):

        # ! unlearning loss
        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()
        answer_mask = batch.get("answer_mask", None)  # use answer_mask if it exists

        output = model(**batch, output_hidden_states=True)
        loss_fn = getattr(loss_fns, cfg.loss_fn_name)
        if cfg.loss_fn_name == "mlp_confuse":
            loss = loss_fn(model, batch, cfg, answer_mask)
        elif cfg.loss_fn_name == "mlp_confuse_mcu":
            loss = loss_fn(model, batch, cfg, org_mlp_out_pcs, answer_mask)
        elif cfg.loss_fn_name == "rmu_mcu":
            loss = loss_fn(output, batch, cfg, rmu_control_pcs, answer_mask)
        else:
            loss = loss_fn(output, batch, cfg, answer_mask)
        loss.backward()

        # ! here we modify the grad
        if cfg.algorithm == "CIR":
            for n, m in trainable_modules(model):
                if m.weight.grad is None:
                    continue
                acts = get_last_act(m, batch["attention_mask"], cfg.cut_off_tokens)
                grads = get_last_grad(m, batch["attention_mask"], cfg.cut_off_tokens)
                acts_list[n].append(acts.clone().to("cpu"))
                grads_list[n].append(grads.clone().to("cpu"))
                assert len(acts.shape) == len(grads.shape) == 2

                if act_to_collapse is None:
                    assert epoch == 0
                    continue

                # ! proj out the means and PCA components
                for comp in act_to_collapse[n]:
                    acts -= project_out(acts, comp)
                for comp in grad_to_collapse[n]:
                    grads -= project_out(grads, comp)

                # without the projections, this is the equivalent of normal backprop
                m.weight.grad = pt.einsum("ti,tj->ij", grads, acts)
                assert m.weight.grad.shape == m.weight.shape

            if act_to_collapse is None:
                assert epoch == 0
                continue

        if b_num == 0:
            stats = dict(
                update_norm=get_update_norm(model),
                act_norm=output.hidden_states[4].norm(dim=-1).mean(),
            )

        # * normalize grads
        norm = get_update_norm(model)
        for p in model.parameters():
            if p.grad is not None:
                p.grad *= cfg.max_norm / norm.to(p.grad.device)

        unit_optimizer.step()  # unit_optimizer has lr=1.0
        global_step += 1

        model.zero_grad(set_to_none=True)
        pt.cuda.empty_cache()

        if cfg.get("retaining_rate", 0) > 0:
            batch = retain_batches[_retain_iter % len(retain_batches)]
            _retain_iter += 1
            _mask = batch["attention_mask"].bool()

            output = model(**batch, output_hidden_states=True)

            loss = 0
            if "kl_loss" in cfg.retaining_loss_fns:
                loss += kl_loss(output, batch, model, _mask)
            if "cross_entropy" in cfg.retaining_loss_fns:
                loss += cross_entropy(output, batch)
            if "cb_retain" in cfg.retaining_loss_fns:
                loss += loss_fns.cb_retain(output, batch, cfg)
            if "rmu_retain" in cfg.retaining_loss_fns:
                loss += loss_fns.rmu_retain(output, batch, cfg)

            loss.backward()

            scale_grads_(model, cfg.retaining_rate)  # apply intended lr
            unit_optimizer.step()  # unit_optimizer has lr=1.0

        if cfg.algorithm == "CIR" and (cfg.pca_every_steps is not None and global_step % cfg.pca_every_steps == 0):
            for n, m in trainable_modules(model):
                if m.weight.grad is None:
                    continue
                pca_cache_size = cfg.pca_cache_size if cfg.get("pca_cache_size", None) is not None else len(training_batches)
                acts_list[n] = acts_list[n][-pca_cache_size:]
                grads_list[n] = grads_list[n][-pca_cache_size:]
            act_to_collapse = get_projections(acts_list, cfg.act_proj_num, cfg.cir_niter)
            grad_to_collapse = get_projections(grads_list, cfg.grad_proj_num, cfg.cir_niter)

    model.zero_grad(set_to_none=True)
    pt.cuda.empty_cache()

    if cfg.algorithm == "CIR" and (epoch == 0 or (cfg.pca_every_n is not None and epoch % cfg.pca_every_n == 0)):
        # ! calculate means and PCA components
        # _start_time = time.time()
        act_to_collapse = get_projections(acts_list, cfg.act_proj_num, cfg.cir_niter)
        grad_to_collapse = get_projections(grads_list, cfg.grad_proj_num, cfg.cir_niter)
        # logging.info(f"time taken to calculate PCA: {time.time() - _start_time:.2f}s")

        if cfg.pca_every_n is not None:
            acts_list = {n: [] for n, _ in trainable_modules(model)}
            grads_list = {n: [] for n, _ in trainable_modules(model)}

        if epoch == 0:
            continue  # no need to report metrics, because nothing has changed

    # ! get metrics
    model.model.layers = all_layers  # for trimmed model
    res = get_metrics(model)
    if cfg.loss_fn_name in ["mlp_confuse", "mlp_confuse_mcu", "rmu", "rmu_mcu"]:  # trim the model
        max_layer = max(cfg.layer_range)
        model.model.layers = model.model.layers[: max_layer + 1]

    wandb.log(res | stats)
    if (res["wikitext_loss"] > init_res["wikitext_loss"] * cfg.get("loss_budget", 1.01)):
        break

model.model.layers = all_layers  # for trimmed model

remove_all_hooks(model)

if cfg.get("save_model", False):
    save_path = cfg.get("save_path", repo_root() / "saved_models" / f"{run_name}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logging.info(f"Model saved to {save_path}")

lm_eval_results = {}
if not cfg.get("skip_mmlu_eval", False):
    lm_eval_results = lm_eval(
        model=model,
        tasks=["mmlu"],
        batch_size=2,
        num_fewshot=5,
    )

forget_acc_t0, forget_acc_t1 = res["forget_acc_t0"], res["forget_acc_t1"]
wikitext_loss_ratio = round(res["wikitext_loss"] / init_res["wikitext_loss"], 3)

unlearn_metrics = {
    "wikitext_loss_ratio": wikitext_loss_ratio,
}

unlearn_metrics = unlearn_metrics | lm_eval_results
wandb.log(unlearn_metrics)

# inspect per question acc
unlearn_acc_list = []
for ex in eval_qs:
    acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
    unlearn_acc_list.append(acc)
    logging.info(f"{acc=}, {ex['question']}")

wandb.finish()
logging.info(f"time taken: {time.time() - start_time:.2f}s")


# %% retraining on T

if cfg.get("skip_relearn", False):
    logging.info("Skipping relearn stage because skip_relearn=True")
    exit(0)

if "retraining_epochs" not in cfg:
    exit(0)

# * avoid compute gradient for all params and save memory
for param in model.parameters():
    param.requires_grad = False
if cfg.loss_fn_name in ["mlp_confuse", "mlp_confuse_mcu", "rmu", "rmu_mcu"]:  # trim the model
    model.model.layers = model.model.layers[: max_layer + 1]

# * set trainable params
mlp_modules = ["gate_proj", "up_proj", "down_proj"]
for n, p in model.named_parameters():
    p.requires_grad = any(pattern in n for pattern in mlp_modules)
    if p.requires_grad:
        logging.info(f"training {n}")

model.model.layers = all_layers  # for trimmed model

wandb.init(
    project="ret_" + project_name,
    group=group,
    name=run_name,
    tags=tags,
    config=OmegaConf.to_container(cfg),
)

# * get metrics
res = get_metrics(model)
wandb.log(res)

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
    wandb.log(res)
    retraining_acc_t0.append(res["forget_acc_t0"])
    retraining_acc_t1.append(res["forget_acc_t1"])

if "bio" in cfg.dataset or "cyber" in cfg.dataset:
    interval = 10
else:
    interval = 3
    
max_ret_acc_t0 = max(sum(retraining_acc_t0[i:i+interval]) / interval for i in range(0, len(retraining_acc_t0), interval))
max_ret_acc_t1 = max(sum(retraining_acc_t1[i:i+interval]) / interval for i in range(0, len(retraining_acc_t1), interval))

wandb.log({
    "max_ret_acc_t0": round(max_ret_acc_t0, 4) * 100,
    "max_ret_acc_t1": round(max_ret_acc_t1, 4) * 100,
    "△_acc_t0": round(max_ret_acc_t0 - forget_acc_t0, 4) * 100,
    "△_acc_t1": round(max_ret_acc_t1 - forget_acc_t1, 4) * 100,
})

# inspect per question acc
for a1, a2, ex in zip(acc_list, unlearn_acc_list, eval_qs):
    acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
    logging.info(f"{a1=}, {a2=}, {acc=}, {acc - a1=}, {acc - a2=}, {ex['question']}, choices={ex['choices']}, correct_answer={ex['choices'][ex['answer']]}")

wandb.finish()

import pandas as pd

results_df = pd.DataFrame([{
    "run_name": run_name,
    "tag_0": tags[0] if len(tags) > 0 else "",
    "tag_1": tags[1] if len(tags) > 1 else "",
    "mmlu_acc": lm_eval_results.get("mmlu_acc", None),
    "wikitext_loss_ratio": wikitext_loss_ratio,
    "end_forget_acc_t1": round(forget_acc_t1, 3) * 100,
    "max_ret_acc_t1": round(max_ret_acc_t1, 3) * 100,
    "delta_acc_t1": round(max_ret_acc_t1 - forget_acc_t1, 3) * 100,
}])

output_path = Path(__file__).parent.parent / f"{Path(__file__).stem}_results.csv"
if output_path.exists():
    existing_df = pd.read_csv(output_path)
    results_df = pd.concat([existing_df, results_df], ignore_index=True)

results_df.to_csv(output_path, index=False)
logging.info(f"Results saved to {output_path}")

"""Evaluations for WMDP, MMLU, WMDP-Deduped."""

# %%
from copy import deepcopy
import torch as pt
from transformers import AutoTokenizer

pt.set_default_device("cuda")
answer_tokens = [" A", " B", " C", " D"]


# %%
def format_prompt(ex):
    # taken from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wmdp/_default_template_yaml
    # "The following are multiple choice questions (with answers) about biology."
    return f"""\
{ex["question"].strip()}
A. {ex["choices"][0]}
B. {ex["choices"][1]}
C. {ex["choices"][2]}
D. {ex["choices"][3]}
Answer:"""


# # data augmentation - but it does not make acc much less noisy, so not using it
# def get_rotations(question):
#     for _ in range(4):
#         q_copy = deepcopy(question)
#         # rotate the possible answers
#         _tmp = q_copy["choices"].pop(0)
#         q_copy["choices"].append(_tmp)
#         q_copy["answer"] = (q_copy["answer"] - 1) % len(q_copy["choices"])
#         yield q_copy


# %%
def eval_on(dataset, model, batch_size=4, subset=None, temperature="both"):
    pt.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # note that this assumes start-of-sequence token is used (which is true for llama)
    # answer_ids = pt.tensor([tokenizer.encode(t)[1:] for t in answer_tokens]).reshape(4)
    # modify for compatibility with tokenizers that do not use or have a start-of-sequence token
    answer_ids = pt.tensor([tokenizer.encode(t)[-1] for t in answer_tokens]).reshape(4)

    # sort wmdp_bio by the prompt length
    dataset = sorted(dataset, key=lambda ex: len(format_prompt(ex)))
    if subset is not None:
        dataset = dataset[:subset]

    acc_t0 = 0
    acc_t1 = 0
    for i in range(0, len(dataset), batch_size):
        # print(i)
        batch = dataset[i : i + batch_size]
        batch_text = [format_prompt(ex) for ex in batch]

        input_dict = tokenizer(batch_text, return_tensors="pt", padding=True, padding_side="right")

        with pt.inference_mode():
            output = model(**input_dict)
        last_positions = input_dict["attention_mask"].sum(dim=-1) - 1
        last_token_logits = output.logits[range(len(batch)), last_positions]

        probs = pt.softmax(last_token_logits, dim=-1)
        answer_probs = probs[:, answer_ids]
        # if not all(answer_probs.sum(dim=-1) > 0.2):
            # raise ValueError("Sum of answer probs is too low")

        answer_probs /= answer_probs.sum(dim=-1, keepdim=True)  # normalize
        # assert pt.allclose(answer_probs.sum(dim=-1), pt.tensor(1.0, dtype=pt.bfloat16))
        _correct_answers = pt.tensor([ex["answer"] for ex in batch])

        # temperature=1
        correct_answer_probs = answer_probs[range(len(batch)), _correct_answers]
        acc_t1 += correct_answer_probs.sum().item()

        # for temperature=0
        hits = answer_probs.argmax(dim=-1) == _correct_answers
        acc_t0 += hits.sum().item()

        del answer_probs, probs, last_token_logits, output
        pt.cuda.empty_cache()

    acc_t0 /= len(dataset)
    acc_t1 /= len(dataset)

    if temperature == "both":
        return float(acc_t0), float(acc_t1)
    elif temperature == 1:
        return float(acc_t1)
    elif temperature == 0:
        return float(acc_t0)
    else:
        raise ValueError(f"Not supported temperature: {temperature}")
    
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval import simple_evaluate


def lm_eval(
    model,
    tasks,
    batch_size=8,
    num_fewshot=5,
    limit=None,
    device=None,
    log_samples=False,
    cache_requests=True,
    verbosity="INFO",
):
    """
    Evaluate a model using lm-evaluation-harness.
    
    Args:
        model: A HuggingFace model (transformers.PreTrainedModel)
        tasks: List of task names to evaluate, e.g. ["hellaswag", "arc_easy", "mmlu"]
        batch_size: Batch size for evaluation, can be int or "auto"
        num_fewshot: Number of few-shot examples (None uses task default)
        limit: Limit number of examples per task (useful for debugging)
        device: Device to use (None for auto-detect)
        log_samples: Whether to log individual samples
        cache_requests: Whether to cache requests (speeds up repeated evaluations)
        verbosity: Logging verbosity level
        
    Returns:
        dict: Dictionary with evaluation results
            - "results": Per-task metrics
            - "summary": Flattened metrics for easy logging
    """
    # Ensure model is in eval mode
    model.eval()
    
    # Wrap the model with HFLM for lm-eval compatibility
    lm = HFLM(pretrained=model, batch_size=batch_size)
    
    # Initialize task manager
    task_manager = TaskManager(verbosity=verbosity)
    
    # Build kwargs for simple_evaluate
    eval_kwargs = {
        "model": lm,
        "tasks": tasks if isinstance(tasks, list) else [tasks],
        "task_manager": task_manager,
        "log_samples": log_samples,
        "cache_requests": cache_requests,
    }
    
    if num_fewshot is not None:
        eval_kwargs["num_fewshot"] = num_fewshot
    if limit is not None:
        eval_kwargs["limit"] = limit
    if device is not None:
        eval_kwargs["device"] = device
    
    # Run evaluation
    results = simple_evaluate(**eval_kwargs)
    
    # Create a flattened summary for easy logging (e.g., to wandb)
    # summary = {}
    # for task_name, task_results in results["results"].items():
    #     for metric_name, value in task_results.items():
    #         if metric_name == "alias":
    #             continue
    #         clean_metric = metric_name.split(",")[0].strip()  # Strip ",none" suffix if present
    #         key = f"{task_name}/{clean_metric}"
    #         summary[key] = round(value * 100, 1)
    
    summary = {}
    for task in tasks:
        task_results = results["results"][task]
        for metric_name, value in task_results.items():
            if metric_name == "alias":
                continue
            clean_metric = metric_name.split(",")[0].strip()  # Strip ",none" suffix if present
            key = f"{task}_{clean_metric}"
            summary[key] = round(value * 100, 1)

    return summary

# %%
import os
import re
import torch as pt
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root

main_device = pt.device("cuda:0")
base_path = repo_root() / "data/mmlu_cats_random_trimmed"

def load_local_set(paths):
    return load_dataset(
        "json", data_files=[str(base_path / path) for path in paths], split="train"
    )

def process_category(cat, split_name):
    ds = load_local_set([f"mmlu_{cat}.jsonl"])
    corpus_ds = load_local_set([f"corpus_mmlu_{cat}.jsonl"])

    q_to_texts = {}
    for x in corpus_ds:
        q = x["original_question"]
        if q not in q_to_texts:
            q_to_texts[q] = []
        q_to_texts[q].append(x["text"])
    
    ds = ds.map(lambda x: {"s": split_name, "category": cat, "sentences": q_to_texts[x["question"]]})
    return ds

def evaluate_and_update(dataset, model_path, column_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=pt.bfloat16, device_map=main_device
    )

    question_to_acc = {}
    for ex in dataset:
        print(ex["question"])
        acc = eval_on(Dataset.from_list([ex]), model, temperature=1)
        print(acc)
        question_to_acc[ex["question"]] = acc

    # update the questions with the accuracy
    updated_dataset = dataset.map(lambda ex: {column_name: question_to_acc[ex["question"]]})
    del model
    return updated_dataset

# %%
mmlu_cats_forget = ["STEM", "business", "chemistry", "culture", "geography"]
mmlu_cats_retain = ["health", "history", "law", "philosophy", "social sciences"]

# T: STEM, business, chemistry, culture
T_cats = [c for c in mmlu_cats_forget if c != "geography"]
# V: geography
V_cat = "geography"

# %%
# Load T
questions_T_list = [process_category(cat, "T") for cat in T_cats]
questions_T = concatenate_datasets(questions_T_list)

# Load V
questions_V = process_category(V_cat, "V")

# Load Retain
questions_retain_list = [process_category(cat, "retain") for cat in mmlu_cats_retain]
questions_retain = concatenate_datasets(questions_retain_list)

questions = concatenate_datasets([questions_T, questions_V, questions_retain])
print(f"{len(questions)=}")

# %%
# filter out all negated questions
def is_not_a_not_question(q):
    text = q["question"].lower()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    last_sentence = sentences[-1]
    return "not" not in last_sentence

questions = questions.filter(is_not_a_not_question)
print(f"{len(questions)=}")

# %%
# filter out all "none of the above" questions
def is_not_a_none_of_the_above_question(q):
    correct_ans = q["choices"][q["answer"]].lower()
    if "none" in correct_ans and ("above" in correct_ans or "choices" in correct_ans):
        return False
    return True

questions = questions.filter(is_not_a_none_of_the_above_question)
print(f"{len(questions)=}")

# # %% Llama-3.1-8B accuracies eval
# questions = evaluate_and_update(
#     questions, 
#     "meta-llama/Llama-3.1-8B", 
#     "Llama-3.1-8B"
# )

# # %% Qwen3-8B-Base accuracies eval
# questions = evaluate_and_update(
#     questions, 
#     "Qwen/Qwen3-8B-Base", 
#     "Qwen3-8B-Base"
# )

# %%
output_dir = repo_root() / "data/mmlu"
output_dir.mkdir(exist_ok=True)

# for split in ["T", "V", "retain"]:
#     sub_q = questions.filter(
#         lambda ex: (ex["s"] == split)
#     )
#     sub_q = sub_q.remove_columns(["s"])
#     print(output_dir / f"{split}.jsonl", len(sub_q))
#     sub_q.to_json(output_dir / f"{split}.jsonl")


corpus = load_local_set(
    [f"corpus_mmlu_{cat}.jsonl" for cat in mmlu_cats_forget + mmlu_cats_retain]
)
corpus.to_json(output_dir / "corpus_mmlu.jsonl")

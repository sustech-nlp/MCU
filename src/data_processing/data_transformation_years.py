# %%
import torch as pt
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM
from utils.evals import eval_on
from utils.git_and_reproducibility import repo_root

main_device = pt.device("cuda:0")
base_path = repo_root() / "data/dates-years-trimmed"

def load_local_set(paths):
    return load_dataset(
        "json", data_files=[str(base_path / path) for path in paths], split="train"
    )


def evaluate_and_update(dataset, model_path, column_name):
    """Run per-question evals with a model and append the accuracy column."""
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
questions_T = load_local_set([
    "split_0.jsonl",
    "split_1.jsonl",
    "split_2.jsonl",
    "split_3.jsonl",
])
questions_V = load_local_set(["split_4.jsonl"])

questions_V = questions_V.map(lambda x: {"s": "V"})
questions_T = questions_T.map(lambda x: {"s": "T"})

questions = concatenate_datasets([questions_T, questions_V])
print(f"{len(questions)=}")


# %% Llama-3.1-8B accuracies eval
questions = evaluate_and_update(
    questions, 
    "meta-llama/Llama-3.1-8B", 
    "Llama-3.1-8B"
)

# %% Qwen3-8B-Base accuracies eval
questions = evaluate_and_update(
    questions, 
    "Qwen/Qwen3-8B-Base", 
    "Qwen3-8B-Base"
)

# %%
output_dir = repo_root() / "data/dates_years"
output_dir.mkdir(exist_ok=True)

for split in ["T", "V"]:
    sub_q = questions.filter(
        lambda ex: (ex["s"] == split)
    )
    sub_q = sub_q.remove_columns(["s"])
    print(output_dir / f"{split}.jsonl", len(sub_q))
    sub_q.to_json(output_dir / f"{split}.jsonl")

# %%
corpus_T = load_local_set([
    "corpus_split_0.jsonl",
    "corpus_split_1.jsonl",
    "corpus_split_2.jsonl",
    "corpus_split_3.jsonl",
])
corpus_V = load_local_set(["corpus_split_4.jsonl"])

corpus_T.to_json(output_dir / "corpus_T.jsonl")
corpus_V.to_json(output_dir / "corpus_V.jsonl")

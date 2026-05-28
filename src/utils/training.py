import random

import numpy as np
import torch as pt
import torch.nn.functional as F
from tensordict import TensorDict
from transformers import set_seed as set_transformers_seed

from utils import loss_fns


# --- Setup and Environment ---
def set_seeds(seed):
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    set_transformers_seed(seed)
    # pt.use_deterministic_algorithms(True)


def relearn(model, relearn_batches, conf, eval_callback):
    # relearning
    set_seeds(42)
    optimizer = pt.optim.SGD(model.parameters(), lr=conf.lr)
    for p in model.parameters():
        p.requires_grad = True
    num_of_loops = int(len(relearn_batches) * conf.epochs)
    for loop_num in range(num_of_loops):
        pt.cuda.empty_cache()
        batch_index = loop_num % len(relearn_batches)
        batch = relearn_batches[batch_index]

        if batch_index == 0:
            eval_callback(model)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(**batch)
        loss = loss_fns.cross_entropy(output, batch)
        loss.backward()
        optimizer.step()

    return model
def trainable_modules(model):
    return [
        (n, m)
        for n, m in model.named_modules()
        if "_proj" in n and m.weight.requires_grad
    ]


def scale_grads_(model, factor: float):
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= factor
def prepare_answer_mask(beginning_batch, full_batch):
    long_attn = full_batch["attention_mask"]
    short_attn = beginning_batch["attention_mask"]
    pad_amount = long_attn.shape[1] - short_attn.shape[1]
    short_attn_padded = F.pad(short_attn, (0, pad_amount), value=0)
    answer_mask = (long_attn != short_attn_padded).to(pt.int64)
    return answer_mask


def PCA_gpu(v, n_components=10, center=True):
    # Center the data
    if center:
        v = v - v.mean(axis=0)
    # Compute covariance matrix
    cov = (v.T @ v) / (v.shape[0] - 1)
    # Compute eigenvalues and eigenvectors
    # * pt.linalg.eigh seems to leak memory!!
    eigenvalues, eigenvectors = pt.linalg.eigh(cov)
    # Sort in descending order
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    # Get the top n_components
    return eigenvectors.T[:n_components]
def get_update_norm(model):
    """L2 norm of weight.grad, computed across all the trainable weights."""
    first_device = next(model.parameters()).device
    return (
        sum(
            m.weight.grad.to(pt.float32).norm().to(first_device) ** 2
            for _, m in trainable_modules(model)
            if m.weight.grad is not None
        )
        ** 0.5
    )


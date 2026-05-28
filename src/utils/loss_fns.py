import logging
import torch as pt
import torch.nn.functional as F

from utils.common_cir import project_out

# new_logits = []
# new_target_probs = []
# for id_, target_prob, logit in zip(ids, target_probs, logits):
#     mask = pt.ones_like(logit, dtype=pt.bool)
#     mask[id_] = False
#     new_logits.append(logit[mask])
#     new_target_probs.append(target_prob[mask])
# new_logits = pt.stack(new_logits)
# new_target_probs = pt.stack(new_target_probs)


def _normalize_logits(logits):
    """Shift the raw logits to make them logs of probabilities summing to one."""
    return logits - logits.exp().sum(dim=-1, keepdim=True).log()


def kl_loss(output, batch, model, mask):
    mask = mask.bool()
    logits = output.logits[mask]
    # we store acts and recalculate logits to save memory
    original_last_act = batch["original_last_act"].to("cuda")[mask]
    original_logits = (model.model.embed_tokens.weight @ original_last_act.T).T

    logits = _normalize_logits(logits.float())
    original_logits = _normalize_logits(original_logits.float())

    # calculate KL divergence between original and current logits
    kl_div = F.kl_div(logits, original_logits, reduction="batchmean", log_target=True)
    assert kl_div > -1e-6  # it can be slightly negative due to numerical errors
    # this kl_div calculation is the same as:
    # (original_logits.exp() * (original_logits - logits)).sum(dim=-1).mean()
    return kl_div


def cross_entropy(output, batch, cfg=None, answer_mask=None):
    shifted_logits = output.logits[:, :-1, :].float()
    shifted_ids = batch["input_ids"][:, 1:]

    # mask out the beginning tokens
    if answer_mask is not None:
        # note, that answer_mask is a subset of attention mask
        assert pt.all(batch["attention_mask"] * answer_mask == answer_mask)
        mask = answer_mask
    else:
        mask = batch["attention_mask"]
    mask = mask[:, 1:].bool()

    return pt.nn.functional.cross_entropy(
        shifted_logits[mask],
        shifted_ids[mask],
        # reduction="sum",
    )
def neg_cross_entropy(output, batch, cfg=None, answer_mask=None):
    return -cross_entropy(output, batch, cfg, answer_mask)


def npo(output, batch, cfg, answer_mask=None,):
    shifted_logits = output.logits[:, :-1, :].float()
    shifted_org_logits = batch["org_logits"][:, :-1, :].to(shifted_logits.device).float()
    shifted_ids = batch["input_ids"][:, 1:]

    # mask out the beginning tokens
    if answer_mask is not None:
        # note, that answer_mask is a subset of attention mask
        assert pt.all(batch["attention_mask"] * answer_mask == answer_mask)
        mask = answer_mask
    else:
        mask = batch["attention_mask"]
    mask = mask[:, 1:].bool()

    logits = shifted_logits[mask]
    org_logits = shifted_org_logits[mask]
    labels = shifted_ids[mask]

    log_probs = F.log_softmax(logits, dim=-1)
    current_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    org_log_probs = F.log_softmax(org_logits, dim=-1)
    ref_log_probs = org_log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    log_ratio = current_log_probs - ref_log_probs
    loss = -F.logsigmoid(-cfg.npo_beta * log_ratio).mean()
    return loss


def cb_retain(output, batch, cfg, answer_mask=None):
    assert answer_mask is None
    _mask = batch["attention_mask"].bool().clone()
    # _mask[:, :cfg.cut_off_tokens] = False  # do not do it! retain everywhere!
    
    loss_acc = 0
    for layer_id in cfg.cb_retaining_layers:
        acts = output.hidden_states[layer_id][_mask].float()
        org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
        assert acts.shape == org_acts.shape
        assert len(acts.shape) == 2

        avg_act_norm = org_acts.norm(dim=-1).mean()
        dist = (acts - org_acts).norm(dim=-1).mean() / avg_act_norm

        loss_acc += dist ** cfg.cb_retaining_pow

    return loss_acc / len(cfg.cb_retaining_layers)


def rmu(output, batch, cfg, answer_mask=None):
    _mask = answer_mask if answer_mask is not None else batch["attention_mask"]
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        acts = output.hidden_states[layer_id][_mask].float()
        control_vec = batch["rmu_control_vec"].to(acts.device).float()
        # acts: (N, D), control_vec: (1, D) or (D,)
        loss_acc += pt.nn.functional.mse_loss(acts, control_vec.expand_as(acts))

    return loss_acc / len(range(*cfg.layer_range))

def rmu_mcu(output, batch, cfg, rmu_control_pcs, answer_mask=None):
    _mask = answer_mask if answer_mask is not None else batch["attention_mask"]
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        acts = output.hidden_states[layer_id][_mask].float()
        control_vec = batch["rmu_control_vec"].to(acts.device).float()

        for comp in rmu_control_pcs[layer_id]:
            acts -= project_out(acts, comp)

        loss_acc += pt.nn.functional.mse_loss(acts, control_vec.expand_as(acts))

    return loss_acc / len(range(*cfg.layer_range))


def rmu_retain(output, batch, cfg, answer_mask=None):
    assert answer_mask is None
    _mask = batch["attention_mask"].bool().clone()

    loss_acc = 0
    for layer_id in cfg.rmu_retaining_layers:
        acts = output.hidden_states[layer_id][_mask].float()
        org_acts = batch["retain_acts"][layer_id].to(acts.device)[_mask].float()
        loss_acc += pt.nn.functional.mse_loss(acts, org_acts)

    return loss_acc / len(cfg.rmu_retaining_layers)


def mlp_confuse(model, batch, cfg, answer_mask=None):
    _mask = answer_mask if answer_mask is not None else batch["attention_mask"]
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        out = model.model.layers[layer_id].mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        org_norm = batch["org_mlp_out_norm"][layer_id].to(out.device)
        dotproducts = pt.einsum("ts,ts->t", out, org_out)
        dotproducts = dotproducts / org_norm ** 2
        # logging.debug(dotproducts)
        loss_acc += dotproducts.clip(min=cfg.mlp_floor).mean()
        # used to also do max=1, but that's catastrophic - stops unlearning but not disruption

    return loss_acc / len(range(*cfg.layer_range))


def mlp_confuse_mcu(model, batch, cfg, mlp_out_pcs, answer_mask=None):
    _mask = answer_mask if answer_mask is not None else batch["attention_mask"]
    _mask = _mask.bool().clone()
    _mask[:, :cfg.cut_off_tokens] = False

    loss_acc = 0
    for layer_id in range(*cfg.layer_range):
        out = model.model.layers[layer_id].mlp.cached_out
        out = out[_mask].float()
        org_out = batch["org_mlp_out"][layer_id].to(out.device).float()
        org_out_projected = batch["org_mlp_out_projected"][layer_id].to(out.device).float()
        assert out.shape == org_out.shape
        assert len(out.shape) == 2

        for comp in mlp_out_pcs[layer_id]:
            out -= project_out(out, comp)

        org_out_projected_norm = batch["org_mlp_out_projected_norm"][layer_id].to(out.device)
        dotproducts = pt.einsum("ts,ts->t", out, org_out_projected)
        dotproducts = dotproducts / org_out_projected_norm ** 2
        loss_acc += dotproducts.clip(min=cfg.mlp_floor).mean()

    return loss_acc / len(range(*cfg.layer_range))

"""The guided objective on the world-model condition (loss.name=bridge_cbm): sbm_rae.guided_cbm unchanged, with the
class label replaced by the world model's condition dict {"obs": history latents, "act": actions, "null": (B,) bool}.

The null class of ImageNet's label dropout becomes the null condition: the map sees the learned null action sequence
(sbm_rae.models.bridge_unet) and learns the action-marginal prediction; the guide's teacher reads b(y) - b(null) with that
branch, exactly as on ImageNet. The history latents are never dropped (the prediction is always conditioned on the past).
"""
import torch

from sbm_rae.guided_cbm import GuidedCBMLoss


def _batch_of(cond):
    return cond["act"].shape[0]


class BridgeCBMLoss(GuidedCBMLoss):
    def __init__(self, cfg):
        super().__init__(cfg, label_dim=1)

    def _cast_labels(self, labels):
        d = dict(labels)
        if d.get("null", None) is None:
            d["null"] = torch.zeros(_batch_of(d), dtype=torch.bool, device=d["act"].device)
        return d

    def _drop_labels(self, labels, drop):
        d = dict(labels)
        d["null"] = d["null"] | drop.to(d["null"].device)
        return d

    def _null_mask(self, labels):
        return labels["null"]

    def _null_labels(self, labels):
        d = dict(labels)
        d["null"] = torch.ones_like(d["null"])
        return d


def get_bridge_cbm_loss_fn(cfg):
    return BridgeCBMLoss(cfg.loss.guided_cbm)

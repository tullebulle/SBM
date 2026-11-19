"""Text conditioning for the text-to-image maps.

The loss addresses conditioning through one index per example (a class id on ImageNet) and one null index. `TextBound`
keeps that contract for text: it binds a batch's text rows (the frozen encoder's token states and token masks) to a
map, so `bound(x, rows, omega=...)` reads row i's prompt and the null index reads the null prompt (every token
replaced by the map's mask token, as MiniT2I's prompt dropout does). The loss, its guide and its teacher then run
unchanged. `TextEncoder` is the frozen FLAN-T5 encoder; like `Frozen` (an external guidance teacher) it is held by
reference, outside the module tree: not trained, not checkpointed, and shared rather than copied by the EMA copies.
"""
from __future__ import annotations

import torch


class Frozen:
    """A frozen network held by reference; moved to the caller's device at first use."""

    def __init__(self, module):
        self.module = module.eval().requires_grad_(False)

    def __deepcopy__(self, memo):
        return self

    def on(self, device):
        if next(self.module.parameters()).device != device:
            self.module.to(device)
        return self.module


class TextEncoder(Frozen):
    """MiniT2I's text side: FLAN-T5-Large encoder states of the caption padded to `max_len` tokens, fp32."""

    def __init__(self, path, max_len=256):
        from transformers import T5EncoderModel
        super().__init__(T5EncoderModel.from_pretrained(str(path), torch_dtype=torch.float32))
        self.path, self.max_len = str(path), int(max_len)

    @torch.no_grad()
    def __call__(self, ids, mask):
        """Token ids / mask [B, L] -> (token states [B, L, D] fp32, mask [B, L] float)."""
        with torch.autocast(ids.device.type, enabled=False):
            states = self.on(ids.device)(input_ids=ids, attention_mask=mask).last_hidden_state
        return states.float(), mask.float()


def tokenizer(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path))


def tokenize(tok, captions, max_len=256):
    """-> (ids, mask) int64 [B, max_len], padded and truncated as MiniT2I's pipeline does."""
    out = tok(list(captions), return_tensors="pt", padding="max_length", truncation=True, max_length=int(max_len))
    return out.input_ids, out.attention_mask


class TextBound:
    """`model` with one batch of text bound to it, called like the class-conditional maps: y holds a row index per
    example, `null` the null prompt. `label_scale` s moves every kept token state away from the mask token by the factor
    s (the label scale of the class-conditional reads, applied to the text encoder's states; 1 = off)."""

    def __init__(self, model, context, mask, null, label_scale=1.0):
        self.model, self.context, self.mask, self.null, self.label_scale = model, context, mask, int(null), float(label_scale)

    def __getattr__(self, name):   # parameters, sub-modules (the balancer's final_layer), geometry
        return getattr(self.model, name)

    def rows(self, y):
        keep = y != self.null
        rows = torch.where(keep, y, torch.zeros_like(y))
        context, mask = self.context[rows], self.mask[rows] * keep[:, None].to(self.mask.dtype)
        if self.label_scale != 1.0:
            token = self.model.mask_token.to(context.dtype)
            context = token + self.label_scale * (context - token)
        return context, mask

    def __call__(self, x, y, omega=None, **kw):
        return self.model(x, self.rows(y), omega=omega, **kw)

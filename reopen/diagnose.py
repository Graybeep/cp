"""Week-8 circuit-diagnosis primitives (seed-parametric).

Question this module exists to answer (Circuit Identity, CLAUDE.md operational def):
did an intervention that RECOVERED a skill *repair the original circuit* or *build a
new one*?  Concretely for the arithmetic critical period:
    A = early baseline   (arith_delay0 ...)          healthy   (~0.79 exact-match)
    B = delayed/deficient (arith_delay50 ...)          deficient (~0.52)
    C = lr_shift recovered (arith_delay50 ..._ivlrshift) recovered (~0.77)

  * "Repaired original circuit": ablating the components important IN A damages C
    similarly to A, and activation-patching A's components reproduces behaviour in C.
  * "Built a new circuit": ablating/patching A's components has little effect on C.

Primitives here (all torch-hook based, non-invasive, model unchanged on exit):
  - load_diag_model(run_id, step)           -> (model, model_config)
  - component ids: ("attn", L) | ("mlp", L) | ("head", L, h)   (residual-stream writers)
  - capture_means(model, ...)               -> mean activation per component (for mean-ablation)
  - ablate(model, targets, means)           context manager: mean-ablate the listed components
  - patch(model, donor_acts, targets)       context manager: overwrite components with donor's
  - linear_cka(X, Y)                        representation-similarity (0..1)

The arithmetic readout is reopen.skills.arithmetic.grade_exact_match — imported by callers.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import torch

from reopen.checkpoint import load_checkpoint
from reopen.config import CHECKPOINT_DIR, ModelConfig
from reopen.model import GPT


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_diag_model(run_id: str, step: int | None, device: str):
    """Load a checkpoint into a fresh GPT in eval mode. step=None -> latest."""
    ckdir = CHECKPOINT_DIR / run_id
    if step is None:
        path = sorted(ckdir.glob("step_*.pt"))[-1]
    else:
        path = ckdir / f"step_{step:07d}.pt"
    ck = load_checkpoint(path, device="cpu")
    model = GPT(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["model_config"]


# ---------------------------------------------------------------------------
# component addressing
# A "component" writes to the residual stream. We hook the sub-module whose
# forward OUTPUT is the residual contribution:
#   ("attn", L) -> blocks[L].attn output      (B,T,C)
#   ("mlp",  L) -> blocks[L].mlp  output      (B,T,C)
#   ("head", L, h) -> the h-th head slice of blocks[L].attn.proj INPUT
#      (proj is linear, so zero/replace the head's head_dim slice pre-proj)
# ---------------------------------------------------------------------------
def _submodule(model, comp):
    kind = comp[0]
    L = comp[1]
    if kind == "attn":
        return model.blocks[L].attn
    if kind == "mlp":
        return model.blocks[L].mlp
    if kind == "head":
        return model.blocks[L].attn.proj  # pre-hook on its input
    raise ValueError(comp)


def all_layer_components(n_layer: int):
    """Coarse (decisive-first) granularity: attn + mlp per layer."""
    comps = []
    for L in range(n_layer):
        comps.append(("attn", L))
        comps.append(("mlp", L))
    return comps


def all_head_components(n_layer: int, n_head: int):
    return [("head", L, h) for L in range(n_layer) for h in range(n_head)]


# ---------------------------------------------------------------------------
# mean capture (for mean-ablation: replace a component's output by its dataset
# mean, removing component-specific info while staying on-distribution)
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_means(model, input_batches, device, head_dim: int):
    """Return {comp: mean_vector} averaged over all real token positions across
    `input_batches` (list of LongTensor (B,T), right-padded with 0). Padding is
    excluded via a per-batch length mask passed alongside as (ids, lengths)."""
    sums, counts = {}, {}
    handles = []

    def acc(comp, vec):  # vec: (N, d) real-token rows only
        if comp not in sums:
            sums[comp] = torch.zeros(vec.shape[-1], device=device)
            counts[comp] = 0
        sums[comp] += vec.sum(0)
        counts[comp] += vec.shape[0]

    def mk_out_hook(comp):
        def hook(mod, inp, out):
            acc(comp, out[mask_ref["m"]].float())
        return hook

    def mk_head_prehook(L):
        def hook(mod, inp):
            x = inp[0]  # (B,T,C) concat heads
            B, T, C = x.shape
            nh = C // head_dim
            xr = x.view(B, T, nh, head_dim)
            for h in range(nh):
                acc(("head", L, h), xr[:, :, h, :][mask_ref["m"]].float())
            return None
        return hook

    mask_ref = {"m": None}
    for L in range(len(model.blocks)):
        handles.append(model.blocks[L].attn.register_forward_hook(mk_out_hook(("attn", L))))
        handles.append(model.blocks[L].mlp.register_forward_hook(mk_out_hook(("mlp", L))))
        handles.append(model.blocks[L].attn.proj.register_forward_pre_hook(mk_head_prehook(L)))
    try:
        for ids, lengths in input_batches:
            ids = ids.to(device)
            m = torch.zeros(ids.shape, dtype=torch.bool, device=device)
            for r, L_ in enumerate(lengths):
                m[r, :L_] = True
            mask_ref["m"] = m
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return {c: sums[c] / max(1, counts[c]) for c in sums}


# ---------------------------------------------------------------------------
# ablation: mean-ablate a set of components for the duration of the context
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def ablate(model, targets, means, head_dim: int):
    """Mean-ablate each component in `targets` (uses `means[comp]`). If a comp's
    mean is missing, falls back to zero-ablation."""
    handles = []
    by_kind = {"out": {}, "head": {}}
    for comp in targets:
        if comp[0] in ("attn", "mlp"):
            by_kind["out"][comp] = _submodule(model, comp)
        else:
            by_kind["head"].setdefault(comp[1], []).append(comp[2])

    def mk_out_hook(comp):
        mv = means.get(comp)
        def hook(mod, inp, out):
            if mv is None:
                return torch.zeros_like(out)
            return mv.to(out.dtype).expand_as(out)
        return hook

    def mk_head_prehook(L, heads):
        def hook(mod, inp):
            x = inp[0].clone()
            B, T, C = x.shape
            nh = C // head_dim
            xr = x.view(B, T, nh, head_dim)
            for h in heads:
                mv = means.get(("head", L, h))
                if mv is None:
                    xr[:, :, h, :] = 0.0
                else:
                    xr[:, :, h, :] = mv.to(x.dtype)
            return (xr.view(B, T, C),) + inp[1:]
        return hook

    try:
        for comp, sub in by_kind["out"].items():
            handles.append(sub.register_forward_hook(mk_out_hook(comp)))
        for L, heads in by_kind["head"].items():
            handles.append(model.blocks[L].attn.proj.register_forward_pre_hook(
                mk_head_prehook(L, heads)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# cross-model activation patching: run `model` but overwrite the listed
# components' outputs with activations captured from a donor on the SAME inputs.
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_acts(model, ids, device):
    """Capture full (B,T,C) outputs of every attn/mlp component on one batch."""
    acts, handles = {}, []
    def mk(comp):
        def hook(mod, inp, out):
            acts[comp] = out.detach().clone()
        return hook
    for L in range(len(model.blocks)):
        handles.append(model.blocks[L].attn.register_forward_hook(mk(("attn", L))))
        handles.append(model.blocks[L].mlp.register_forward_hook(mk(("mlp", L))))
    try:
        model(ids.to(device))
    finally:
        for h in handles:
            h.remove()
    return acts


@contextlib.contextmanager
def patch(model, donor_acts, targets):
    """Overwrite each ("attn"/"mlp", L) component output with donor_acts[comp]."""
    handles = []
    def mk(comp):
        def hook(mod, inp, out):
            return donor_acts[comp].to(out.dtype)
        return hook
    try:
        for comp in targets:
            handles.append(_submodule(model, comp).register_forward_hook(mk(comp)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---- head-level capture + patch (for causal tracing at head granularity) ----
@torch.no_grad()
def capture_head_acts(model, ids, device, head_dim: int):
    """Capture each head's pre-proj slice (its residual-writing signal) on one
    batch. Returns {("head",L,h): (B,T,head_dim)}."""
    acts, handles = {}, []
    def mk(L):
        def hook(mod, inp):
            x = inp[0]; B, T, C = x.shape; nh = C // head_dim
            xr = x.view(B, T, nh, head_dim)
            for h in range(nh):
                acts[("head", L, h)] = xr[:, :, h, :].detach().clone()
        return hook
    for L in range(len(model.blocks)):
        handles.append(model.blocks[L].attn.proj.register_forward_pre_hook(mk(L)))
    try:
        model(ids.to(device))
    finally:
        for h in handles:
            h.remove()
    return acts


@contextlib.contextmanager
def patch_heads(model, donor_head_acts, targets, head_dim: int):
    """Overwrite the listed heads' pre-proj slices with donor activations."""
    handles = []
    by_L = {}
    for comp in targets:
        by_L.setdefault(comp[1], []).append(comp[2])
    def mk(L, heads):
        def hook(mod, inp):
            x = inp[0].clone(); B, T, C = x.shape; nh = C // head_dim
            xr = x.view(B, T, nh, head_dim)
            for h in heads:
                d = donor_head_acts[("head", L, h)]
                xr[:, :, h, :] = d.to(x.dtype)
            return (xr.view(B, T, C),) + inp[1:]
        return hook
    try:
        for L, heads in by_L.items():
            handles.append(model.blocks[L].attn.proj.register_forward_pre_hook(mk(L, heads)))
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# representation similarity
# ---------------------------------------------------------------------------
def linear_cka(X, Y):
    """Linear CKA between two (N, d) activation matrices (numpy or torch).
    Returns a scalar in [0,1]; 1 = identical representational geometry."""
    import numpy as np
    X = np.asarray(X, dtype=np.float64); Y = np.asarray(Y, dtype=np.float64)
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    # HSIC-based linear CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    xty = X.T @ Y
    num = (xty ** 2).sum()
    den = np.sqrt((X.T @ X).__pow__(2).sum() * (Y.T @ Y).__pow__(2).sum())
    return float(num / den) if den > 0 else 0.0

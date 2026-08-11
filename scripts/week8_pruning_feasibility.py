"""Week-8 FEASIBILITY PILOT: does pruning-based localization discriminate the known
reorganized triad (23M seed 1, ablation transfer 0.187) from the known repair triads
(seeds 2/3, 0.710/0.868)? Gates fixed in notes/week8-pruning-feasibility-prereg.md.

Method: learned gates over {head, mlp} components with model weights FROZEN. A gate
g in [0,1] interpolates a component toward its own mean activation, so g=0 is exactly
mean-ablation (same baseline convention as the primary method) and the difference from
ablation is marginal-vs-joint selection, not the intervention. Objective =
answer-span CE + lambda*||g||_1. Binarize at 0.5 -> mask; compare mask(A) vs mask(C) by IoU.

Gates (ALL must pass for GO):
  1. IoU(A, A') >= 0.70 across two independent fits   [reproducibility floor]
  2. spread of IoU(A,C) across seeds >= 0.20, not inside one 0.10 band  [dynamic range]
  3. seed 1 lowest IoU(A,C)                            [ordering vs primary readout]
  4. masked-A keeps >=80% of A's accuracy; complement-A <= 0.15  [functional validity]

Run: python scripts/week8_pruning_feasibility.py [--seeds 1 2 3] [--device cuda]
Outputs: logs/week8_pruning_feasibility/{results.json, results.md}
"""
from __future__ import annotations
import argparse, contextlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from reopen.config import DATA_DIR
from reopen import diagnose as D
from reopen.skills import arithmetic as A
from reopen.skills.arithmetic import ArithConfig, grade_exact_match

OUT = Path("logs/week8_pruning_feasibility")
# primary readout to compare against (logs/week8_circuit_identity/method_agreement.json)
ABL_TRANSFER_23M = {1: 0.187, 2: 0.710, 3: 0.868}


def run_ids(seed, size="23M"):
    return {"A": f"arith_delay0_exp20_size{size}_seed{seed}",
            "C": f"arith_delay50_exp20_size{size}_seed{seed}_ivlrshift0at2501"}


def make_batches(tok, problems, device, batch_size=32):
    """(ids, answer-span mask, lengths) batches. `_answer_span` returns the INDEX of
    the first answer token, so the answer span is [start, len(ids)). Lengths are the
    real (pre-padding) token counts, needed by capture_means to exclude padding."""
    out = []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        rows, starts = [], []
        for a, b in chunk:
            prompt = f"\n{a}+{b}="
            ids, start = A._answer_span(tok, prompt, prompt + str(a + b))
            rows.append(ids); starts.append(start)
        maxlen = max(len(r) for r in rows)
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        m = torch.zeros(len(rows), maxlen, dtype=torch.bool)
        for r, (ids, s0) in enumerate(zip(rows, starts)):
            x[r, :len(ids)] = torch.tensor(ids)
            m[r, s0:len(ids)] = True
        out.append((x.to(device), m.to(device), [len(r) for r in rows]))
    return out


class Gates:
    """Per-head and per-mlp gates applied via hooks. g=1 -> untouched, g=0 -> mean-ablated."""

    def __init__(self, model, means, n_layer, n_head, head_dim, device, init=1.5, seed=0):
        self.model, self.means = model, means
        self.n_layer, self.n_head, self.head_dim = n_layer, n_head, head_dim
        gen = torch.Generator(device="cpu").manual_seed(seed)
        jitter = torch.randn(n_layer, n_head, generator=gen) * 0.05
        self.head_logit = (torch.full((n_layer, n_head), init) + jitter).to(device).requires_grad_(True)
        jm = torch.randn(n_layer, generator=gen) * 0.05
        self.mlp_logit = (torch.full((n_layer,), init) + jm).to(device).requires_grad_(True)
        self.hard = None  # optional {(kind,L,h): 0/1} to force a binary mask

    def params(self):
        return [self.head_logit, self.mlp_logit]

    def g_head(self):
        return torch.sigmoid(self.head_logit)

    def g_mlp(self):
        return torch.sigmoid(self.mlp_logit)

    @contextlib.contextmanager
    def applied(self):
        handles = []
        gh, gm = self.g_head(), self.g_mlp()
        if self.hard is not None:
            gh = torch.tensor([[self.hard[("head", L, h)] for h in range(self.n_head)]
                               for L in range(self.n_layer)],
                              dtype=gh.dtype, device=gh.device)
            gm = torch.tensor([self.hard[("mlp", L)] for L in range(self.n_layer)],
                              dtype=gm.dtype, device=gm.device)

        def mk_head_pre(L):
            def hook(mod, inp):
                x = inp[0]
                B, T, C = x.shape
                xr = x.view(B, T, self.n_head, self.head_dim)
                mu = torch.stack([self.means[("head", L, h)] for h in range(self.n_head)])
                g = gh[L].view(1, 1, self.n_head, 1)
                y = g * xr + (1 - g) * mu.view(1, 1, self.n_head, self.head_dim)
                return (y.view(B, T, C),) + inp[1:]
            return hook

        def mk_mlp_out(L):
            def hook(mod, i, out):
                mu = self.means[("mlp", L)].view(1, 1, -1)
                return gm[L] * out + (1 - gm[L]) * mu
            return hook

        for L in range(self.n_layer):
            handles.append(self.model.blocks[L].attn.proj.register_forward_pre_hook(mk_head_pre(L)))
            handles.append(self.model.blocks[L].mlp.register_forward_hook(mk_mlp_out(L)))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def mask(self, thr=0.5):
        m = {}
        gh, gm = self.g_head().detach(), self.g_mlp().detach()
        for L in range(self.n_layer):
            for h in range(self.n_head):
                m[("head", L, h)] = int(gh[L, h] > thr)
            m[("mlp", L)] = int(gm[L] > thr)
        return m


def fit_mask(model, means, batches, n_layer, n_head, head_dim, device,
             lam=0.03, steps=300, lr=0.1, seed=0, log=None):
    """Learn the sparsest gate configuration that preserves arithmetic loss."""
    G = Gates(model, means, n_layer, n_head, head_dim, device, seed=seed)
    opt = torch.optim.Adam(G.params(), lr=lr)
    order = np.random.default_rng(seed).permutation(len(batches))
    for t in range(steps):
        ids, am, _ = batches[order[t % len(batches)]]
        with G.applied():
            logits, _ = model(ids)
        lp = F.log_softmax(logits[:, :-1].float(), dim=-1)
        tgt, tm = ids[:, 1:], am[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        ce = (nll * tm).sum() / tm.sum().clamp(min=1)
        sparsity = G.g_head().sum() + G.g_mlp().sum()
        loss = ce + lam * sparsity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            G.head_logit.clamp_(-12, 12); G.mlp_logit.clamp_(-12, 12)
        if log is not None and t % 100 == 0:
            log.append({"step": t, "ce": float(ce), "kept": int(sum(G.mask().values()))})
    return G


def iou(m1, m2):
    keys = set(m1) | set(m2)
    inter = sum(1 for k in keys if m1.get(k) and m2.get(k))
    union = sum(1 for k in keys if m1.get(k) or m2.get(k))
    return inter / union if union else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--size", default="23M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lam", type=float, default=0.03)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    rep = {"config": vars(args), "seeds": {}}

    for seed in args.seeds:
        cfg = ArithConfig(carrying="mixed", seed=seed)
        problems = A.make_eval_set(cfg, args.n_eval)
        batches = make_batches(tok, problems, dev)
        rids = run_ids(seed, args.size)
        ent = {"run_ids": rids, "abl_transfer": ABL_TRANSFER_23M.get(seed)}

        models, mcfg = {}, None
        for k, rid in rids.items():
            models[k], mcfg = D.load_diag_model(rid, None, dev)
        n_layer, n_head = mcfg["n_layer"], mcfg["n_head"]
        head_dim = mcfg["n_embd"] // mcfg["n_head"]

        mean_batches = [(ids, lengths) for ids, _am, lengths in batches]
        means = {k: D.capture_means(models[k], mean_batches, dev, head_dim) for k in models}
        base = {k: grade_exact_match(models[k], tok, problems, device=dev) for k in models}
        ent["baseline_acc"] = base

        masks, logs = {}, {}
        for k in ("A", "C"):
            for tag, gseed in ((k, 0),) + ((("A_rep", 1),) if k == "A" else ()):
                lg = []
                G = fit_mask(models[k], means[k], batches, n_layer, n_head, head_dim, dev,
                             lam=args.lam, steps=args.steps, seed=gseed, log=lg)
                masks[tag] = G.mask(); logs[tag] = lg
                if tag == "A":
                    # gate 4: functional validity of the mask, on A
                    G.hard = masks["A"]
                    with G.applied():
                        keep = grade_exact_match(models["A"], tok, problems, device=dev)
                    G.hard = {kk: 1 - v for kk, v in masks["A"].items()}
                    with G.applied():
                        comp = grade_exact_match(models["A"], tok, problems, device=dev)
                    ent["gate4"] = {"A_acc": base["A"], "masked_A_acc": keep,
                                    "retained_frac": keep / base["A"] if base["A"] else float("nan"),
                                    "complement_A_acc": comp,
                                    "mask_size": int(sum(masks["A"].values())),
                                    "n_components": len(masks["A"])}
        ent["mask_sizes"] = {k: int(sum(v.values())) for k, v in masks.items()}
        ent["iou_A_C"] = iou(masks["A"], masks["C"])
        ent["iou_A_Arep"] = iou(masks["A"], masks["A_rep"])
        ent["fit_logs"] = logs
        ent["masks"] = {k: {"|".join(map(str, kk)): v for kk, v in m.items()}
                        for k, m in masks.items()}
        rep["seeds"][seed] = ent
        print(f"[seed {seed}] acc A={base['A']:.3f} C={base['C']:.3f} | "
              f"IoU(A,C)={ent['iou_A_C']:.3f} | IoU(A,A')={ent['iou_A_Arep']:.3f} | "
              f"|mask_A|={ent['mask_sizes']['A']} |mask_C|={ent['mask_sizes']['C']} | "
              f"gate4 retain={ent['gate4']['retained_frac']:.2f} comp={ent['gate4']['complement_A_acc']:.3f}")

    # ---------------- gates ----------------
    ious = {s: rep["seeds"][s]["iou_A_C"] for s in rep["seeds"]}
    floors = {s: rep["seeds"][s]["iou_A_Arep"] for s in rep["seeds"]}
    vals = np.array(list(ious.values()))
    g1 = min(floors.values()) >= 0.70
    g2 = bool((vals.max() - vals.min()) >= 0.20)
    g3 = (min(ious, key=ious.get) == 1) if 1 in ious else None
    g4 = all(rep["seeds"][s]["gate4"]["retained_frac"] >= 0.80
             and rep["seeds"][s]["gate4"]["complement_A_acc"] <= 0.15 for s in rep["seeds"])
    rep["gates"] = {
        "g1_reproducibility_floor": {"min_iou_A_Arep": min(floors.values()),
                                     "threshold": 0.70, "pass": bool(g1)},
        "g2_dynamic_range": {"spread": float(vals.max() - vals.min()),
                             "threshold": 0.20, "pass": g2,
                             "within_0.10_band": bool(vals.max() - vals.min() < 0.10)},
        "g3_ordering": {"iou_by_seed": ious, "seed1_lowest": g3, "pass": bool(g3)},
        "g4_functional_validity": {"pass": bool(g4)},
    }
    rep["VERDICT"] = ("GO" if (g1 and g2 and g3 and g4) else "NO-GO")
    if len(ious) >= 3:
        a = np.array([ious[s] for s in sorted(ious)])
        b = np.array([ABL_TRANSFER_23M[s] for s in sorted(ious)])
        rep["descriptive_vs_primary"] = {
            "note": "n=3: DESCRIPTIVE ONLY, no correlation claimed (pre-registered)",
            "iou": a.tolist(), "abl_transfer": b.tolist(),
            "pearson_r_uninterpretable_at_n3": float(np.corrcoef(a, b)[0, 1]),
            "iou_sd": float(a.std(ddof=1)), "abl_transfer_sd": float(b.std(ddof=1))}

    fname = f"results{args.tag}.json"
    json.dump(rep, open(OUT / fname, "w"), indent=1, default=float)
    print("\n=== GATES ===")
    for k, v in rep["gates"].items():
        print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'}  {v}")
    print(f"\nVERDICT: {rep['VERDICT']}")
    print(f"wrote {OUT/fname}")


if __name__ == "__main__":
    main()

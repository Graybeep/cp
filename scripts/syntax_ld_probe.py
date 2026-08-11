"""
Representational probing for filler-gap dependency — does DELAY change what the model
ENCODES about long-distance wh-licensing, even though behavioral accuracy is floored/flat
in every condition (mean, dynamics, rate all null)?

Motivation (representation-precedes-behavior): a net can encode a structural distinction in
its hidden states before its next-token output reliably uses it. If early-exposure (delay0)
checkpoints encode filler-gap structure and delayed (delay50) ones don't — while behavioral
LD stays at floor for both — that is a critical period visible only at the representational
level, invisible to every behavioral metric tried. Costs NO new training: reads hidden states
from checkpoints already on disk (step_0005000.pt), fits a lightweight linear probe on the
existing BLiMP minimal-pair items.

DESIGN — why this probe is clean (the pooling trick):
  Each minimal pair (good, bad) differs ONLY in the complementizer (who/what vs that); gap
  presence is fixed within a paradigm. We POOL:
     with_gap paradigm: good = wh   (gap licensed by filler)   bad = that
     no_gap  paradigm: good = that (no gap, no filler)        bad = wh
  In the pooled set the GRAMMATICAL class = {wh+gap, that+nogap} and UNGRAMMATICAL =
  {that+gap, wh+nogap}. Complementizer identity is thus BALANCED across the label — a probe
  predicting grammaticality cannot cheat off the wh/that token; it must read whether the
  complementizer MATCHES the gap structure. That is the licensing interaction itself, read
  off the representation. We read the LAST-TOKEN hidden state (after the gap) so the LD items
  require the model to have MAINTAINED the filler across the long dependency.

GUARDS (per the plan):
  * SELECTIVITY control (Hewitt-Liang): re-fit with grammaticality labels SHUFFLED. A real
    structural signal must beat its own shuffled control, or "delay encodes less" is a
    probe-capacity artifact. Report probe_acc AND selectivity = probe_acc - control_acc.
  * POSITIVE control: the LOCAL paradigm probe. Local behavioral licensing is real (~0.58-0.60),
    so if even the local probe can't beat chance the probe method is too weak at this scale and
    the LD result is uninformative. Local-probe passing is the gate that makes the LD read meaningful.
  * SEED depth: run seeds 1-3 per delay, not seed 1 alone — a "representational CP" at seed 1
    only would just be the known seed-1 high-acquisition outlier in new clothing.
  * GROUPED CV: good and bad of the same pair share lexical content -> kept in the same fold
    (group = paradigm+pairID) so the probe can't memorize surface lexis.

Run: python scripts/syntax_ld_probe.py [--device cuda] [--n 1000]
Outputs: logs/syntax_ld_probe/{per_run.csv, summary.md}
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.checkpoint import load_checkpoint
from reopen.config import CHECKPOINT_DIR, DATA_DIR, ModelConfig
from reopen.model import GPT
from reopen.skills import syntax
from tokenizers import Tokenizer

DELAYS = [0, 25, 50, 75]
SEEDS = [1, 2, 3]
OUT = Path("logs/syntax_ld_probe")


def run_id(delay, seed):
    return f"syntax_delay{delay}_exp10_size23M_seed{seed}"


@torch.no_grad()
def hidden_states(model, tok, sentences, device, n_layers, batch_size=64):
    """Return list over layers[0..n_layers] of (N, d) last-token hidden states.
    Layer 0 = token embeddings; layers 1..n_layer = block outputs; last = final norm."""
    # register hooks on each block + capture embedding + final norm
    caps = {}
    handles = []
    def mk(i):
        def hook(mod, inp, out):
            caps[i] = out.detach()
        return hook
    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(mk(i + 1)))
    # embedding + final norm captured manually below via a wrapper
    feats = [[] for _ in range(n_layers + 2)]  # emb, blocks..., normf
    try:
        for s in range(0, len(sentences), batch_size):
            batch = sentences[s:s + batch_size]
            ids = [tok.encode(x).ids for x in batch]
            lens = [len(t) for t in ids]
            maxlen = max(lens)
            x = torch.zeros(len(ids), maxlen, dtype=torch.long, device=device)
            for r, t in enumerate(ids):
                x[r, :len(t)] = torch.tensor(t, device=device)
            emb = model.wte(x)                       # (B,T,d) layer 0
            _ = model(x)                             # populates block hooks
            last = torch.tensor([l - 1 for l in lens], device=device)
            ar = torch.arange(len(ids), device=device)
            feats[0].append(emb[ar, last].float().cpu())
            for i in range(len(model.blocks)):
                feats[i + 1].append(caps[i + 1][ar, last].float().cpu())
            # final norm applied to last block output
            nf = model.norm_f(caps[len(model.blocks)])
            feats[-1].append(nf[ar, last].float().cpu())
    finally:
        for h in handles:
            h.remove()
    return [torch.cat(f, 0).numpy() for f in feats]


def logreg_cv(X, y, groups, lam=10.0, folds=5, seed=0):
    """Closed-form L2 (ridge) linear probe with grouped K-fold CV; returns mean
    test accuracy. Standard linear-probe choice: solve (X'X + lam*I) w = X'y with
    y in {-1,+1}, predict sign. One linear solve per fold (fast, deterministic).
    Features standardized per-fold on train statistics only."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    fold_of = {g: i % folds for i, g in enumerate(uniq)}
    gfold = np.array([fold_of[g] for g in groups])
    ypm = np.where(y == 1, 1.0, -1.0)
    accs = []
    d = X.shape[1]
    for f in range(folds):
        tr, te = gfold != f, gfold == f
        if te.sum() == 0 or tr.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        A = Xtr.T @ Xtr + lam * np.eye(d)
        w = np.linalg.solve(A, Xtr.T @ ypm[tr])
        b = ypm[tr].mean()  # intercept ~ class balance in {-1,+1}
        pred = ((Xte @ w + b) > 0).astype(int)
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs))


def build_pool(tok, names, n):
    """Return sentences, y(grammatical=1), groups for a pooled paradigm set."""
    sents, y, groups = [], [], []
    for nm in names:
        pairs = syntax.load_blimp_pairs(DATA_DIR / "blimp" / f"{nm}.jsonl")[:n]
        for pid, (good, bad) in enumerate(pairs):
            sents += [good, bad]
            y += [1, 0]
            groups += [f"{nm}_{pid}", f"{nm}_{pid}"]
    return sents, np.array(y), np.array(groups)


def probe_run(delay, seed, tok, device, n):
    rid = run_id(delay, seed)
    path = CHECKPOINT_DIR / rid / "step_0005000.pt"
    if not path.exists():
        return None
    ck = load_checkpoint(path, device="cpu")
    model = GPT(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    nl = ck["model_config"]["n_layer"]

    results = {"delay": delay, "seed": seed}
    for tag, names in [("LD", ["wh_vs_that_with_gap_long_distance",
                               "wh_vs_that_no_gap_long_distance"]),
                       ("local", ["wh_vs_that_with_gap",
                                  "wh_vs_that_no_gap"])]:
        sents, y, groups = build_pool(tok, names, n)
        layers = hidden_states(model, tok, sents, device, nl)
        # real probe at every layer; shuffled control only at the best layer
        per_layer = [logreg_cv(X, y, groups) for X in layers]
        best = int(np.argmax(per_layer))
        yshuf = y.copy(); np.random.RandomState(7).shuffle(yshuf)
        ctrl = logreg_cv(layers[best], yshuf, groups)
        results[f"{tag}_best_layer"] = best
        results[f"{tag}_probe_acc"] = per_layer[best]
        results[f"{tag}_ctrl_acc"] = ctrl
        results[f"{tag}_selectivity"] = per_layer[best] - ctrl
        results[f"{tag}_per_layer"] = [round(a, 4) for a in per_layer]
        print(f"    [{tag}] best layer {best}: probe={per_layer[best]:.4f} ctrl={ctrl:.4f} "
              f"per_layer={[round(a,3) for a in per_layer]}", flush=True)
    del model
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))

    import csv
    keys = ["delay","seed","LD_probe_acc","LD_ctrl_acc","LD_selectivity","LD_best_layer",
            "local_probe_acc","local_ctrl_acc","local_selectivity","local_best_layer"]
    rows = []
    for d in DELAYS:
        for s in SEEDS:
            r = probe_run(d, s, tok, args.device, args.n)
            if r is None:
                print(f"MISSING {run_id(d,s)}"); continue
            print(f"delay{d} seed{s}: LD probe={r['LD_probe_acc']:.4f} "
                  f"(ctrl {r['LD_ctrl_acc']:.4f}, sel {r['LD_selectivity']:+.4f}, "
                  f"layer {r['LD_best_layer']})  |  local probe={r['local_probe_acc']:.4f} "
                  f"(sel {r['local_selectivity']:+.4f})", flush=True)
            rows.append(r)
            # incremental save so a mid-run death never loses completed work
            with open(OUT / "per_run.csv", "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                wr.writeheader()
                for rr in rows: wr.writerow(rr)
            (OUT / "per_run_full.json").write_text(json.dumps(rows, indent=2))

    # ---- aggregate + tests ----
    from scipy import stats
    def grp(delay, key):
        return np.array([r[key] for r in rows if r["delay"] == delay])
    lines = ["# Representational probe — filler-gap dependency (23M, seeds 1-3)\n"]
    lines.append("Last-token linear probe on pooled minimal pairs; grammaticality target with "
                 "complementizer balanced across the label; grouped CV; selectivity control.\n")
    lines.append("## LD (long-distance) probe accuracy by delay (chance=0.5)\n")
    lines.append("| delay | LD probe mean±SD | selectivity mean±SD | local probe mean±SD | local sel |")
    lines.append("|------:|:----------------:|:-------------------:|:-------------------:|:---------:|")
    for d in DELAYS:
        ld = grp(d, "LD_probe_acc"); sel = grp(d, "LD_selectivity")
        lo = grp(d, "local_probe_acc"); lsel = grp(d, "local_selectivity")
        if len(ld) == 0: continue
        lines.append(f"| {d} | {ld.mean():.4f} ± {ld.std(ddof=1):.4f} | "
                     f"{sel.mean():+.4f} ± {sel.std(ddof=1):.4f} | "
                     f"{lo.mean():.4f} ± {lo.std(ddof=1):.4f} | {lsel.mean():+.4f} |")
    # key contrast delay0 vs delay50
    a, c = grp(0, "LD_probe_acc"), grp(50, "LD_probe_acc")
    if len(a) >= 2 and len(c) >= 2:
        t, p = stats.ttest_ind(a, c, equal_var=False)
        lines.append(f"\n**Representational CP contrast (LD probe, delay0 vs delay50):** "
                     f"delay0={a.mean():.4f}±{a.std(ddof=1):.4f}, delay50={c.mean():.4f}±{c.std(ddof=1):.4f}, "
                     f"Welch t={t:.3f} p={p:.4f}")
    # is LD encoded at all? one-sample vs chance and vs selectivity
    allld = np.array([r["LD_probe_acc"] for r in rows])
    allsel = np.array([r["LD_selectivity"] for r in rows])
    t1, p1 = stats.ttest_1samp(allld, 0.5)
    t2, p2 = stats.ttest_1samp(allsel, 0.0)
    lines.append(f"\n**Is LD represented at all (pooled across all runs, n={len(allld)})?** "
                 f"probe vs chance 0.5: mean={allld.mean():.4f} t={t1:.3f} p={p1:.4f}; "
                 f"selectivity vs 0: mean={allsel.mean():+.4f} t={t2:.3f} p={p2:.4f}")
    allloc = np.array([r["local_probe_acc"] for r in rows])
    lines.append(f"\n**Positive control (local probe, all runs):** mean={allloc.mean():.4f} "
                 f"(must exceed chance + LD for the LD read to be informative)")
    txt = "\n".join(lines)
    (OUT / "summary.md").write_text(txt)
    print("\n" + txt)
    print(f"\nWrote {OUT}/summary.md , per_run.csv")


if __name__ == "__main__":
    main()

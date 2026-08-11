"""THE BLOCKING CONTROL for the agreement-attraction finding.

WHY THIS EXISTS. Every BLiMP distractor item has the attractor number *mismatching* the
head, so "below chance on attractor items" (0.383 balanced, n=6) is confounded with
"these items are longer / contain a PP or RC". The sg/pl asymmetry does NOT disambiguate:
the unigram prior pushes the same direction (control sg 0.105 / pl 0.830). So the existing
data cannot separate ATTRACTION (linear-proximity heuristic) from COMPLEXITY (any
intervening material hurts).

DESIGN — a fully-crossed sextet per lexical frame. For one (head, attractor, frame, verb)
combination we generate all six cells, sharing lexical content:

    bare_sg        The farmer is / are very happy.                     good = is
    bare_pl        The farmers are / is very happy.                    good = are
    sg_match       The farmer near the clerk is / are very happy.      good = is
    sg_mismatch    The farmer near the clerks is / are very happy.     good = is
    pl_match       The farmers near the clerks are / is very happy.    good = are
    pl_mismatch    The farmers near the clerk are / is very happy.     good = are

THE KEY PROPERTY: within a head number, match vs mismatch uses the IDENTICAL verb pair
with the IDENTICAL correct answer. Only the attractor noun's number differs. So the
verb-form frequency prior — the thing that wrecked raw BLiMP accuracy — cancels EXACTLY,
not merely on average. And all four attractor cells have matched length/structure, so
complexity is held constant too. The unigram-only control therefore has an analytic
attraction of exactly 0.000, which the script verifies numerically.

DECOMPOSITION:
    complexity cost = acc(bare)  - acc(match)        # does intervening material hurt?
    attraction      = acc(match) - acc(mismatch)     # does the attractor's NUMBER hurt?
Balanced = mean over head numbers (cancels any residual number preference).

PRE-REGISTERED READ (fixed before running):
  (a) ATTRACTION CONFIRMED   — balanced attraction > 0, p < 0.05, SAME SIGN in both head-
      number directions, and mismatch cells below chance. The linear-proximity claim holds
      and the write-up proceeds.
  (b) COMPLEXITY-ONLY        — balanced attraction CI includes 0 while complexity cost is
      large and match cells are ALSO below chance. The linear-proximity claim DIES; the
      finding is "intervening material hurts", which is far less interesting and largely
      already known. Do not write it up as attraction.
  (c) MIXED                  — both components non-zero. Attraction survives but is partial;
      report the decomposition, never the raw below-chance number alone.
Sign consistency across head numbers is required for (a) because a one-direction effect is
indistinguishable from a residual number bias interacting with sentence length.

Scoring: verb-position log-prob (PRIMARY — prefix is identical across the pair, so this is
the clean conditional) and whole-sentence log-prob (SECONDARY, for comparability with the
0.383 / 0.726 figures). These are two scorings of ONE probe, not independent methods —
per the project's cross-method rule they must not be reported as mutual corroboration.

Run: python scripts/agreement_attraction_quartet.py [--n 1000] [--sizes 23M]
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from reopen.config import DATA_DIR, CHECKPOINT_DIR
from reopen import diagnose as D
from reopen.skills.syntax import _sentence_logprobs

OUT = Path("logs/sv_agreement_probe")

HEADS = [("farmer", "farmers"), ("teacher", "teachers"), ("doctor", "doctors"),
         ("boy", "boys"), ("girl", "girls"), ("dog", "dogs"), ("cat", "cats"),
         ("author", "authors"), ("driver", "drivers"), ("sister", "sisters"),
         ("man", "men"), ("woman", "women"), ("child", "children")]
ATTRS = [("clerk", "clerks"), ("student", "students"), ("neighbor", "neighbors"),
         ("pilot", "pilots"), ("nurse", "nurses"), ("guard", "guards"),
         ("actor", "actors"), ("waiter", "waiters"), ("man", "men"), ("child", "children")]
# PP frames and object-relative frames. RC verbs are PAST TENSE on purpose: past-tense
# verbs are number-neutral in English, so the RC contributes no extra agreement cue.
FRAMES = ["near the {a}", "next to the {a}", "beside the {a}", "behind the {a}",
          "that the {a} visited", "that the {a} liked", "that the {a} watched"]
# (singular verb, plural verb, continuation)
VERBS = [("is", "are", "very happy."), ("was", "were", "very happy."),
         ("has", "have", "a new bicycle."), ("does", "do", "not know the answer."),
         ("isn't", "aren't", "very happy."), ("wasn't", "weren't", "ready yet.")]

CELLS = ["bare_sg", "bare_pl", "sg_match", "sg_mismatch", "pl_match", "pl_mismatch"]


def usable_verbs(tok):
    """Keep only verb pairs whose two forms tokenize to the SAME number of tokens.

    Summed log-prob over unequal token counts is length-biased (fewer tokens => higher
    score), which would masquerade as a syntactic preference. Enforcing equality removes
    that bias outright rather than correcting for it."""
    keep, drop = [], []
    for v_sg, v_pl, cont in VERBS:
        pre = tok.encode("The farmer").ids
        n_sg = len(tok.encode(f"The farmer {v_sg}").ids) - len(pre)
        n_pl = len(tok.encode(f"The farmer {v_pl}").ids) - len(pre)
        (keep if n_sg == n_pl else drop).append((v_sg, v_pl, cont, n_sg, n_pl))
    for v_sg, v_pl, _, a, b in drop:
        print(f"  dropped verb pair {v_sg}/{v_pl}: {a} vs {b} tokens (length bias)")
    return [(a, b, c) for a, b, c, _, _ in keep]


def build(n: int, verbs, seed: int = 0):
    """Return list of sextets; each is {cell: (prefix, good_verb, bad_verb)}."""
    rng = random.Random(seed)
    combos = [(h, a, f, v) for h in HEADS for a in ATTRS for f in FRAMES for v in verbs
              if h[0] != a[0]]
    rng.shuffle(combos)
    out = []
    for (h_sg, h_pl), (a_sg, a_pl), frame, (v_sg, v_pl, cont) in combos[:n]:
        item = {
            "bare_sg":     (f"The {h_sg}",                          v_sg, v_pl),
            "bare_pl":     (f"The {h_pl}",                          v_pl, v_sg),
            "sg_match":    (f"The {h_sg} {frame.format(a=a_sg)}",    v_sg, v_pl),
            "sg_mismatch": (f"The {h_sg} {frame.format(a=a_pl)}",    v_sg, v_pl),
            "pl_match":    (f"The {h_pl} {frame.format(a=a_pl)}",    v_pl, v_sg),
            "pl_mismatch": (f"The {h_pl} {frame.format(a=a_sg)}",    v_pl, v_sg),
        }
        out.append({"cont": cont, "frame_type": "rc" if "that the" in frame else "pp",
                    "cells": item})
    return out


# ------------------------------------------------------------------ verb-position scoring
def _spans(tok, prefix: str, verb: str):
    """Token ids for prefix+verb, and the [start,end) index span of the verb.
    Returns None if the tokenizer does not treat prefix as a clean token prefix."""
    pre = tok.encode(prefix).ids
    full = tok.encode(f"{prefix} {verb}").ids
    if full[:len(pre)] != pre or len(full) <= len(pre):
        return None
    return full, len(pre), len(full)


@torch.no_grad()
def verb_logprobs(model, tok, items, device, batch_size=128):
    """sum log P(verb tokens | prefix). items = [(prefix, verb, cont), ...]."""
    out = np.full(len(items), np.nan)
    prepared = []
    for i, (prefix, verb, _cont) in enumerate(items):
        s = _spans(tok, prefix, verb)
        if s is not None:
            prepared.append((i, *s))
    for b in range(0, len(prepared), batch_size):
        chunk = prepared[b:b + batch_size]
        maxlen = max(len(ids) for _, ids, _, _ in chunk)
        x = torch.zeros(len(chunk), maxlen, dtype=torch.long)
        for r, (_, ids, _, _) in enumerate(chunk):
            x[r, :len(ids)] = torch.tensor(ids)
        logits, _ = model(x.to(device))
        logp = torch.log_softmax(logits.float(), dim=-1)
        for r, (i, ids, s0, s1) in enumerate(chunk):
            # logits[:, j] predicts token j+1 => logprob of ids[j] is logp[r, j-1, ids[j]]
            idx = torch.arange(s0, s1, device=device)
            out[i] = logp[r, idx - 1, torch.tensor(ids[s0:s1], device=device)].sum().item()
    return out


def score_cells(scorer, sextets):
    """scorer(list[(prefix, verb, cont)]) -> np.array. Returns {cell: 0/1 array}."""
    flat, meta = [], []
    for k, sx in enumerate(sextets):
        for cell in CELLS:
            prefix, good, bad = sx["cells"][cell]
            flat += [(prefix, good, sx["cont"]), (prefix, bad, sx["cont"])]
            meta += [(k, cell, "g"), (k, cell, "b")]
    vals = scorer(flat)
    acc = {c: np.full(len(sextets), np.nan) for c in CELLS}
    for j in range(0, len(vals), 2):
        k, cell, _ = meta[j]
        g, b = vals[j], vals[j + 1]
        if np.isfinite(g) and np.isfinite(b):
            acc[cell][k] = float(g > b)
    return acc


def decompose(acc):
    """Balanced complexity cost and attraction, plus per-direction values."""
    m = {c: np.nanmean(acc[c]) for c in CELLS}
    r = {"cells": m}
    for num in ("sg", "pl"):
        r[f"attraction_{num}"] = m[f"{num}_match"] - m[f"{num}_mismatch"]
        r[f"complexity_{num}"] = m[f"bare_{num}"] - m[f"{num}_match"]
    r["attraction"] = 0.5 * (r["attraction_sg"] + r["attraction_pl"])
    r["complexity"] = 0.5 * (r["complexity_sg"] + r["complexity_pl"])
    r["balanced_bare"] = 0.5 * (m["bare_sg"] + m["bare_pl"])
    r["balanced_match"] = 0.5 * (m["sg_match"] + m["pl_match"])
    r["balanced_mismatch"] = 0.5 * (m["sg_mismatch"] + m["pl_mismatch"])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--runs", nargs="+", default=[
        f"arith_delay0_exp20_size23M_seed{s}" for s in (1, 2, 3)] + [
        f"syntax_delay0_exp10_size23M_seed{s}" for s in (1, 2, 3)])
    ap.add_argument("--preset", choices=["scale", "primary"], default=None,
                    help="scale = the 72-model size sweep (3 sizes x 24 seeds)")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to load; default = latest")
    ap.add_argument("--skip-sentence", action="store_true",
                    help="skip whole-sentence scoring (secondary, never counted as "
                         "corroboration) — halves cost on large run sets")
    ap.add_argument("--tag", default="quartet",
                    help="output basename: logs/sv_agreement_probe/<tag>.json")
    args = ap.parse_args()
    if args.preset == "scale":
        args.runs = [f"syntax_delay0_exp10_size{s}_seed{d}_szsweep"
                     for s in ("11M", "23M", "60M") for d in range(1, 25)]
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    verbs = usable_verbs(tok)
    print(f"usable verb pairs: {[f'{a}/{b}' for a, b, _ in verbs]}")
    sextets = build(args.n, verbs)
    dropped = sum(1 for sx in sextets for c in CELLS
                  if _spans(tok, sx["cells"][c][0], sx["cells"][c][1]) is None)
    print(f"{len(sextets)} sextets ({len(sextets) * 6} cells, {len(sextets) * 12} sentences); "
          f"tokenizer-unclean cells dropped: {dropped}")
    print(f"frame types: pp={sum(s['frame_type'] == 'pp' for s in sextets)} "
          f"rc={sum(s['frame_type'] == 'rc' for s in sextets)}")

    # ---- unigram control: attraction must be EXACTLY 0 (analytic guarantee) ----
    arr = np.memmap(DATA_DIR / "tokenized" / "train.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(arr), minlength=tok.get_vocab_size()).astype(np.float64) + 1.0
    ulp = np.log(counts / counts.sum())

    def uni_scorer(items):
        return np.array([ulp[np.array(tok.encode(f"{p} {v}").ids)].sum()
                         for p, v, _c in items])

    ctrl = decompose(score_cells(uni_scorer, sextets))
    print(f"\n[CONTROL unigram-only] attraction {ctrl['attraction']:+.4f} "
          f"(must be 0.0000)   complexity {ctrl['complexity']:+.4f}   "
          f"balanced_bare {ctrl['balanced_bare']:.3f}")

    results = {}
    for rid in args.runs:
        ck = CHECKPOINT_DIR / rid
        if not ck.exists() or (args.step is not None
                               and not (ck / f"step_{args.step:07d}.pt").exists()):
            print(f"!! missing {rid}" + (f" step {args.step}" if args.step else ""))
            continue
        model, _ = D.load_diag_model(rid, args.step, args.device)
        acc_v = score_cells(lambda it: verb_logprobs(model, tok, it, args.device), sextets)
        if args.skip_sentence:
            acc_s = {c: np.full(len(sextets), np.nan) for c in CELLS}
        else:
            acc_s = score_cells(
                lambda it: _sentence_logprobs(
                    model, tok, [f"{p} {v} {c}" for p, v, c in it],
                    args.device, 128).numpy(), sextets)
        rv, rs = decompose(acc_v), decompose(acc_s)
        # per-frame-type split on the primary scoring
        by_ft = {}
        for ft in ("pp", "rc"):
            mask = np.array([sx["frame_type"] == ft for sx in sextets])
            by_ft[ft] = decompose({c: acc_v[c][mask] for c in CELLS})
        results[rid] = {"verb": rv, "sentence": rs, "by_frame": by_ft}
        if len(args.runs) <= 12:
            results[rid]["_raw_verb"] = {c: acc_v[c].tolist() for c in CELLS}
        print(f"[{rid}] bare {rv['balanced_bare']:.3f} match {rv['balanced_match']:.3f} "
              f"mismatch {rv['balanced_mismatch']:.3f} | cpx {rv['complexity']:+.3f} "
              f"ATTR {rv['attraction']:+.3f} (sg {rv['attraction_sg']:+.3f}/"
              f"pl {rv['attraction_pl']:+.3f}) | pp {by_ft['pp']['attraction']:+.3f} "
              f"rc {by_ft['rc']['attraction']:+.3f}"
              + ("" if args.skip_sentence else f" | sent {rs['attraction']:+.3f}"))
        del model
        torch.cuda.empty_cache()

    # ---------------- verdict against the pre-registered read ----------------
    if results:
        att = np.array([results[r]["verb"]["attraction"] for r in results])
        a_sg = np.array([results[r]["verb"]["attraction_sg"] for r in results])
        a_pl = np.array([results[r]["verb"]["attraction_pl"] for r in results])
        cpx = np.array([results[r]["verb"]["complexity"] for r in results])
        mis = np.array([results[r]["verb"]["balanced_mismatch"] for r in results])
        mat = np.array([results[r]["verb"]["balanced_match"] for r in results])
        t = stats.ttest_1samp(att, 0.0)
        print("\n" + "=" * 78)
        print(f"ATTRACTION  {att.mean():+.4f} +/- {att.std(ddof=1):.4f}  n={len(att)}  "
              f"t={t.statistic:.2f} p={t.pvalue:.2e}")
        print(f"  sign-consistent across head number: "
              f"sg {a_sg.mean():+.4f} (pos in {int((a_sg > 0).sum())}/{len(a_sg)}), "
              f"pl {a_pl.mean():+.4f} (pos in {int((a_pl > 0).sum())}/{len(a_pl)})")
        print(f"COMPLEXITY  {cpx.mean():+.4f} +/- {cpx.std(ddof=1):.4f}  "
              f"p={stats.ttest_1samp(cpx, 0.0).pvalue:.2e}")
        print(f"match {mat.mean():.4f} vs chance p={stats.ttest_1samp(mat, 0.5).pvalue:.2e} | "
              f"mismatch {mis.mean():.4f} vs chance p={stats.ttest_1samp(mis, 0.5).pvalue:.2e}")
        same_sign = (a_sg > 0).all() and (a_pl > 0).all()
        conf = t.pvalue < 0.05 and att.mean() > 0 and same_sign and mis.mean() < 0.5
        cplx_only = t.pvalue >= 0.05 and mat.mean() < 0.5
        print("\nVERDICT: " + ("(a) ATTRACTION CONFIRMED" if conf else
                               "(b) COMPLEXITY-ONLY — attraction claim dies" if cplx_only else
                               "(c) MIXED / does not cleanly meet (a) or (b)"))
        summary = dict(attraction_mean=float(att.mean()), attraction_sd=float(att.std(ddof=1)),
                       attraction_p=float(t.pvalue), attraction_sg=float(a_sg.mean()),
                       attraction_pl=float(a_pl.mean()), sign_consistent=bool(same_sign),
                       complexity_mean=float(cpx.mean()), match=float(mat.mean()),
                       mismatch=float(mis.mean()), n=len(att),
                       verdict="a" if conf else ("b" if cplx_only else "c"))
    else:
        summary = {}

    (OUT / f"{args.tag}.json").write_text(json.dumps(
        {"control_unigram": ctrl, "models": results, "summary": summary,
         "n_sextets": len(sextets), "step": args.step}, indent=2))
    print(f"\nwrote {OUT}/{args.tag}.json")


if __name__ == "__main__":
    main()

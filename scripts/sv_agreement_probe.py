"""Cheapest-possible headroom check for subject-verb agreement as a candidate
replacement skill for wh-filler-gap syntax.

QUESTION: on THIS 10M-word BabyLM corpus at 23M params, does subject-verb agreement
clear a real ceiling (the way local wh-licensing did, ~0.60) or sit at floor (the way
long-distance did, ~0.53)? No training is required — existing delay-0 checkpoints
already saw the corpus; agreement was never an injected skill, so this measures what
the corpus alone teaches.

FREQUENCY-CONFOUND CHECK (the reason this script is not just blimp_accuracy):
Every BLiMP agreement pair differs in exactly ONE word — the verb form
("hasn't"/"haven't", "references"/"reference"). That is the SAME shape as the
wh_vs_that bug, where the score was dominated by the unigram prior for "that" and a
step-100 model scored 0.000/1.000 with zero syntax. So raw agreement accuracy is
untrustworthy for the same reason, and we compute three things:

  1. raw per-paradigm accuracy                       (the confounded number)
  2. NUMBER-BALANCED INTERACTION = mean(acc on singular-subject items,
     acc on plural-subject items). A model that always prefers one number form
     scores 1.0 on one subset and 0.0 on the other => exactly 0.5. This is the
     direct analogue of syntax.wh_licensing_interaction().
  3. UNIGRAM-FREQUENCY CONTROL — an explicit frequency-only "model" scoring each
     sentence by summed unigram log-prob estimated from train.bin. Whatever this
     scores IS the frequency prior. (Stronger than the step-100 proxy used for wh:
     it is exact, and no early checkpoint survives on disk.)

Read: trust metric 2, and only if metric 3 lands near 0.5 on it.

Run: python scripts/sv_agreement_probe.py [--device cuda]
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from reopen.config import DATA_DIR
from reopen import diagnose as D
from reopen.skills import syntax as S

OUT = Path("logs/sv_agreement_probe")
BLIMP = DATA_DIR / "blimp"

ATTRACTOR = ["distractor_agreement_relational_noun", "distractor_agreement_relative_clause"]
NO_ATTRACTOR = ["regular_plural_subject_verb_agreement_1", "regular_plural_subject_verb_agreement_2",
                "irregular_plural_subject_verb_agreement_1", "irregular_plural_subject_verb_agreement_2"]
PARADIGMS = ATTRACTOR + NO_ATTRACTOR

# singular-agreeing -> plural-agreeing verb forms
SG_PL = {"is": "are", "was": "were", "has": "have", "does": "do",
         "isn't": "aren't", "wasn't": "weren't", "hasn't": "haven't", "doesn't": "don't"}
PL_SG = {v: k for k, v in SG_PL.items()}


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w).lower()


def subject_number(good: str, bad: str) -> str | None:
    """'sg' if the GOOD sentence uses a singular-agreeing verb (=> singular subject),
    'pl' if it uses a plural-agreeing verb. None if the pair is not a clean
    number-minimal contrast (e.g. cross-tense 'isn't'/'weren't')."""
    gw, bw = good.split(), bad.split()
    if len(gw) != len(bw):
        return None
    diff = [(g, b) for g, b in zip(gw, bw) if g != b]
    if len(diff) != 1:
        return None
    g, b = _norm(diff[0][0]), _norm(diff[0][1])
    if SG_PL.get(g) == b:
        return "sg"
    if PL_SG.get(g) == b:
        return "pl"
    # regular 3sg -s / bare form
    for suf in ("es", "s"):
        if g == b + suf:
            return "sg"
        if b == g + suf:
            return "pl"
    return None


def load_split(paradigm: str):
    """Return (pairs, numbers) with unclassifiable pairs dropped."""
    path = BLIMP / f"{paradigm}.jsonl"
    pairs, nums, dropped = [], [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        n = subject_number(d["sentence_good"], d["sentence_bad"])
        if n is None:
            dropped += 1
            continue
        pairs.append((d["sentence_good"], d["sentence_bad"]))
        nums.append(n)
    return pairs, np.array(nums), dropped


def balanced_interaction(correct: np.ndarray, nums: np.ndarray):
    """mean(acc | singular-subject, acc | plural-subject). Chance for a pure
    number-prior model = 0.5 exactly."""
    sg, pl = nums == "sg", nums == "pl"
    a_sg = correct[sg].mean() if sg.any() else np.nan
    a_pl = correct[pl].mean() if pl.any() else np.nan
    return float(np.nanmean([a_sg, a_pl])), float(a_sg), float(a_pl), int(sg.sum()), int(pl.sum())


# ---------------------------------------------------------------- unigram control
def unigram_logprobs(tok) -> np.ndarray:
    """Unigram log-prob per token id, estimated from the actual training corpus."""
    arr = np.memmap(DATA_DIR / "tokenized" / "train.bin", dtype=np.uint16, mode="r")
    counts = np.bincount(np.asarray(arr), minlength=tok.get_vocab_size()).astype(np.float64)
    counts += 1.0  # Laplace, so unseen forms are finite
    return np.log(counts / counts.sum())


def unigram_correct(tok, ulp: np.ndarray, pairs) -> np.ndarray:
    """Frequency-only scoring: sum of unigram log-probs over the sentence."""
    def score(s):
        return float(ulp[np.array(tok.encode(s).ids)].sum())
    return np.array([score(g) > score(b) for g, b in pairs])


@torch.no_grad()
def model_correct(model, tok, pairs, device, batch_size=64) -> np.ndarray:
    good = S._sentence_logprobs(model, tok, [p[0] for p in pairs], device, batch_size)
    bad = S._sentence_logprobs(model, tok, [p[1] for p in pairs], device, batch_size)
    return (good > bad).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", nargs="+", default=[
        f"arith_delay0_exp20_size23M_seed{s}" for s in (1, 2, 3)] + [
        f"syntax_delay0_exp10_size23M_seed{s}" for s in (1, 2, 3)])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))

    data = {p: load_split(p) for p in PARADIGMS}
    print("paradigm balance (after dropping non-number-minimal pairs):")
    for p in PARADIGMS:
        pairs, nums, dropped = data[p]
        print(f"  {p:<45} n={len(pairs):>4}  sg={int((nums=='sg').sum()):>4} "
              f"pl={int((nums=='pl').sum()):>4}  dropped={dropped}")

    # ---- frequency-only control FIRST: if it already scores high on a metric,
    # that metric is confounded and no model number computed with it is meaningful.
    ulp = unigram_logprobs(tok)
    print("\n[CONTROL] unigram-frequency-only model (no context, no syntax):")
    control = {}
    for p in PARADIGMS:
        pairs, nums, _ = data[p]
        c = unigram_correct(tok, ulp, pairs)
        inter, a_sg, a_pl, n_sg, n_pl = balanced_interaction(c, nums)
        control[p] = dict(raw=float(c.mean()), interaction=inter, acc_sg=a_sg, acc_pl=a_pl)
        print(f"  {p:<45} raw={c.mean():.3f}   balanced={inter:.3f}   "
              f"(sg {a_sg:.3f} / pl {a_pl:.3f})")

    # ---- real models ----
    results = {}
    for rid in args.runs:
        try:
            model, _ = D.load_diag_model(rid, None, args.device)
        except Exception as e:
            print(f"\n!! skip {rid}: {type(e).__name__} {e}")
            continue
        r = {}
        print(f"\n[{rid}]")
        for p in PARADIGMS:
            pairs, nums, _ = data[p]
            c = model_correct(model, tok, pairs, args.device)
            inter, a_sg, a_pl, _, _ = balanced_interaction(c, nums)
            r[p] = dict(raw=float(c.mean()), interaction=inter, acc_sg=a_sg, acc_pl=a_pl)
            print(f"  {p:<45} raw={c.mean():.3f}   balanced={inter:.3f}   "
                  f"(sg {a_sg:.3f} / pl {a_pl:.3f})")
        r["_attractor_mean"] = float(np.nanmean([r[p]["interaction"] for p in ATTRACTOR]))
        r["_no_attractor_mean"] = float(np.nanmean([r[p]["interaction"] for p in NO_ATTRACTOR]))
        print(f"  {'>> balanced interaction: attractor':<45} {r['_attractor_mean']:.3f}")
        print(f"  {'>> balanced interaction: no-attractor':<45} {r['_no_attractor_mean']:.3f}")
        results[rid] = r
        del model
        torch.cuda.empty_cache()

    # ---- headroom summary: the number that decides whether a Detect grid is viable ----
    summary = {}
    if results:
        print("\n" + "=" * 72)
        print("HEADROOM SUMMARY — balanced interaction (chance 0.5)")
        print(f"  reference: wh local 0.604 | wh long-distance 0.534 (floor) | chance 0.500")
        for key, label in [("_attractor_mean", "attractor (Linzen-style)"),
                           ("_no_attractor_mean", "no-attractor (simple)")]:
            vals = np.array([results[r][key] for r in results])
            summary[key] = dict(mean=float(vals.mean()), sd=float(vals.std(ddof=1)),
                                lo=float(vals.min()), hi=float(vals.max()), n=len(vals))
            print(f"  {label:<28} {vals.mean():.3f} +/- {vals.std(ddof=1):.3f}  "
                  f"[{vals.min():.3f}, {vals.max():.3f}]  n={len(vals)}")
        ctrl_a = np.nanmean([control[p]["interaction"] for p in ATTRACTOR])
        ctrl_n = np.nanmean([control[p]["interaction"] for p in NO_ATTRACTOR])
        print(f"  unigram control: attractor {ctrl_a:.3f} | no-attractor {ctrl_n:.3f} "
              f"(both ~0.5 => metric is NOT frequency-confounded)")
        summary["control"] = {"attractor": float(ctrl_a), "no_attractor": float(ctrl_n)}

    payload = {"control_unigram": control, "models": results, "summary": summary,
               "paradigm_notes": {
                   "_1_variants": "verb form varies (has/have); balancing cancels verb-form frequency",
                   "_2_variants": "SUBJECT NOUN varies (students/student); balancing cancels noun-number frequency",
                   "irregular_plural_subject_verb_agreement_2":
                       "dropped: irregular noun plurals (women/woman) not handled by the "
                       "-s classifier; irregular_1 already covers irregular agreement"},
               "reference_points": {"wh_local_licensing_interaction": 0.604,
                                    "wh_long_distance_licensing_interaction": 0.534,
                                    "chance": 0.5}}
    (OUT / "results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()

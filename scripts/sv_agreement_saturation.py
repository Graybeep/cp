"""How early does subject-verb agreement saturate, and how deep is the attraction error?

Two questions, both answered with ZERO training on checkpoints already on disk.

(A) SATURATION / plausible-effect-size estimate.
    The quantity that predicts whether a delay manipulation can produce a deficit is
    TIME-TO-PLATEAU RELATIVE TO BUDGET. A skill already at asymptote by step 2500 of a
    5000-step run has, by construction, nothing to lose from a 50%-delay: the delayed
    model still gets 2500 steps, which is all the skill ever needed. Arithmetic is the
    calibration case (its own curve is read from metrics.csv).

(B) SCALE + TRAJECTORY for the agreement-attraction error.
    Attraction sits at 0.383 balanced (below chance, below the 0.488 unigram control) at
    23M. Does it deepen with training (step 2500 -> 5000) and does it change with scale
    (11M/23M/60M)? The size-sweep gives n=24 seeds per size for free.

Run: python scripts/sv_agreement_saturation.py [--stage traj|scale|both]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from reopen.config import DATA_DIR, CHECKPOINT_DIR
from reopen import diagnose as D
from sv_agreement_probe import (PARADIGMS, ATTRACTOR, NO_ATTRACTOR, load_split,
                                balanced_interaction, model_correct)

OUT = Path("logs/sv_agreement_probe")


def score_model(model, tok, data, device):
    r = {}
    for p in PARADIGMS:
        pairs, nums, _ = data[p]
        c = model_correct(model, tok, pairs, device)
        inter, a_sg, a_pl, _, _ = balanced_interaction(c, nums)
        r[p] = dict(raw=float(c.mean()), interaction=inter, acc_sg=a_sg, acc_pl=a_pl)
    r["_attractor_mean"] = float(np.nanmean([r[p]["interaction"] for p in ATTRACTOR]))
    r["_no_attractor_mean"] = float(np.nanmean([r[p]["interaction"] for p in NO_ATTRACTOR]))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", default="both", choices=["traj", "scale", "both"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    data = {p: load_split(p) for p in PARADIGMS}
    payload = {}

    # ---------------- (A) trajectory: step 2500 vs 5000 on the 6 delay-0 models
    if args.stage in ("traj", "both"):
        runs = [f"arith_delay0_exp20_size23M_seed{s}" for s in (1, 2, 3)] + \
               [f"syntax_delay0_exp10_size23M_seed{s}" for s in (1, 2, 3)]
        traj = {}
        print("=" * 78)
        print("(A) TRAJECTORY — balanced interaction at step 2500 vs 5000")
        for rid in runs:
            traj[rid] = {}
            for step in (2500, 5000):
                if not (CHECKPOINT_DIR / rid / f"step_{step:07d}.pt").exists():
                    print(f"  !! missing {rid} step {step}")
                    continue
                model, _ = D.load_diag_model(rid, step, args.device)
                traj[rid][str(step)] = score_model(model, tok, data, args.device)
                del model
                torch.cuda.empty_cache()
            a = traj[rid]
            if "2500" in a and "5000" in a:
                print(f"  {rid}")
                for key, lab in [("_no_attractor_mean", "simple"), ("_attractor_mean", "attractor")]:
                    v25, v50 = a["2500"][key], a["5000"][key]
                    print(f"     {lab:<10} {v25:.3f} -> {v50:.3f}   (delta {v50 - v25:+.3f})")
        payload["trajectory"] = traj

    # ---------------- (B) scale: size sweep, n=24 seeds x 3 sizes, final step
    if args.stage in ("scale", "both"):
        print("=" * 78)
        print("(B) SCALE — size sweep (final checkpoint), 24 seeds per size")
        scale = {}
        for size in ("11M", "23M", "60M"):
            rows = []
            for seed in range(1, 25):
                rid = f"syntax_delay0_exp10_size{size}_seed{seed}_szsweep"
                if not (CHECKPOINT_DIR / rid).exists():
                    continue
                try:
                    model, _ = D.load_diag_model(rid, None, args.device)
                except Exception as e:
                    print(f"  !! skip {rid}: {type(e).__name__}")
                    continue
                r = score_model(model, tok, data, args.device)
                rows.append({"run": rid, "seed": seed,
                             "attractor": r["_attractor_mean"],
                             "no_attractor": r["_no_attractor_mean"],
                             "per_paradigm": {p: r[p] for p in PARADIGMS}})
                del model
                torch.cuda.empty_cache()
            scale[size] = rows
            if rows:
                at = np.array([x["attractor"] for x in rows])
                na = np.array([x["no_attractor"] for x in rows])
                print(f"  {size}: n={len(rows)}  attractor {at.mean():.3f} +/- {at.std(ddof=1):.3f}"
                      f"   simple {na.mean():.3f} +/- {na.std(ddof=1):.3f}"
                      f"   [attractor below 0.5 in {(at < 0.5).sum()}/{len(at)} seeds]")
        payload["scale"] = scale

    (OUT / "saturation_scale.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT}/saturation_scale.json")


if __name__ == "__main__":
    main()

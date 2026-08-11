"""Bootstrap CIs on the ablation transfer ratio — is the seed-2 vs seed-3 difference
(0.71 vs 0.87) a real internal gradient or noise within one 'repair' bucket?

For each seed: reuse the already-identified A_top / C_top head sets, compute per-problem
correctness for A and C under {no-ablation, ablate A_top}, then bootstrap-resample the eval
problems to get a 95% CI on transfer_ratio = relDamage(C by A_top) / relDamage(A by A_top).
Overlapping CIs across seeds 2 and 3 => the 'gradient' is not resolvable.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import DATA_DIR
from reopen import diagnose as D
from reopen.skills import arithmetic as A
from reopen.skills.arithmetic import ArithConfig, _answer_span

OUT = Path("logs/week8_circuit_identity")


def per_problem_correct(model, tok, problems, means, targets, head_dim, device, batch=64):
    """Bool array: exact-match per problem, with `targets` heads mean-ablated (targets=[]
    -> unablated)."""
    import contextlib
    ctx = D.ablate(model, targets, means, head_dim) if targets else contextlib.nullcontext()
    out = np.zeros(len(problems), dtype=bool)
    with torch.no_grad(), ctx:
        for i in range(0, len(problems), batch):
            chunk = problems[i:i+batch]
            rows, spans = [], []
            for a, b in chunk:
                prompt = f"\n{a}+{b}="
                ids, s = _answer_span(tok, prompt, prompt + str(a+b))
                rows.append(ids); spans.append((s, len(ids)-s))
            maxlen = max(len(r) for r in rows)
            x = torch.zeros(len(rows), maxlen, dtype=torch.long)
            for r, ids in enumerate(rows):
                x[r, :len(ids)] = torch.tensor(ids)
            preds = model(x.to(device))[0].argmax(-1).cpu()
            for r, (s, n) in enumerate(spans):
                out[i+r] = all(preds[r, s-1+j] == rows[r][s+j] for j in range(n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--size", default="23M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--eval-seed", type=int, default=1,
                    help="ArithConfig seed for the eval draw. 1 = the held-out "
                         "set; anything else is train-contaminated (see "
                         "week8_circuit_identity.py docstring).")
    ap.add_argument("--tag-suffix", default="",
                    help="suffix of the circuit-identity report to read AND of "
                         "the bootstrap artifact written")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    rng = np.random.RandomState(0)

    results = {}
    for seed in args.seeds:
        tag = (f"seed{seed}" + ("" if args.size == "23M" else f"_{args.size}")
               + args.tag_suffix)
        rep = json.load(open(OUT / f"{tag}.json"))
        A_top = [tuple(x["comp"]) for x in rep["A_head_localization"][:6]]
        cfg = ArithConfig(carrying="mixed", seed=args.eval_seed)
        problems = A.make_eval_set(cfg, args.n_eval)

        from scripts.week8_circuit_identity import run_ids, make_eval_batches
        rids = run_ids(seed, args.size)
        mA, mc = D.load_diag_model(rids["A"], None, args.device)
        mC, _ = D.load_diag_model(rids["C"], None, args.device)
        hd = mc["n_embd"] // mc["n_head"]
        batches = make_eval_batches(tok, problems, args.device)
        meansA = D.capture_means(mA, batches, args.device, hd)
        meansC = D.capture_means(mC, batches, args.device, hd)

        A_base = per_problem_correct(mA, tok, problems, meansA, [], hd, args.device)
        A_abl  = per_problem_correct(mA, tok, problems, meansA, A_top, hd, args.device)
        C_base = per_problem_correct(mC, tok, problems, meansC, [], hd, args.device)
        C_abl  = per_problem_correct(mC, tok, problems, meansC, A_top, hd, args.device)

        n = len(problems)
        def transfer(idx):
            ab = A_base[idx].mean(); aa = A_abl[idx].mean()
            cb = C_base[idx].mean(); ca = C_abl[idx].mean()
            relA = (ab - aa) / ab if ab > 0 else np.nan
            relC = (cb - ca) / cb if cb > 0 else np.nan
            return relC / relA if relA and relA > 0 else np.nan
        point = transfer(np.arange(n))
        boots = np.array([transfer(rng.randint(0, n, n)) for _ in range(args.boot)])
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        results[seed] = dict(point=float(point), lo=float(lo), hi=float(hi),
                             A_base=float(A_base.mean()), C_base=float(C_base.mean()))
        print(f"seed {seed} ({args.size}): transfer C/A = {point:.3f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]  (A={A_base.mean():.3f} C={C_base.mean():.3f})")

    if len(args.seeds) == 2:
        a, b = args.seeds
        overlap = not (results[a]["hi"] < results[b]["lo"] or results[b]["hi"] < results[a]["lo"])
        print(f"\nseed{a} vs seed{b} transfer-ratio 95% CIs "
              f"{'OVERLAP -> difference NOT resolvable (one repair bucket)' if overlap else 'DISJOINT -> real gradient'}")
        results["ci_overlap"] = bool(overlap)
    (OUT / f"transfer_bootstrap_{args.size}{args.tag_suffix}.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}/transfer_bootstrap_{args.size}{args.tag_suffix}.json")


if __name__ == "__main__":
    main()

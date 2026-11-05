"""Week-8 circuit-identity battery (seed-parametric): did lr_shift REPAIR the
original arithmetic circuit or BUILD A NEW one?

Models per seed (CLAUDE.md / week8-diagnosis-seed-scope):
  A = arith_delay0_exp20_size23M_seed{S}                 (early baseline, healthy)
  B = arith_delay50_exp20_size23M_seed{S}                (delayed, deficient)
  C = arith_delay50_exp20_size23M_seed{S}_ivlrshift0at2501 (lr_shift recovered)

Battery:
  0. Baseline exact-match accuracy for A, B, C (context for all damage numbers).
  1. ABLATION LOCALIZATION on A: mean-ablate each component (attn/mlp per layer),
     measure A's accuracy drop -> A's arithmetic circuit = the high-damage set.
  2. ABLATION TRANSFER: mean-ablate A's top-k components in C (and B), measure damage.
       C damaged like A  -> C reuses A's components        -> REPAIR
       C ~unharmed       -> C's arithmetic lives elsewhere -> NEW circuit
     (B included as the deficient reference.)
  3. REPRESENTATION SIMILARITY: linear CKA between A,C and A,B residual streams
     per layer on the same arithmetic inputs. High CKA(A,C) at circuit layers = repair.

Decision uses the operational definition directly; verdict is reported per seed and the
tooling is seed-parametric (run seed 1 full; seed 2 for the partial confirm).

EVAL SET (fixed 2026-07-28 — read this before changing `--eval-seed`):
  Only ONE arithmetic problem set is held out from training:
  `make_eval_set(ArithConfig(seed=1), 1000)` == `data/skills/arith/eval.jsonl`,
  which `prepare_skills.prepare_arith` passes as `exclude=` when writing train.bin.
  Any OTHER draw (e.g. `ArithConfig(seed=2)`) is ~17-19% problems the model was
  TRAINED ON, and is also a different item universe, so its accuracies are neither
  clean nor comparable across seeds.

  This script originally built its eval set with `ArithConfig(seed=args.seed)`, so the
  seed-2/3 runs (23M and 60M) were scored on contaminated, non-common draws. The default
  is now the held-out set regardless of run seed; `--eval-seed N` reproduces an original
  run and requires `--allow-contaminated-eval`.

Run: python scripts/week8_circuit_identity.py --seed 1 [--device cuda] [--heads]
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

from reopen.config import DATA_DIR
from reopen import diagnose as D
from reopen.skills import arithmetic as A
from reopen.skills.arithmetic import ArithConfig, grade_exact_match

OUT = Path("logs/week8_circuit_identity")


def run_ids(seed: int, size: str = "23M"):
    return {
        "A": f"arith_delay0_exp20_size{size}_seed{seed}",
        "B": f"arith_delay50_exp20_size{size}_seed{seed}",
        "C": f"arith_delay50_exp20_size{size}_seed{seed}_ivlrshift0at2501",
    }


def make_eval_batches(tok, problems, device, batch_size=64, prompt_prefix="\n"):
    """Tokenize problems into right-padded (ids, lengths) batches for mean-capture."""
    batches = []
    for i in range(0, len(problems), batch_size):
        chunk = problems[i:i + batch_size]
        rows = []
        for a, b in chunk:
            prompt = f"{prompt_prefix}{a}+{b}="
            ids, _ = A._answer_span(tok, prompt, prompt + str(a + b))
            rows.append(ids)
        maxlen = max(len(r) for r in rows)
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        for r, ids in enumerate(rows):
            x[r, :len(ids)] = torch.tensor(ids)
        batches.append((x, [len(r) for r in rows]))
    return batches


@torch.no_grad()
def residual_matrix(model, ids_batches, device):
    """Per layer, the last-real-token residual-stream vector for each problem,
    stacked into (N, d). Uses block outputs (post-attn+mlp) = residual stream."""
    caps = {L: [] for L in range(len(model.blocks))}
    handles = []
    store = {}
    def mk(L):
        def hook(mod, inp, out):
            store[L] = out.detach()
        return hook
    for L in range(len(model.blocks)):
        handles.append(model.blocks[L].register_forward_hook(mk(L)))
    try:
        for ids, lengths in ids_batches:
            model(ids.to(device))
            last = torch.tensor([l - 1 for l in lengths], device=device)
            ar = torch.arange(ids.shape[0], device=device)
            for L in range(len(model.blocks)):
                caps[L].append(store[L][ar, last].float().cpu().numpy())
    finally:
        for h in handles:
            h.remove()
    return {L: np.concatenate(caps[L], 0) for L in caps}


def acc(model, tok, problems, device):
    return grade_exact_match(model, tok, problems, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="23M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--topk", type=int, default=6, help="A-heads transferred to B/C")
    ap.add_argument("--eval-seed", type=int, default=1,
                    help="ArithConfig seed for the eval draw. Default 1 = THE "
                         "held-out set. Anything else is train-contaminated.")
    ap.add_argument("--allow-contaminated-eval", action="store_true",
                    help="permit an eval draw containing trained-on problems "
                         "(only to reproduce the pre-2026-07-28 runs)")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the output filename")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    cfg = ArithConfig(carrying="mixed", seed=args.eval_seed)
    problems = A.make_eval_set(cfg, args.n_eval)

    # FAIL-LOUD contamination guard: every eval problem must be one of the pairs
    # excluded from train.bin. A silent overlap here inflates every accuracy,
    # every ablation damage and therefore every transfer ratio in the report.
    held = set()
    for line in (DATA_DIR / "skills" / "arith" / "eval.jsonl").read_text(
            encoding="utf-8").splitlines():
        d = json.loads(line)
        held.add((d["a"], d["b"]))
    contaminated = [p for p in problems if p not in held]
    if contaminated:
        msg = (f"eval draw (seed {args.eval_seed}) contains "
               f"{len(contaminated)}/{len(problems)} "
               f"({100*len(contaminated)/len(problems):.1f}%) problems NOT in the "
               f"held-out set — they were trained on.")
        if not args.allow_contaminated_eval:
            raise SystemExit("REFUSING: " + msg + " Use --eval-seed 1, or pass "
                             "--allow-contaminated-eval to reproduce an old run.")
        print("WARNING: " + msg)
    ids_batches = make_eval_batches(tok, problems, dev)

    rids = run_ids(args.seed, args.size)
    models, mcfg = {}, None
    for k, rid in rids.items():
        models[k], mcfg = D.load_diag_model(rid, None, dev)
    n_layer, n_head, head_dim = mcfg["n_layer"], mcfg["n_head"], mcfg["n_embd"] // mcfg["n_head"]

    report = {"seed": args.seed, "run_ids": rids,
              "eval_seed": args.eval_seed, "n_eval": args.n_eval,
              "eval_contamination_frac": round(len(contaminated) / len(problems), 4)}

    # ---- 0. baseline accuracy ----
    base = {k: acc(models[k], tok, problems, dev) for k in "ABC"}
    report["baseline_acc"] = base
    print(f"\n[seed {args.seed}] baseline exact-match:  "
          f"A(early)={base['A']:.3f}  B(delayed)={base['B']:.3f}  C(lr_shift)={base['C']:.3f}")

    # mean-activations for each model (mean-ablation reference, on-distribution)
    means = {k: D.capture_means(models[k], ids_batches, dev, head_dim) for k in "ABC"}

    def damage(model, mns, targets, base_acc):
        with D.ablate(model, targets, mns, head_dim):
            a = acc(model, tok, problems, dev)
        return base_acc - a, a

    # ---- 0b. layer-level ablation: SATURATES (documented caveat) ----
    # Mean-ablating a whole attn/mlp layer drives every model to ~0, so it cannot
    # discriminate circuit identity. We localize + transfer at HEAD granularity.
    lay = []
    for c in D.all_layer_components(n_layer):
        d, a = damage(models["A"], means["A"], [c], base["A"])
        lay.append((list(c), round(d, 4)))
    report["layer_ablation_note"] = {
        "reason": "whole-layer mean-ablation saturates (any core layer -> acc~0); "
                  "not discriminative, transfer done at head level instead",
        "A_layer_damage": lay}

    # ---- 1. ablation localization on A at HEAD granularity (discriminative) ----
    headcomps = D.all_head_components(n_layer, n_head)
    locA = []
    for c in headcomps:
        d, a = damage(models["A"], means["A"], [c], base["A"])
        locA.append((c, d, a))
    locA.sort(key=lambda t: -t[1])
    report["A_head_localization"] = [
        {"comp": list(c), "A_damage": round(d, 4), "A_acc_ablated": round(a, 4)}
        for c, d, a in locA[:12]]
    print(f"\n[1] A's arithmetic circuit — top heads by A-damage (mean-ablation, "
          f"single-head; discriminative unlike layer-level):")
    for c, d, a in locA[:8]:
        print(f"    L{c[1]}H{c[2]}  A-damage={d:+.3f}  (A acc {base['A']:.3f}->{a:.3f})")

    # ---- 1b. localize C's OWN circuit (reciprocal test + robustness control) ----
    locC = []
    for c in headcomps:
        d, a = damage(models["C"], means["C"], [c], base["C"])
        locC.append((c, d, a))
    locC.sort(key=lambda t: -t[1])
    report["C_head_localization"] = [
        {"comp": list(c), "C_damage": round(d, 4)} for c, d, a in locC[:12]]

    A_top = [c for c, _, _ in locA[:args.topk]]
    C_top = [c for c, _, _ in locC[:args.topk]]
    inter = set(A_top) & set(C_top)
    jacc = len(inter) / len(set(A_top) | set(C_top))
    print(f"\n[1b] C's own top heads: {['L%dH%d'%(c[1],c[2]) for c in C_top]}")
    print(f"     A_top ∩ C_top = {['L%dH%d'%(c[1],c[2]) for c in inter] or 'none'}  "
          f"Jaccard={jacc:.2f}")

    # ---- 2. reciprocal ablation transfer (the decisive causal test) ----
    # frac of own accuracy each model loses when its OWN-top vs the OTHER's-top
    # heads are ablated.  REPAIR: A_top tanks C too.  NEW CIRCUIT: A_top spares C,
    # C_top (not A_top) is what tanks C -> a control against "C just robust to any ablation".
    def relloss(model_key, headset):
        d, a = damage(models[model_key], means[model_key], headset, base[model_key])
        return (d / base[model_key]) if base[model_key] > 0 else 0.0, a
    grid = {}
    for mk in "ABC":
        grid[mk] = {
            "own_A_top": relloss(mk, A_top),
            "own_C_top": relloss(mk, C_top),
        }
    report["reciprocal_ablation"] = {
        "A_top": [list(c) for c in A_top], "C_top": [list(c) for c in C_top],
        "jaccard": round(jacc, 3),
        "frac_own_acc_lost": {mk: {"A_top_heads": round(grid[mk]["own_A_top"][0], 4),
                                    "C_top_heads": round(grid[mk]["own_C_top"][0], 4)}
                              for mk in "ABC"}}
    print(f"\n[2] Reciprocal ablation — fraction of own accuracy lost:")
    print(f"     {'model':<14} ablate A's-heads   ablate C's-heads")
    for mk, lab in [("A","A early"),("B","B delayed"),("C","C lr_shift")]:
        print(f"     {lab:<14} {100*grid[mk]['own_A_top'][0]:>8.0f}%        "
              f"{100*grid[mk]['own_C_top'][0]:>8.0f}%")
    transfer_ratio = grid["C"]["own_A_top"][0] / (grid["A"]["own_A_top"][0] or 1e-9)
    control_ratio = grid["C"]["own_C_top"][0] / (grid["C"]["own_A_top"][0] or 1e-9)
    report["reciprocal_ablation"]["C_over_A_transfer_ratio"] = round(transfer_ratio, 3)
    report["reciprocal_ablation"]["C_ownheads_over_Aheads_ratio"] = round(control_ratio, 3)

    # ---- 3. representation similarity (CKA) ----
    R = {k: residual_matrix(models[k], ids_batches, dev) for k in "ABC"}
    cka_AC = {L: D.linear_cka(R["A"][L], R["C"][L]) for L in range(n_layer)}
    cka_AB = {L: D.linear_cka(R["A"][L], R["B"][L]) for L in range(n_layer)}
    report["cka"] = {"A_vs_C": {L: round(v, 4) for L, v in cka_AC.items()},
                     "A_vs_B": {L: round(v, 4) for L, v in cka_AB.items()}}
    print(f"\n[3] Linear CKA (residual stream) per layer:")
    print("    layer:   " + "  ".join(f"{L}" for L in range(n_layer)))
    print("    A vs C:  " + "  ".join(f"{cka_AC[L]:.2f}" for L in range(n_layer)))
    print("    A vs B:  " + "  ".join(f"{cka_AB[L]:.2f}" for L in range(n_layer)))

    # ---- verdict — leads with the CAUSAL test (operational def), CKA corroborates ----
    # The operational definition (CLAUDE.md) is causal: "built a new circuit if
    # ablating the baseline's components has little effect on the recovered model."
    # Primary = transfer_ratio (how much of A's damage C inherits). Control =
    # control_ratio (C's own heads must tank C far more than A's heads do, else C is
    # merely ablation-robust). Jaccard of the head sets corroborates. CKA is secondary
    # (global geometry; shared init keeps it high even when causal wiring differs).
    circuit_layers = sorted({c[1] for c in A_top})
    cka_circuit = float(np.mean([cka_AC[L] for L in circuit_layers]))

    # ---- CONTINUOUS readout — no categorical verdict is assigned ----------
    # HISTORY (Appendix B). This block used to emit REPAIR / NEW CIRCUIT /
    # MIXED from the rule `transfer_ratio >= 0.6 and jacc >= 0.4`. That rule
    # was never pre-registered (notes/week8-diagnosis-seed-scope.md fixes seed
    # scope and which diagnostics run, and specifies no verdict thresholds),
    # and its Jaccard conjunct contradicted the docstring directly above:
    # Jaccard is designated CORROBORATIVE there, but the conjunct made it a
    # hard gate on the repair branch only, so only one of the two verdicts
    # could ever be blocked by it. The conjunct was formally demoted on
    # 2026-07-26 (notes/week8-60M-seed2-prereg.md: "Jaccard is corroborative
    # only — it did NOT discriminate at 23M: seed 2 had Jaccard 0.09 WITH
    # transfer 0.71"), but the code was never changed, so the tool went on
    # applying a rule the project had already retired. Max Jaccard across the
    # six triads is 0.20, so the repair label was unreachable by construction.
    #
    # THE FIX IS NOT A REPLACEMENT THRESHOLD. Choosing new cut-points now,
    # with all six triads already measured, would be selecting a decision rule
    # after seeing the numbers it decides — the same error in a new coat, and
    # it would move the highest-transfer triad (23M seed 3, transfer 0.868)
    # into "repair" by fiat. §5.2 reports the transfer axis continuously and
    # assigns no categorical label; this tool now does the same. The measured
    # quantities below are unchanged — only the labelling was ever in question.
    verdict = (f"CONTINUOUS READOUT (no categorical label): "
               f"transfer={transfer_ratio:.3f}, control={control_ratio:.3f}, "
               f"jaccard={jacc:.3f}, cka@circuit={cka_circuit:.3f}. "
               f"Low transfer => A's heads are not causal in C (reorganization); "
               f"high transfer => C's arithmetic runs through A's heads (repair). "
               f"Read on the axis, not against a cut-point (Appendix B).")
    report["verdict"] = {
        "text": verdict,
        # explicit, machine-readable: this tool no longer classifies
        "classification": None,
        "classification_note": (
            "No categorical verdict is emitted. The former thresholds "
            "(transfer>=0.6 AND jaccard>=0.4) were unregistered and the "
            "Jaccard conjunct was demoted to corroborative on 2026-07-26; "
            "see Appendix B."),
        "transfer_ratio_C_over_A": round(transfer_ratio, 3),
        "control_ratio_Cheads_over_Aheads": round(control_ratio, 3),
        "jaccard_A_top_C_top": round(jacc, 3),
        "cka_AC_circuit_layers": round(cka_circuit, 3),
        "circuit_layers": circuit_layers}
    print(f"\n[READOUT seed {args.seed}] {verdict}")
    print(f"    causal transfer C/A = {transfer_ratio:.2f}  (low => A's heads not causal in C)")
    print(f"    control C-own/A-heads = {control_ratio:.2f}  (high => C DOES have its own circuit)")
    print(f"    Jaccard(A_top,C_top) = {jacc:.2f}   CKA(A,C)@circuit = {cka_circuit:.2f}")

    tag = (f"seed{args.seed}" + ("" if args.size == "23M" else f"_{args.size}")
           + args.tag_suffix)
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/{tag}.json")


if __name__ == "__main__":
    main()

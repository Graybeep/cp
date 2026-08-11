"""Week-8 activation-patching corroboration of the circuit-identity verdict.

The NEW-CIRCUIT claim (seed 1) so far rests on ONE method (mean-ablation). This adds an
INDEPENDENT causal method — denoising activation patching (causal tracing) — at head
granularity, and asks the same question a different way.

Setup: aligned minimal-pair prompts that differ in one operand digit, chosen so (a) the two
prompts tokenize to the SAME length (positions align) and (b) the first answer token differs.
    clean:  "\n347+285="   -> first answer token "6" (of 632)
    corrupt:"\n447+285="   -> first answer token "7" (of 732)
DENOISING patch: run the model on the CORRUPT prompt, patch in a single head's activation
captured from the CLEAN run, and check whether the model now predicts the CLEAN first answer
token. restoration(head) = fraction of pairs flipped corrupt->clean by that head = how much
that head causally carries the answer-distinguishing computation.

Circuit-identity read (independent of ablation):
  - restoration profile of A  -> A's causal heads (patching-derived)
  - restoration profile of C  -> C's causal heads
  - if A and C localize to DIFFERENT heads (low overlap), NEW CIRCUIT is corroborated by a
    second method; if the SAME heads, the ablation verdict is challenged.
  - cross-model: patch A-clean head acts into C-corrupt (A and C share seed-init but drift;
    CKA~0.66) -> does A's signal drive C? Reported with the basis-drift caveat.

Run: python scripts/week8_causal_trace.py --seed 1 --size 23M [--device cuda] [--n 400]
"""
from __future__ import annotations
import argparse, json, random, sys
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
from reopen.skills.arithmetic import ArithConfig, sample_problem, _answer_span

OUT = Path("logs/week8_causal_trace")


def run_ids(seed, size):
    return {
        "A": f"arith_delay0_exp20_size{size}_seed{seed}",
        "C": f"arith_delay50_exp20_size{size}_seed{seed}_ivlrshift0at2501",
    }


def make_corrupt_pairs(tok, cfg, n, prefix="\n"):
    """Aligned (clean_ids, corr_ids, first_ans_clean_tok, first_ans_corr_tok)."""
    rng = random.Random(cfg.seed + 777)
    pairs = []
    tries = 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        a, b = sample_problem(rng, cfg)
        sa = str(a)
        pos = rng.randint(0, len(sa) - 1)
        nd = str(rng.randint(0, 9))
        if (pos == 0 and nd == "0") or nd == sa[pos]:
            continue
        a2 = int(sa[:pos] + nd + sa[pos + 1:])
        c, c2 = a + b, a2 + b
        pc, pk = f"{prefix}{a}+{b}=", f"{prefix}{a2}+{b}="
        ec, ek = tok.encode(pc).ids, tok.encode(pk).ids
        if len(ec) != len(ek):
            continue
        ids_c, as_c = _answer_span(tok, pc, pc + str(c))
        ids_k, as_k = _answer_span(tok, pk, pk + str(c2))
        fa_c, fa_k = ids_c[as_c], ids_k[as_k]
        if fa_c == fa_k:
            continue
        pairs.append((ec, ek, fa_c, fa_k))
    return pairs


def batchify(seqs, device):
    maxlen = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), maxlen, dtype=torch.long)
    for r, s in enumerate(seqs):
        x[r, :len(s)] = torch.tensor(s)
    lengths = torch.tensor([len(s) for s in seqs])
    return x.to(device), (lengths - 1).to(device)  # last real index


@torch.no_grad()
def predict_last(model, ids, last_idx):
    logits, _ = model(ids)
    ar = torch.arange(ids.shape[0], device=ids.device)
    return logits[ar, last_idx].argmax(-1)


@torch.no_grad()
def restoration_profile(model, pairs, n_layer, n_head, head_dim, device, donor_model=None,
                        batch=128):
    """Per head: fraction of pairs where patching the CLEAN head-act into the CORRUPT run
    makes the model predict the clean first-answer token. donor_model=None -> within-model
    (clean acts from `model`); else cross-model (clean acts from donor_model)."""
    heads = D.all_head_components(n_layer, n_head)
    rest = {h: 0 for h in heads}
    corr_matches_corr = 0  # sanity: unpatched corrupt predicts corrupt token
    total = 0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        clean_ids, cl_last = batchify([p[0] for p in chunk], device)
        corr_ids, co_last = batchify([p[1] for p in chunk], device)
        fa_clean = torch.tensor([p[2] for p in chunk], device=device)
        fa_corr = torch.tensor([p[3] for p in chunk], device=device)
        # capture clean head acts from the donor (self or A)
        src = donor_model if donor_model is not None else model
        clean_acts = D.capture_head_acts(src, clean_ids, device, head_dim)
        # sanity: unpatched corrupt
        base_pred = predict_last(model, corr_ids, co_last)
        corr_matches_corr += (base_pred == fa_corr).sum().item()
        total += len(chunk)
        for h in heads:
            with D.patch_heads(model, {h: clean_acts[h]}, [h], head_dim):
                pred = predict_last(model, corr_ids, co_last)
            rest[h] += (pred == fa_clean).sum().item()
    return ({h: rest[h] / total for h in heads},
            corr_matches_corr / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="23M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = args.device

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    cfg = ArithConfig(carrying="mixed", seed=args.seed)
    pairs = make_corrupt_pairs(tok, cfg, args.n)
    print(f"[seed {args.seed} {args.size}] {len(pairs)} aligned corrupt pairs")

    rids = run_ids(args.seed, args.size)
    models, mc = {}, None
    for k, rid in rids.items():
        models[k], mc = D.load_diag_model(rid, None, dev)
    n_layer, n_head, head_dim = mc["n_layer"], mc["n_head"], mc["n_embd"] // mc["n_head"]

    report = {"seed": args.seed, "size": args.size, "run_ids": rids, "n_pairs": len(pairs)}

    # within-model causal traces
    profiles = {}
    for k in "AC":
        prof, sanity = restoration_profile(models[k], pairs, n_layer, n_head, head_dim, dev)
        profiles[k] = prof
        top = sorted(prof, key=lambda h: -prof[h])[:args.topk]
        report[f"{k}_top_causal_heads"] = [{"comp": list(h), "restoration": round(prof[h], 4)}
                                           for h in top]
        report[f"{k}_corrupt_sanity"] = round(sanity, 4)
        print(f"\n[{k}] causal trace — top heads by restoration "
              f"(unpatched-corrupt baseline predicts corrupt token {sanity:.2f}):")
        for h in top:
            print(f"    L{h[1]}H{h[2]}  restoration={prof[h]:+.3f}")

    A_top = [tuple(x["comp"]) for x in report["A_top_causal_heads"]]
    C_top = [tuple(x["comp"]) for x in report["C_top_causal_heads"]]
    inter = set(A_top) & set(C_top)
    jacc = len(inter) / len(set(A_top) | set(C_top))
    report["patching_overlap"] = {"A_top": [list(h) for h in A_top],
                                   "C_top": [list(h) for h in C_top],
                                   "intersection": [list(h) for h in inter],
                                   "jaccard": round(jacc, 3)}
    print(f"\n[overlap] patching-derived A_top ∩ C_top = "
          f"{['L%dH%d'%(h[1],h[2]) for h in inter] or 'none'}  Jaccard={jacc:.2f}")

    # how much of A's top-causal restoration survives IN C (patching A's heads into C-corrupt,
    # using C's OWN clean acts) — i.e., are A's causal heads causal in C?
    A_top_restoration_in_C = np.mean([profiles["C"][h] for h in A_top])
    C_top_restoration_in_C = np.mean([profiles["C"][h] for h in C_top])
    report["A_heads_causal_in_C"] = {
        "mean_restoration_of_A_top_in_C": round(float(A_top_restoration_in_C), 4),
        "mean_restoration_of_C_top_in_C": round(float(C_top_restoration_in_C), 4)}
    print(f"\n[causal-in-C] mean restoration in C: A's top heads={A_top_restoration_in_C:.3f}  "
          f"vs C's own top heads={C_top_restoration_in_C:.3f}")

    # cross-model: patch A-CLEAN head acts into C-corrupt (basis-drift caveat)
    prof_cross, _ = restoration_profile(models["C"], pairs, n_layer, n_head, head_dim, dev,
                                        donor_model=models["A"])
    cross_A_top = np.mean([prof_cross[h] for h in A_top])
    report["cross_model_A_into_C"] = {
        "mean_restoration_A_clean_into_C_at_A_top": round(float(cross_A_top), 4),
        "caveat": "A,C share seed-init but drift (CKA~0.66); low value is consistent with "
                  "different circuits but also partly reflects basis drift"}
    print(f"[cross-model] patch A-clean into C-corrupt at A's top heads: "
          f"restoration={cross_A_top:.3f}  (basis-drift caveat)")

    # corroboration verdict
    if jacc <= 0.2 and A_top_restoration_in_C < 0.5 * C_top_restoration_in_C:
        v = ("CORROBORATES NEW CIRCUIT — patching localizes A and C to different heads, and "
             "A's causal heads are weakly causal in C")
    elif jacc >= 0.4:
        v = "CORROBORATES REPAIR — patching localizes A and C to overlapping heads"
    else:
        v = "PARTIAL / MIXED — see numbers"
    report["corroboration_verdict"] = v
    print(f"\n[CORROBORATION seed {args.seed} {args.size}] {v}")

    (OUT / f"seed{args.seed}_{args.size}.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}/seed{args.seed}_{args.size}.json")


if __name__ == "__main__":
    main()

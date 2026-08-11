"""Grow the syntax skill pool by frame recombination (see
reopen/skills/syntax_augment.py for why and how).

DRY RUN BY DEFAULT — the naturalness gate. Prints stats and writes a
review sample; nothing under data/ changes until --commit:

    python scripts/augment_syntax.py                # stats + 200-sentence sample
    python scripts/augment_syntax.py --commit       # rebuild syntax.bin

--commit rebuilds data/skills/syntax/syntax.bin as originals + generated
(shuffled), writes augment_meta.json alongside, and updates stats.json.
COMPARABILITY: runs trained on the old vs new syntax.bin are not
comparable (same rule as vocab.bin); all pilot/grid syntax runs must
train on the final committed pool. The old pool is reproducible from
held_lines.txt, which is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import DATA_DIR, LOG_DIR  # noqa: E402
from reopen.skills import syntax, syntax_augment  # noqa: E402

# PROVISIONAL (standing rule): sized so exp20 (16.4M skill tokens) sees
# each pool sentence ~3-4x instead of ~49x. Revisit with Week 3 numbers.
N_TARGET_PLACEHOLDER = 500_000  # TODO: tune once pilot dose is fixed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=N_TARGET_PLACEHOLDER)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--commit", action="store_true",
                    help="rebuild syntax.bin (default: dry run)")
    ap.add_argument("--no-embedded", action="store_true",
                    help="matrix recombination only (pool v1: embedded "
                         "recombination regressed — clause-template "
                         "memorization + complementizer-prior flooding)")
    args = ap.parse_args()

    syn_dir = DATA_DIR / "skills" / "syntax"
    held = (syn_dir / "held_lines.txt").read_text(encoding="utf-8").splitlines()

    blimp_sentences: list[str] = []
    for name in syntax.BLIMP_PARADIGMS:
        for good, bad in syntax.load_blimp_pairs(DATA_DIR / "blimp" / f"{name}.jsonl"):
            blimp_sentences += [good, bad]

    generated, st, holdout = syntax_augment.generate(
        held, blimp_sentences, n_target=args.n_target, seed=args.seed
    )
    if args.no_embedded:
        gen_embed, st_e, holdout_e = [], None, None
    else:
        seen = {syntax_augment._normalize(l) for l in held}
        seen |= {syntax_augment._normalize(s) for s in generated}
        for k in ("eval_trained", "eval_heldout"):
            seen |= {syntax_augment._normalize(r["sentence"])
                     for r in holdout[k]}
        gen_embed, st_e, holdout_e = syntax_augment.generate_embedded(
            held, blimp_sentences, n_target=args.n_target, seed=args.seed,
            seen=seen,
        )

    print(f"held lines scanned:      {st.lines_scanned:,}")
    print(f"frames extracted:        {st.frames_extracted:,}")
    print(f"candidate combinations:  {st.candidates_raw:,}")
    print(f"generated (kept):        {st.generated:,}")
    ld_pct = 100 * st.generated_long_distance / max(1, st.generated)
    print(f"  long-distance:         {st.generated_long_distance:,} ({ld_pct:.1f}%)")
    print(f"dropped: dup-original {st.dropped_dup_original:,}, "
          f"dup-blimp {st.dropped_dup_blimp:,}, "
          f"failed-classify {st.dropped_classify:,}")
    top = sorted(st.groups.items(), key=lambda kv: -kv[1])[:8]
    print("largest groups: " + ", ".join(f"{k}={v:,}" for k, v in top))
    print(f"frame holdout: {holdout['n_tails_heldout']:,}/"
          f"{holdout['n_tails_total']:,} tails held out; "
          f"{len(holdout['exclude_line_idxs']):,} source lines excluded "
          f"from pool; eval sets trained={len(holdout['eval_trained'])} "
          f"heldout={len(holdout['eval_heldout'])}")
    if st_e is not None:
        ld_e = 100 * st_e.generated_long_distance / max(1, st_e.generated)
        print(f"embedded: {st_e.frames_extracted:,} frames -> "
              f"{st_e.generated:,} generated ({ld_e:.1f}% LD) from "
              f"{st_e.candidates_raw:,} prefix x clause candidates")
        print(f"embedded clause holdout: {holdout_e['n_clauses_heldout']:,}/"
              f"{holdout_e['n_clauses_total']:,} clauses; "
              f"{len(holdout_e['exclude_line_idxs']):,} source lines "
              f"excluded; eval sets trained={len(holdout_e['eval_trained'])} "
              f"heldout={len(holdout_e['eval_heldout'])}")

    rng = __import__("random").Random(args.seed + 1)
    half = args.sample // 2 if gen_embed else args.sample
    sample = rng.sample(generated, min(half, len(generated))) \
        + (rng.sample(gen_embed, min(args.sample - half, len(gen_embed)))
           if gen_embed else [])
    sample_path = LOG_DIR / "syntax_augment_sample.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("\n".join(sample) + "\n", encoding="utf-8")
    print(f"\nreview sample ({len(sample)}) -> {sample_path}")

    if not args.commit:
        print("DRY RUN — no data files changed. Re-run with --commit "
              "after the sample passes review.")
        return

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    excluded = set(holdout["exclude_line_idxs"])
    if holdout_e is not None:
        excluded |= set(holdout_e["exclude_line_idxs"])
    originals = [l for i, l in enumerate(held) if i not in excluded]
    pool = [l + "\n" for l in originals] \
        + [s + "\n" for s in generated] + [s + "\n" for s in gen_embed]
    rng2 = __import__("random").Random(args.seed + 2)
    rng2.shuffle(pool)
    ids: list[int] = []
    chunk = 20_000
    for i in range(0, len(pool), chunk):
        for enc in tok.encode_batch(pool[i: i + chunk]):
            ids.extend(enc.ids)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(syn_dir / "syntax.bin")

    eval_path = syn_dir / "frame_generalization_eval.json"
    eval_path.write_text(json.dumps({
        "note": ("Structural-leak tests for the recombined pool, one per "
                 "recombination level. In each: eval_trained = unseen "
                 "recombinations of frames the model trained on; "
                 "eval_heldout = recombinations of frames excluded from "
                 "training entirely (source lines dropped from the pool). "
                 "Trained >> heldout = skeleton memorization at that level; "
                 "small gap = construction generalized."),
        "matrix": holdout,
        **({"embedded": holdout_e} if holdout_e is not None else {}),
    }, indent=2))
    print(f"wrote {eval_path}")

    meta = {
        "n_original_lines": len(held),
        "n_original_lines_excluded_for_holdout": len(excluded),
        "n_generated_matrix": st.generated,
        "n_generated_embedded": st_e.generated if st_e else 0,
        "n_generated_long_distance": (
            st.generated_long_distance
            + (st_e.generated_long_distance if st_e else 0)),
        "n_target": args.n_target,
        "seed": args.seed,
        "pool_tokens": int(arr.size),
        "note": ("syntax.bin = originals + frame-recombined variants, "
                 "shuffled. Runs on the pre-augment pool are not "
                 "comparable with runs on this one."),
    }
    (syn_dir / "augment_meta.json").write_text(json.dumps(meta, indent=2))

    stats_path = syn_dir / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["augment"] = meta
    stats["syntax_tokens_augmented"] = int(arr.size)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"committed: syntax.bin rebuilt — {arr.size:,} tokens "
          f"({len(pool):,} lines)")


if __name__ == "__main__":
    main()

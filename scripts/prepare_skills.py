"""Materialize skill datasets (build priority #2).

1. Arithmetic: held-out eval set (data/skills/arith/eval.jsonl) + tokenized
   training stream (train.bin), after verifying the BPE gives clean grading
   boundaries at '='.
2. Syntax: split BabyLM train lines into base_nosyntax.bin (early stream)
   and syntax.bin (held-out wh-object-gap lines, injected later by the
   delay scheduler), with stats.json documenting what was held.
3. BLiMP: download the wh/filler-gap paradigms for skill evaluation.

Run:  python scripts/prepare_skills.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import DATA_DIR  # noqa: E402
from reopen.skills import arithmetic as arith  # noqa: E402
from reopen.skills import syntax  # noqa: E402

# PROVISIONAL (standing rule): the arithmetic token budget interacts with
# the matched-total-exposure design decided in Week 3; this default just
# needs to be "plenty" for pilot runs.
ARITH_TRAIN_PROBLEMS_PLACEHOLDER = 200_000  # TODO: tune with Week 3 design
ARITH_EVAL_PROBLEMS = 1_000


def prepare_arith(tok, n_train: int, n_eval: int) -> None:
    out = DATA_DIR / "skills" / "arith"
    out.mkdir(parents=True, exist_ok=True)
    cfg = arith.ArithConfig()  # 2-4 digits, balanced carry mix, seed 1

    print("verifying tokenizer grading boundaries on 10K samples ...")
    arith.verify_tokenizer_boundaries(tok, cfg)

    eval_set = arith.make_eval_set(cfg, n_eval)
    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for a, b in eval_set:
            f.write(json.dumps(
                {"a": a, "b": b, "sum": a + b, "carry": arith.has_carry(a, b)}
            ) + "\n")

    n_tokens = arith.write_train_bin(
        tok, cfg, n_train, exclude=set(eval_set), out_path=out / "train.bin"
    )
    meta = {
        "config": cfg.__dict__,
        "n_train_problems": n_train,
        "n_train_tokens": n_tokens,
        "n_eval_problems": len(eval_set),
        "dtype": "uint16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"arith: {n_train:,} problems -> {n_tokens:,} tokens, "
          f"{len(eval_set)} held-out eval problems")


def _encode_lines_to_bin(tok, lines: list[str], out_path: Path) -> int:
    ids: list[int] = []
    chunk = 20_000
    for i in range(0, len(lines), chunk):
        for enc in tok.encode_batch(lines[i : i + chunk]):
            ids.extend(enc.ids)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out_path)
    return len(arr)


def prepare_syntax(tok) -> None:
    out = DATA_DIR / "skills" / "syntax"
    out.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / "raw" / "babylm_data" / "babylm_10M"
    files = sorted(src.glob("*.train"))
    assert files, f"no .train files under {src}"

    print(f"filtering {len(files)} BabyLM files for wh-object-gap lines ...")
    kept, held, stats = syntax.filter_corpus_files(files)
    (out / "held_lines.txt").write_text("".join(held), encoding="utf-8")

    n_base = _encode_lines_to_bin(tok, kept, out / "base_nosyntax.bin")
    n_held = _encode_lines_to_bin(tok, held, out / "syntax.bin")
    totals = {
        "kept_lines": len(kept),
        "held_lines": len(held),
        "base_nosyntax_tokens": n_base,
        "syntax_tokens": n_held,
        "held_pct_lines": round(100 * len(held) / (len(kept) + len(held)), 2),
        "per_file": stats,
    }
    (out / "stats.json").write_text(json.dumps(totals, indent=2))
    print(f"syntax: held {len(held):,} lines "
          f"({totals['held_pct_lines']}%) = {n_held:,} tokens; "
          f"base stream {n_base:,} tokens")


def prepare_vocab(tok) -> None:
    """Negative control (see reopen/skills/vocab_frequency.py docstring:
    this skill is EXPECTED to show no critical period — a null result
    validates the method)."""
    from reopen.skills import vocab_frequency as vocab

    out = DATA_DIR / "skills" / "vocab"
    out.mkdir(parents=True, exist_ok=True)
    src = DATA_DIR / "raw" / "babylm_data" / "babylm_10M"
    files = sorted(src.glob("*.train"))
    assert files, f"no .train files under {src}"

    cfg = vocab.VocabControlConfig()
    print("counting word frequencies over the train corpus ...")
    counts, cap_counts = vocab.count_word_frequencies(files)
    targets = vocab.select_target_words(counts, cfg, cap_counts)
    fam_set = vocab.build_family_set(targets)

    print(f"scrubbing {len(targets)} target-word families from corpus ...")
    kept, held, stats = vocab.filter_corpus_files(files, fam_set)
    instances, reinject = vocab.make_eval_instances(held, targets, cfg)

    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")
    n_base = _encode_lines_to_bin(tok, kept, out / "base_novocab.bin")
    n_skill = _encode_lines_to_bin(tok, reinject, out / "vocab.bin")

    (out / "target_words.json").write_text(json.dumps({
        "note": ("NEGATIVE CONTROL: no critical period expected; a null "
                 "result here means the method works"),
        "config": cfg.__dict__,
        "targets": {w: counts[w] for w in targets},
    }, indent=2))
    totals = {
        "kept_lines": len(kept),
        "held_lines": len(held),
        "eval_instances": len(instances),
        "reinjected_lines": len(reinject),
        "base_novocab_tokens": n_base,
        "vocab_tokens": n_skill,
        "held_pct_lines": round(100 * len(held) / (len(kept) + len(held)), 2),
        "per_file": stats,
    }
    (out / "stats.json").write_text(json.dumps(totals, indent=2))
    print(f"vocab: {len(targets)} words, held {len(held):,} lines "
          f"({totals['held_pct_lines']}%), {len(instances)} eval instances "
          f"(excluded from reinjection), skill stream {n_skill:,} tokens")


def prepare_blimp() -> None:
    out = DATA_DIR / "blimp"
    out.mkdir(parents=True, exist_ok=True)
    for name in syntax.BLIMP_PARADIGMS:
        dest = out / f"{name}.jsonl"
        if dest.exists():
            print(f"blimp: {name} already present")
            continue
        url = syntax.BLIMP_RAW_URL.format(name)
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        n = len(dest.read_text(encoding="utf-8").splitlines())
        print(f"blimp: {name} ({n} pairs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arith-problems", type=int,
                    default=ARITH_TRAIN_PROBLEMS_PLACEHOLDER)
    ap.add_argument("--arith-eval", type=int, default=ARITH_EVAL_PROBLEMS)
    ap.add_argument("--skip-arith", action="store_true")
    ap.add_argument("--skip-syntax", action="store_true")
    ap.add_argument("--skip-vocab", action="store_true")
    ap.add_argument("--skip-blimp", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(DATA_DIR / "tokenized" / "tokenizer.json"))
    if not args.skip_arith:
        prepare_arith(tok, args.arith_problems, args.arith_eval)
    if not args.skip_syntax:
        prepare_syntax(tok)
    if not args.skip_vocab:
        prepare_vocab(tok)
    if not args.skip_blimp:
        prepare_blimp()
    print("done")


if __name__ == "__main__":
    main()

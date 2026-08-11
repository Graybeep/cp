"""Prepare the BabyLM strict-small (10M-word) corpus:

1. Extract data/raw/babylm_data.zip (the official babylm_data release:
   babylm_10M train split + babylm_dev split)
2. Train a byte-level BPE tokenizer on the 10M train split only
   (locked spec: custom BPE on the corpus itself, vocab ~8K-16K)
3. Encode train -> data/tokenized/train.bin, dev -> val.bin as flat
   uint16 arrays (memmap-ready), with <|endoftext|> between source files
4. Write meta.json (vocab size, token counts) — consumed by reopen.data

Run:  python scripts/prepare_babylm.py  [--vocab-size 16384]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import DATA_DIR  # noqa: E402

RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "tokenized"
EOT = "<|endoftext|>"


def extract(zip_path: Path) -> Path:
    root = RAW_DIR / "babylm_data"
    if not root.exists():
        print(f"extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
    # The zip may nest everything under a babylm_data/ folder already.
    if not (root / "babylm_10M").exists():
        candidates = list(RAW_DIR.glob("**/babylm_10M"))
        if not candidates:
            sys.exit("babylm_10M folder not found after extraction")
        root = candidates[0].parent
    return root


def train_tokenizer(train_files: list[Path], vocab_size: int) -> "Tokenizer":
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    print(f"training BPE (vocab {vocab_size}) on {len(train_files)} files ...")
    tok.train(
        files=[str(f) for f in train_files],
        vocab_size=vocab_size,
        special_tokens=[EOT],
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(OUT_DIR / "tokenizer.json"))
    return tok


def encode_split(tok, files: list[Path], out_path: Path, eot_id: int) -> int:
    ids: list[int] = []
    for f in sorted(files):
        text = f.read_text(encoding="utf-8")
        # keep newlines (line structure matters for dialogue corpora);
        # encode_batch in chunks of lines to keep memory bounded
        lines = [ln + "\n" for ln in text.splitlines()]
        chunk = 20_000
        for i in range(0, len(lines), chunk):
            for enc in tok.encode_batch(lines[i : i + chunk]):
                ids.extend(enc.ids)
        ids.append(eot_id)  # separator between source files
        print(f"  {f.name}: running total {len(ids):,} tokens")
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out_path)
    return len(arr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--zip", type=Path, default=RAW_DIR / "babylm_data.zip")
    args = ap.parse_args()

    root = extract(args.zip)
    train_files = sorted((root / "babylm_10M").glob("*.train"))
    dev_files = sorted((root / "babylm_dev").glob("*.dev"))
    assert train_files and dev_files, "missing train/dev files"

    tok = train_tokenizer(train_files, args.vocab_size)
    eot_id = tok.token_to_id(EOT)

    print("encoding train split ...")
    n_train = encode_split(tok, train_files, OUT_DIR / "train.bin", eot_id)
    print("encoding dev split (val) ...")
    n_val = encode_split(tok, dev_files, OUT_DIR / "val.bin", eot_id)

    meta = {
        "vocab_size": args.vocab_size,
        "eot_id": eot_id,
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "train_files": [f.name for f in train_files],
        "val_files": [f.name for f in dev_files],
        "dtype": "uint16",
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done: {n_train:,} train tokens, {n_val:,} val tokens -> {OUT_DIR}")


if __name__ == "__main__":
    main()

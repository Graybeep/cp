"""Synthetic arithmetic skill: 2-4 digit addition, carrying as the
difficulty axis. Infinite supply, perfectly gradeable.

Data format (one problem per line, packed into the token stream):
    347+285=632\n

Grading is batched teacher-forced exact match: the model's argmax at every
answer-token position (given the correct prefix) must equal the answer
token. By induction this is identical to greedy-decode exact match, but it
needs a single forward pass per batch instead of a token-by-token loop.

Tokenization caveat handled here: the BPE was trained on BabyLM, so token
boundaries inside "a+b=c" are corpus-dependent. Grading uses the
tokenizer's char offsets and asserts a token boundary falls exactly after
'=' — verify_tokenizer_boundaries() should be run once at data-prep time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

CarryMode = str  # "none" | "required" | "mixed"


@dataclass
class ArithConfig:
    min_digits: int = 2
    max_digits: int = 4
    carrying: CarryMode = "mixed"  # "mixed" = balanced 50/50 none/required
    seed: int = 1
    # Grading prompts are prefixed with "\n" to match the training stream,
    # where every problem (except a block-initial one) follows a newline.
    prompt_prefix: str = "\n"


def has_carry(a: int, b: int) -> bool:
    """True if column-wise addition of a+b produces any carry."""
    carry = 0
    while a > 0 or b > 0:
        s = (a % 10) + (b % 10) + carry
        if s >= 10:
            return True
        carry = 0  # s < 10 means no carry out
        a //= 10
        b //= 10
    return False


def sample_problem(rng: random.Random, cfg: ArithConfig,
                   force_carry: bool | None = None) -> tuple[int, int]:
    """Uniform over operand pairs in the digit range, rejection-sampled to
    the requested carry class. Worst case (no-carry, 4+4 digits) accepts
    ~9% of draws — microseconds per problem, unbiased within the class."""
    if force_carry is None:
        if cfg.carrying == "none":
            force_carry = False
        elif cfg.carrying == "required":
            force_carry = True
        else:  # mixed: balanced difficulty axis
            force_carry = rng.random() < 0.5
    while True:
        da = rng.randint(cfg.min_digits, cfg.max_digits)
        db = rng.randint(cfg.min_digits, cfg.max_digits)
        a = rng.randint(10 ** (da - 1), 10 ** da - 1)
        b = rng.randint(10 ** (db - 1), 10 ** db - 1)
        if has_carry(a, b) == force_carry:
            return a, b


def format_problem(a: int, b: int) -> str:
    return f"{a}+{b}={a + b}"


def make_eval_set(cfg: ArithConfig, n: int) -> list[tuple[int, int]]:
    """Unique held-out problems, balanced across carry classes for
    cfg.carrying == "mixed". Deterministic under cfg.seed."""
    rng = random.Random(cfg.seed + 1_000_003)  # offset: never reuse train rng
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    i = 0
    while len(out) < n:
        force = (i % 2 == 0) if cfg.carrying == "mixed" else None
        p = sample_problem(rng, cfg, force_carry=force)
        i += 1
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def iter_train_lines(cfg: ArithConfig, n_problems: int,
                     exclude: set[tuple[int, int]]) -> Iterable[str]:
    """Training problems, skipping the held-out eval pairs exactly
    (duplicates among train problems are allowed — natural for the small
    2-digit space)."""
    rng = random.Random(cfg.seed)
    made = 0
    while made < n_problems:
        p = sample_problem(rng, cfg)
        if p in exclude:
            continue
        made += 1
        yield format_problem(*p) + "\n"


def write_train_bin(tok, cfg: ArithConfig, n_problems: int,
                    exclude: set[tuple[int, int]], out_path) -> int:
    """Tokenize the training stream to a uint16 memmap-ready file."""
    ids: list[int] = []
    buf: list[str] = []

    def flush():
        nonlocal ids
        if buf:
            for enc in tok.encode_batch(buf):
                ids.extend(enc.ids)
            buf.clear()

    for line in iter_train_lines(cfg, n_problems, exclude):
        buf.append(line)
        if len(buf) >= 20_000:
            flush()
    flush()
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out_path)
    return len(arr)


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def _answer_span(tok, prompt: str, full: str) -> tuple[list[int], int]:
    """Tokenize `full`, return (ids, index of first answer token). Raises if
    no token boundary falls exactly at the end of the prompt."""
    enc = tok.encode(full)
    eq_end = len(prompt)  # char index where the answer starts
    for ti, (start, _end) in enumerate(enc.offsets):
        if start == eq_end:
            return enc.ids, ti
        if start > eq_end:
            break
    raise ValueError(
        f"BPE merges a token across the '=' boundary in {full!r}; "
        "grading span is ambiguous — check tokenizer/data format"
    )


def verify_tokenizer_boundaries(tok, cfg: ArithConfig, n: int = 10_000) -> None:
    """Run at data-prep time: confirm '=' is always a clean token boundary
    so exact-match grading is well-defined for this tokenizer."""
    rng = random.Random(cfg.seed + 42)
    for _ in range(n):
        a, b = sample_problem(rng, cfg)
        prompt = f"{cfg.prompt_prefix}{a}+{b}="
        _answer_span(tok, prompt, prompt + str(a + b))


@torch.no_grad()
def grade_exact_match(
    model,
    tok,
    problems: Sequence[tuple[int, int]],
    device: str = "cuda",
    batch_size: int = 64,
    prompt_prefix: str = "\n",
) -> float:
    """Fraction of problems where every answer token is the argmax given
    the correct prefix (== greedy-decode exact match)."""
    model.eval()
    correct = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        rows: list[list[int]] = []
        spans: list[tuple[int, int]] = []  # (first answer token idx, n answer tokens)
        for a, b in batch:
            prompt = f"{prompt_prefix}{a}+{b}="
            ids, ans_start = _answer_span(tok, prompt, prompt + str(a + b))
            rows.append(ids)
            spans.append((ans_start, len(ids) - ans_start))
        maxlen = max(len(r) for r in rows)
        # Right-padding is safe under causal attention: positions after a
        # row's answer never influence the graded logits.
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        for r, ids in enumerate(rows):
            x[r, : len(ids)] = torch.tensor(ids)
        logits, _ = model(x.to(device))
        preds = logits.argmax(-1).cpu()
        for r, (ans_start, n_ans) in enumerate(spans):
            ok = all(
                preds[r, ans_start - 1 + j] == rows[r][ans_start + j]
                for j in range(n_ans)
            )
            correct += int(ok)
    return correct / len(problems)

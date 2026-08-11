"""Vocabulary-frequency skill — the NEGATIVE CONTROL.

===========================================================================
FRAMING — READ BEFORE INTERPRETING RESULTS:
This component's job is to show NO critical period. Word learning is
believed to remain plastic throughout training, so a delayed word band
should catch up to the early-exposure baseline once injected.

  - A NULL result here (no delay-dependent gap) means the METHOD IS
    WORKING — it demonstrates that the delay/reinjection machinery itself
    doesn't manufacture artificial gaps. Do not mistake it for a bug.
  - A POSITIVE result here (vocab showing a "critical period") flags a
    problem with the method — a confound in the pipeline — NOT a
    discovery about vocabulary.
===========================================================================

Mechanics mirror the syntax skill exactly (same filter-and-reinject
machinery, same delay scheduler, same matched-exposure accounting):

- Target words: ~100-200 mid-to-low frequency word types chosen by
  frequency WITHIN the BabyLM corpus itself (not an external list), so
  the band is consistent with the tokenizer trained on this corpus.
  Most-common words are excluded (they appear too early/often to
  meaningfully delay); ultra-rare words are excluded (too few occurrences
  to measure reliably). "Lemma" is approximated without a parser: only
  base-form-looking types are candidates (no -s/-ed/-ing/-ly endings) and
  each target is scrubbed together with its naive inflection family
  (+s/+es/+ed/+ing, e-drop and consonant-doubling variants). Irregular
  forms (mouse/mice) are not chased — documented limitation.

- Scrubbing: every line containing any target-family word is removed from
  the early stream and held for reinjection on the delay schedule.

- Scoring: cloze-style, NOT raw perplexity — given the held-out context
  up to a target-word occurrence, the model must recover the word itself
  (teacher-forced argmax over the word's tokens == greedy exact match,
  same trick as arithmetic grading). This isolates whether those
  particular words were learned, rather than general sentence fluency.
  Eval contexts come from scrubbed lines that are NEVER reinjected, so
  neither condition can memorize the exact eval sentences.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

_WORD_RE = re.compile(r"[a-z]+")
_TOKEN_RE = re.compile(r"[A-Za-z]+")
# candidate base forms must not look inflected/derived
_INFLECTED_SUFFIXES = ("s", "ed", "ing", "ly")
# words capitalized in most of their occurrences are proper nouns
# (Alaska, Allah, Alzheimer ...) — names, not vocabulary lemmas
_PROPER_NOUN_CAP_RATIO = 0.5


@dataclass
class VocabControlConfig:
    n_words: int = 150
    # PROVISIONAL band (standing-rule spirit: one place, tune with pilot):
    # occurrence counts within the 10M-word train split. 100-500 sits in
    # the mid-to-low band — measurable but not omnipresent.
    min_count: int = 100
    max_count: int = 500
    min_word_len: int = 4
    min_left_words: int = 5     # left-context words required for a cloze
    max_context_chars: int = 400  # truncate long contexts (token budget)
    # Bumped 1000/10 -> 3000/25 at the Week 3 exit gate (2026-07-17): the
    # per-word signal was strong (median SNR ~12.6, every word improving)
    # but instance counts were thin (median 7, min 1 per word) — the cheap
    # branch of the gate. Regenerates eval.jsonl + vocab.bin; pilot is
    # unaffected (no pilot run trained on vocab.bin), grid runs must all
    # use the regenerated files.
    n_eval: int = 3000
    max_eval_per_word: int = 25
    seed: int = 1


def inflection_family(word: str) -> set[str]:
    """Naive inflection family of a base form. Overgenerates harmlessly
    (nonexistent forms simply never match)."""
    fam = {word, word + "s"}
    if word.endswith("e"):
        fam |= {word + "d", word[:-1] + "ing"}
    else:
        last = word[-1]
        fam |= {
            word + "es", word + "ed", word + "ing",
            word + last + "ed", word + last + "ing",  # stop -> stopped
        }
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        fam |= {word[:-1] + "ies", word[:-1] + "ied"}  # carry -> carries
    return fam


def count_word_frequencies(files: Sequence[Path]) -> tuple[Counter, Counter]:
    """(lowercased counts, counts of capitalized occurrences)."""
    counts: Counter = Counter()
    cap_counts: Counter = Counter()
    for f in files:
        for t in _TOKEN_RE.findall(f.read_text(encoding="utf-8")):
            counts[t.lower()] += 1
            if t[0].isupper():
                cap_counts[t.lower()] += 1
    return counts, cap_counts


def select_target_words(counts: Counter, cfg: VocabControlConfig,
                        cap_counts: Counter | None = None) -> list[str]:
    """Deterministic sample of base-form-looking types inside the
    frequency band. Families must not overlap (a word may not be another
    target's inflected form); majority-capitalized types are treated as
    proper nouns and excluded — names are not vocabulary lemmas."""
    cap_counts = cap_counts or Counter()
    candidates = sorted(
        w for w, c in counts.items()
        if cfg.min_count <= c <= cfg.max_count
        and len(w) >= cfg.min_word_len
        and w.isalpha()
        and not any(w.endswith(s) for s in _INFLECTED_SUFFIXES)
        and cap_counts[w] / c <= _PROPER_NOUN_CAP_RATIO
    )
    rng = random.Random(cfg.seed + 7_000_003)
    rng.shuffle(candidates)
    chosen: list[str] = []
    claimed: set[str] = set()
    for w in candidates:
        fam = inflection_family(w)
        if fam & claimed:
            continue
        chosen.append(w)
        claimed |= fam
        if len(chosen) == cfg.n_words:
            break
    assert len(chosen) == cfg.n_words, (
        f"only {len(chosen)} candidates in band "
        f"[{cfg.min_count}, {cfg.max_count}] — widen the band"
    )
    return sorted(chosen)


def build_family_set(targets: Sequence[str]) -> set[str]:
    fam: set[str] = set()
    for w in targets:
        fam |= inflection_family(w)
    return fam


def line_contains_target(line: str, family_set: set[str]) -> bool:
    return any(w in family_set for w in _WORD_RE.findall(line.lower()))


def filter_corpus_files(
    files: Sequence[Path], family_set: set[str]
) -> tuple[list[str], list[str], dict]:
    """Same shape as syntax.filter_corpus_files: (kept, held, stats)."""
    kept: list[str] = []
    held: list[str] = []
    stats: dict = {}
    for f in sorted(files):
        n_held = 0
        n_total = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            n_total += 1
            if line_contains_target(line, family_set):
                held.append(line + "\n")
                n_held += 1
            else:
                kept.append(line + "\n")
        stats[f.name] = {
            "lines": n_total,
            "held": n_held,
            "held_pct": round(100 * n_held / max(1, n_total), 2),
        }
    return kept, held, stats


# ---------------------------------------------------------------------------
# Cloze eval-set construction
# ---------------------------------------------------------------------------

def make_eval_instances(
    held_lines: Sequence[str],
    targets: Sequence[str],
    cfg: VocabControlConfig,
) -> tuple[list[dict], list[str]]:
    """Sample cloze instances from held lines; returns (instances,
    remaining_lines). Lines used for eval are EXCLUDED from the returned
    reinjection stream, so eval sentences are trained on by no condition."""
    fam_to_base = {}
    for w in targets:
        for form in inflection_family(w):
            fam_to_base[form] = w
    pattern = re.compile(
        r"\b(" + "|".join(sorted(fam_to_base, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    rng = random.Random(cfg.seed + 9_000_017)
    order = list(range(len(held_lines)))
    rng.shuffle(order)

    instances: list[dict] = []
    per_word: Counter = Counter()
    used_lines: set[int] = set()
    for i in order:
        if len(instances) >= cfg.n_eval:
            break
        line = held_lines[i].rstrip("\n")
        m = pattern.search(line)
        if not m:
            continue
        # capitalized surface form = proper-noun usage ("United Artists"
        # matching unit+ed), not the vocabulary lemma — never eval on it
        if m.group(1)[0].isupper():
            continue
        base = fam_to_base[m.group(1).lower()]
        if per_word[base] >= cfg.max_eval_per_word:
            continue
        left = line[: m.start()]
        if len(_WORD_RE.findall(left.lower())) < cfg.min_left_words:
            continue
        left = left[-cfg.max_context_chars:]
        instances.append({
            "context": left,           # text before the word (with space)
            "target": line[m.start(): m.end()],  # surface form, cased
            "word_base": base,
        })
        per_word[base] += 1
        used_lines.add(i)
    remaining = [held_lines[i] for i in range(len(held_lines))
                 if i not in used_lines]
    return instances, remaining


# ---------------------------------------------------------------------------
# Cloze scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def cloze_scores(
    model,
    tok,
    instances: Sequence[dict],
    device: str = "cuda",
    batch_size: int = 64,
    per_word: bool = False,
) -> tuple[float, float] | tuple[float, float, dict]:
    """(exact-match accuracy, mean target log-prob) over cloze instances;
    with per_word=True also returns {word_base: {n, acc, mean_logprob}}.

    THIN-SIGNAL CAVEAT for interpreting the control: ~1000 instances give
    the aggregate accuracy a binomial SE of ~1.6%, but per-word counts are
    small (~7 at default config). When judging "no critical period", use
    the per-word breakdown to distinguish a genuinely flat curve from
    noise averaging out, and prefer mean log-prob (continuous, denser
    signal) over exact match while accuracy is near floor.

    Exact match: model's argmax at every target-token position given the
    correct prefix (== greedy decode). The ByteLevel pre-tokenizer never
    merges across a space, so the boundary before ' word' is always a
    clean token boundary; asserted defensively anyway.

    NEGATIVE-CONTROL REMINDER: across delay conditions this metric is
    EXPECTED to show no persistent gap. That null result validates the
    method; it is not a failed experiment.
    """
    model.eval()
    n_correct = 0
    logprobs: list[float] = []
    by_word: dict[str, list[tuple[int, float]]] = {}
    for i in range(0, len(instances), batch_size):
        batch = instances[i : i + batch_size]
        rows: list[list[int]] = []
        spans: list[tuple[int, int]] = []
        for inst in batch:
            context = inst["context"]
            full = context + inst["target"]
            # ByteLevel tokens carry their leading space, so when the
            # context ends with one, the target's first token starts at
            # that space — the boundary sits one char earlier.
            boundary = len(context) - 1 if context.endswith(" ") \
                else len(context)
            enc = tok.encode(full)
            start_tok = None
            for ti, (s, _e) in enumerate(enc.offsets):
                if s == boundary:
                    start_tok = ti
                    break
                if s > boundary:
                    break
            assert start_tok is not None and start_tok > 0, (
                f"no clean token boundary before target in {full!r}"
            )
            rows.append(enc.ids)
            spans.append((start_tok, len(enc.ids) - start_tok))
        maxlen = max(len(r) for r in rows)
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        for r, ids in enumerate(rows):
            x[r, : len(ids)] = torch.tensor(ids)
        logits, _ = model(x.to(device))
        logp = torch.log_softmax(logits.float(), dim=-1).cpu()
        preds = logp.argmax(-1)
        for r, (st, n_t) in enumerate(spans):
            tgt = rows[r][st : st + n_t]
            ok = all(preds[r, st - 1 + j] == tgt[j] for j in range(n_t))
            n_correct += int(ok)
            lp = sum(logp[r, st - 1 + j, tgt[j]].item() for j in range(n_t))
            logprobs.append(lp)
            by_word.setdefault(batch[r]["word_base"], []).append((int(ok), lp))
    acc = n_correct / len(instances)
    mean_lp = sum(logprobs) / len(logprobs)
    if not per_word:
        return acc, mean_lp
    breakdown = {
        w: {
            "n": len(v),
            "acc": sum(ok for ok, _ in v) / len(v),
            "mean_logprob": sum(lp for _, lp in v) / len(v),
        }
        for w, v in sorted(by_word.items())
    }
    return acc, mean_lp, breakdown

"""Syntax skill: long-distance wh-question extraction (object-gap).

Two halves:

1. CORPUS FILTERING (training side): detect lines in BabyLM containing the
   target construction so the delay scheduler can withhold them from the
   early stream and inject them later. The detector is a regex heuristic —
   recall matters more than precision here, because a target sentence that
   LEAKS into the early stream contaminates the delay condition, while an
   over-removed non-target sentence merely shrinks the base corpus
   slightly. Filtering is line-level (the corpora are line-structured;
   a long paragraph containing one wh-question is held out whole, which is
   conservative in the safe direction).

   Detected as targets:
   - Matrix object-gap wh-questions:      "What did you see _?"
     (wh + inverted auxiliary + subject + verb — the aux-inversion pattern
     distinguishes these from subject-gap "Who saw you?", which stays in)
   - Embedded wh-complements:             "I wonder what she bought _."
   - Long-distance flag: a bridge verb after the subject
                                          "What do you think she saw _?"

   Known imprecision (documented, acceptable): copular questions like
   "What is your name?" match the matrix pattern. They are removed from
   the early stream too — safe direction.

2. EVALUATION (BLiMP): forced-choice minimal pairs — accuracy = fraction
   of pairs where the model assigns higher total log-probability to the
   grammatical sentence. Paradigms downloaded by scripts/prepare_skills.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

import torch

# wh-words that extract arguments (object gaps). where/when/why/how are
# adjunct questions — deliberately excluded.
_WH = r"(?:what|who|whom|which\s+\w+|whose\s+\w+)"
_WH_PRON = r"(?:what|who|whom)"
_WH_DET = r"(?:which\s+\w+|whose\s+\w+)"

# Matrix pattern: wh + inverted aux + subject + verb. Precision guards
# (each validated by spot-checking real BabyLM output):
#
# 1. Anchor: the wh-word must open a sentence/quote (optionally after a
#    discourse opener: "So what did you do?"). Mid-sentence wh-words are
#    almost always relative clauses ("the man who did it", "things which
#    may be happening"). After a comma, only wh-PRONOUNS qualify —
#    ", which was old" is a non-restrictive relative, while ", what did
#    you see?" is a real question.
# 2. Aux split: after do-support/modal auxes the next word is always the
#    subject, but after be/have the next word may be the verb itself —
#    "who was holding the hand" (subject gap, excluded) vs "what was she
#    doing" (object gap, kept). So the be/have branch rejects V-ing /
#    V-ed / V-en / common irregular participles right after the aux.
_OPENER = r"(?:(?:so|and|but|then|well|now|oh|hey|um|uh|yeah|ok|okay)\s+)?"
_LEAD_HARD = rf"(?:^|[.!?:;\"'()\[\]])[\s\"'()\[\]]*{_OPENER}"
_LEAD_COMMA = rf",\s*{_OPENER}"
_AUX_DO = r"(?:do|does|did|can|could|will|would|shall|should|may|might|must)"
_AUX_BH = r"(?:is|are|was|were|am|have|has|had)"
_NOT_VFORM = (
    r"(?!\w+ing\b|\w\w+ed\b|\w\w+en\b"
    r"|(?:said|done|made|gone|told|heard|kept|brought|thought|found"
    r"|left|put|set|shown|known|seen|built|sent|meant|held|worn)\b)"
)
MATRIX_WH_OBJ_RE = re.compile(
    rf"(?:{_LEAD_HARD}(?:{_WH_DET}|{_WH_PRON})|{_LEAD_COMMA}{_WH_PRON})"
    rf"\s+(?:{_AUX_DO}\s+\w+|{_AUX_BH}\s+{_NOT_VFORM}\w+)\s+\w+",
    re.IGNORECASE,
)
# embedded wh-complement: attitude/question verb + wh + clause
_EMBED_VERBS = (
    r"(?:wonder(?:s|ed|ing)?|know(?:s|ing)?|knew|ask(?:s|ed|ing)?|"
    r"tell(?:s|ing)?|told|remember(?:s|ed|ing)?|forget(?:s|ting)?|forgot|"
    r"guess(?:es|ed|ing)?|understand(?:s|ing)?|understood)"
)
EMBEDDED_WH_RE = re.compile(
    rf"\b{_EMBED_VERBS}\s+{_WH}\s+\w+\s+\w+", re.IGNORECASE
)
# bridge verbs signalling long-distance extraction across a clause boundary
BRIDGE_RE = re.compile(
    r"\b(?:think|thinks|thought|say|says|said|believe|believes|believed|"
    r"know|knows|knew|want|wants|wanted|hope|hopes|hoped|guess|guesses|"
    r"guessed|suppose|supposes|supposed|imagine|imagines|imagined|"
    r"remember|remembers|remembered|claim|claims|claimed|hear|hears|"
    r"heard|wish|wishes|wished|expect|expects|expected|mean|means|meant)\b",
    re.IGNORECASE,
)


_WH_RE = re.compile(_WH, re.IGNORECASE)


def _long_distance(line: str, search_from: int) -> bool:
    """Bridge verb after the wh-word signals extraction across a clause
    boundary. Search starts at the end of the wh-word itself (not the end
    of the full pattern match, which would consume the bridge verb in
    'Who did Mary say ...')."""
    whm = _WH_RE.search(line, search_from)
    if not whm:
        return False
    return bool(BRIDGE_RE.search(line, whm.end()))


def classify_line(line: str) -> dict:
    """{"target": bool, "long_distance": bool} for one corpus line."""
    m = MATRIX_WH_OBJ_RE.search(line)
    e = EMBEDDED_WH_RE.search(line)
    target = bool(m) or bool(e)
    long_distance = False
    if m:
        long_distance = _long_distance(line, m.start())
    if e and not long_distance:
        long_distance = _long_distance(line, e.start())
    return {"target": target, "long_distance": long_distance}


def filter_corpus_files(
    files: Sequence[Path],
) -> tuple[list[str], list[str], dict]:
    """Split corpus lines into (kept, held) + per-file stats.
    `held` = lines containing the target construction (to be delayed)."""
    kept: list[str] = []
    held: list[str] = []
    stats: dict = {}
    for f in sorted(files):
        n_held = 0
        n_ld = 0
        n_total = 0
        for line in f.read_text(encoding="utf-8").splitlines():
            n_total += 1
            c = classify_line(line)
            if c["target"]:
                held.append(line + "\n")
                n_held += 1
                n_ld += int(c["long_distance"])
            else:
                kept.append(line + "\n")
        stats[f.name] = {
            "lines": n_total,
            "held": n_held,
            "held_long_distance": n_ld,
            "held_pct": round(100 * n_held / max(1, n_total), 2),
        }
    return kept, held, stats


# ---------------------------------------------------------------------------
# BLiMP evaluation
# ---------------------------------------------------------------------------

BLIMP_PARADIGMS = [
    # the wh-licensing 2x2 (see wh_licensing_interaction below) — these
    # four are what the headline syntax metric is computed from
    "wh_vs_that_with_gap",
    "wh_vs_that_no_gap",
    "wh_vs_that_with_gap_long_distance",
    "wh_vs_that_no_gap_long_distance",
    # reported for reference; single-paradigm scores are frequency-confounded
    "wh_questions_object_gap",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
]

# The wh-licensing interaction pairs: (with_gap, no_gap) per distance.
# with_gap: good = wh-filler, bad = "that"   (a gap REQUIRES a filler)
# no_gap:   good = "that",    bad = wh-filler (no gap FORBIDS a filler)
LICENSING_PAIRS = {
    "local": ("wh_vs_that_with_gap", "wh_vs_that_no_gap"),
    "long_distance": ("wh_vs_that_with_gap_long_distance",
                      "wh_vs_that_no_gap_long_distance"),
}
BLIMP_RAW_URL = (
    "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/{}.jsonl"
)


def load_blimp_pairs(path: str | Path) -> list[tuple[str, str]]:
    pairs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        pairs.append((d["sentence_good"], d["sentence_bad"]))
    return pairs


@torch.no_grad()
def _sentence_logprobs(
    model, tok, sentences: Sequence[str], device: str, batch_size: int
) -> torch.Tensor:
    """Sum of log P(t_i | t_<i>) for i >= 1, per sentence."""
    out = torch.empty(len(sentences))
    for i in range(0, len(sentences), batch_size):
        encs = tok.encode_batch(list(sentences[i : i + batch_size]))
        rows = [e.ids for e in encs]
        maxlen = max(len(r) for r in rows)
        x = torch.zeros(len(rows), maxlen, dtype=torch.long)
        for r, ids in enumerate(rows):
            x[r, : len(ids)] = torch.tensor(ids)
        x = x.to(device)
        logits, _ = model(x)
        logp = torch.log_softmax(logits.float(), dim=-1)
        for r, ids in enumerate(rows):
            t = torch.tensor(ids[1:], device=device)
            out[i + r] = logp[r, torch.arange(len(t)), t].sum().cpu()
    return out


@torch.no_grad()
def blimp_accuracy(
    model, tok, pairs: Sequence[tuple[str, str]],
    device: str = "cuda", batch_size: int = 64,
) -> float:
    """Fraction of minimal pairs where the grammatical sentence gets the
    higher total log-probability (chance = 0.5).

    WARNING — do not use a single wh_vs_that paradigm as a skill metric:
    its pairs differ by one word ("what" vs "that"), so the score is
    dominated by the unigram frequency prior for "that", not by syntactic
    knowledge. Use wh_licensing_interaction() instead. (Measured on this
    project's step-100 model: with_gap 0.000, no_gap 1.000 — a pure
    frequency effect masquerading as near-perfect/near-zero syntax.)
    """
    model.eval()
    good = _sentence_logprobs(model, tok, [p[0] for p in pairs], device, batch_size)
    bad = _sentence_logprobs(model, tok, [p[1] for p in pairs], device, batch_size)
    return (good > bad).float().mean().item()


def wh_licensing_interaction(with_gap_acc: float, no_gap_acc: float) -> float:
    """Bias-corrected filler-gap metric (Wilcox-style 2x2 collapsed to the
    interaction term). Chance = 0.5, and CRUCIALLY chance is where a
    frequency-biased model with no syntax lands:

      A model that always prefers "that" scores 0.0 on with_gap (where the
      wh-filler is correct) and 1.0 on no_gap (where "that" is correct);
      the mean is exactly 0.5. The main effect of the wh-word cancels, and
      only the gap-licensing INTERACTION survives — i.e. does the model
      know that a downstream gap is what makes the filler obligatory.

    Verified empirically on this project's baseline: an untrained (step
    100) model scores exactly 0.500 by this metric while showing 0.000 /
    1.000 on the raw paradigms; by step 5000 it reaches 0.604 (local) and
    0.534 (long-distance) — real, if modest, filler-gap knowledge.
    """
    return (with_gap_acc + no_gap_acc) / 2

"""Frame recombination for the syntax skill pool.

WHY (2026-07-16): the extracted wh-object-gap pool is only 336K tokens, so
any exposure above ~5% recycles it into memorization — the exp20 probe
collapsed BOTH licensing interactions to chance and cost +7.7 val ppl
(train/val gap +1.12 vs +0.46 at exp5). The dose lever is capped by pool
size; this module grows the pool while preserving the exact target
construction and the BabyLM register. Fully-synthetic templates are the
Tier 2 fallback if this doesn't scale far enough.

HOW: matrix object-gap questions in the held lines are decomposed into
slots — (wh, aux, pronoun-subject, tail) — and recombined:

    tail x (aux, pronoun)  within the same AUX FAMILY

Aux families make the recombination grammatical BY CONSTRUCTION, not by
luck: every do-family aux (do/does/did + modals) takes a bare-form VP
tail, every be-family aux takes a V-ing/predicate tail, every have-family
aux takes a participle tail — so tails are interchangeable within a
family, and subject agreement is enforced per aux (does -> he/she/it,
are -> you/we/they, ...). Long-distance variants arise automatically from
tails that contain a bridge-verb clause ("think she saw _").

Safety rails (the user-mandated naturalness gate is in
scripts/augment_syntax.py, which runs dry by default):
- every generated sentence must re-pass syntax.classify_line() as target
- dedup against original held lines AND every BLiMP eval sentence
  (good + bad, all paradigms) after normalization
- per-group sampling cap so one giant (wh, family) group can't dominate
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from . import syntax

# Subject pronouns per aux, grouped by aux family (tail-form compatibility).
# Modals beyond can/could/will/would/should are excluded: grammatical but
# register-alien in child-directed speech (must/shall/may/might).
_ALL_PRON = ("i", "you", "he", "she", "we", "they")
DO_FAMILY: dict[str, tuple[str, ...]] = {
    "do": ("i", "you", "we", "they"),
    "does": ("he", "she"),
    "did": _ALL_PRON,
    "can": _ALL_PRON,
    "could": _ALL_PRON,
    "will": _ALL_PRON,
    "would": _ALL_PRON,
    "should": _ALL_PRON,
}
BE_FAMILY: dict[str, tuple[str, ...]] = {
    "am": ("i",),
    "is": ("he", "she"),
    "are": ("you", "we", "they"),
    "was": ("i", "he", "she"),
    "were": ("you", "we", "they"),
}
HAVE_FAMILY: dict[str, tuple[str, ...]] = {
    "have": ("i", "you", "we", "they"),
    "has": ("he", "she"),
    "had": _ALL_PRON,
}
FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "do": DO_FAMILY, "be": BE_FAMILY, "have": HAVE_FAMILY,
}
_AUX_TO_FAMILY = {aux: fam for fam, d in FAMILIES.items() for aux in d}

# Slot capture, calibrated to the lowercase space-tokenized BabyLM lines:
# (wh)(aux)(pronoun)(tail). Applied to the question SPAN (from the matrix
# match to the next terminator), not the whole line.
_AUX_ANY = "|".join(_AUX_TO_FAMILY)
SLOT_RE = re.compile(
    rf"({syntax._WH})\s+({_AUX_ANY})\s+({'|'.join(_ALL_PRON)})\s+(\w.*)$",
    re.IGNORECASE,
)
_TERMINATOR_RE = re.compile(r"[.?!]")
_TAIL_BAD_RE = re.compile(r"[\"'()\[\]:;,.?!]|--")

# --- naturalness filters (added after the first 200-sentence gate review;
# each targets a defect class observed in that sample) ---

# verbless/fragment tails ("what did they | with a sword ?"): the tail must
# not open with a preposition, determiner, conjunction, or bare pronoun
_TAIL_START_BAN = frozenset(
    "with of to in on for at by from about and but or so if that this these "
    "those the a an i you he she it we they them us her him my your guys".split()
)
# run-on spoken lines ("... | what do you want me to do"): no second wh-word
# anywhere in the tail
_TAIL_WH_RE = re.compile(rf"\b{syntax._WH}\b", re.IGNORECASE)
# corpus typos ("f or you"): no stray single-letter tokens besides a/i
_TAIL_TYPO_RE = re.compile(r"\b[b-hj-z]\b")
# binding: a reflexive in the tail must agree with the swapped-in subject
_REFLEXIVE_OWNER = {
    "myself": "i", "yourself": "you", "yourselves": "you", "himself": "he",
    "herself": "she", "ourselves": "we", "themselves": "they",
}
_REFLEXIVE_RE = re.compile(
    r"\b(myself|yourself|yourselves|himself|herself|ourselves|themselves|itself)\b"
)
# tense anchors: past-anchored tails only pair with past auxes, and
# future-anchored tails never do
_PAST_ADV_RE = re.compile(
    r"\b(?:yesterday|ago|last\s+(?:night|week|month|year|time))\b")
_FUTURE_ADV_RE = re.compile(r"\b(?:tomorrow|next\s+(?:week|month|year|time))\b")
_PAST_AUXES = frozenset({"did", "was", "were", "had"})
# person clash ("what will we tell us ...", "what would i want me to do"):
# a subject pronoun shouldn't co-occur with its own non-reflexive object form
_OBJ_CLASH = {"i": re.compile(r"\bme\b"), "we": re.compile(r"\bus\b")}
# do-family auxes take a BARE verb; corpus typos put finite past forms in
# some tails ("what will he got for me") — ban them as do-family tail openers
_PAST_FORM_STARTS = frozenset(
    "got went came saw ate took made said told thought found gave knew "
    "heard meant felt kept left did was were had done gone seen".split()
)


def _tail_ok(tail: str) -> bool:
    first = tail.split()[0]
    if first in _TAIL_START_BAN:
        return False
    if _TAIL_WH_RE.search(tail):
        return False
    if _TAIL_TYPO_RE.search(tail):
        return False
    return True


def _combo_ok(aux: str, pron: str, tail: str) -> bool:
    m = _REFLEXIVE_RE.search(tail)
    if m and _REFLEXIVE_OWNER.get(m.group(1)) != pron:
        return False
    if _PAST_ADV_RE.search(tail) and aux not in _PAST_AUXES:
        return False
    if _FUTURE_ADV_RE.search(tail) and aux in _PAST_AUXES:
        return False
    clash = _OBJ_CLASH.get(pron)
    if clash and clash.search(tail):
        return False
    if _AUX_TO_FAMILY[aux] == "do" and tail.split()[0] in _PAST_FORM_STARTS:
        return False
    return True


@dataclass
class Frame:
    wh: str
    aux: str
    pron: str
    tail: str
    src: int = -1  # index into the held_lines list this frame came from


@dataclass
class AugmentStats:
    lines_scanned: int = 0
    frames_extracted: int = 0
    candidates_raw: int = 0
    dropped_classify: int = 0
    dropped_dup_original: int = 0
    dropped_dup_blimp: int = 0
    generated: int = 0
    generated_long_distance: int = 0
    groups: dict[str, int] = field(default_factory=dict)


def _normalize(s: str) -> str:
    """Comparison key: lowercase word sequence, punctuation-blind — so
    'What did Karen see?' collides with 'what did karen see ?'."""
    return " ".join(re.findall(r"[a-z']+", s.lower()))


def extract_frames(lines: list[str], max_tail_words: int = 12) -> list[Frame]:
    """Decompose matrix object-gap questions into recombination slots.
    Lines that don't fit the strict slot shape (multiword subjects,
    embedded-only, odd punctuation) are simply skipped — they stay in the
    pool as originals; recombination only needs the clean majority."""
    frames = []
    for src, line in enumerate(lines):
        m = syntax.MATRIX_WH_OBJ_RE.search(line)
        if not m:
            continue
        span_start = m.start()
        # regex anchor may consume the leading terminator/quote — step
        # forward to the wh-word itself
        wh_m = syntax._WH_RE.search(line, span_start)
        if not wh_m:
            continue
        term = _TERMINATOR_RE.search(line, wh_m.start())
        span = line[wh_m.start(): term.start() if term else len(line)].strip()
        sm = SLOT_RE.match(span)
        if not sm:
            continue
        wh, aux, pron, tail = (g.strip().lower() for g in sm.groups())
        if _TAIL_BAD_RE.search(tail):
            continue
        if not 1 <= len(tail.split()) <= max_tail_words:
            continue
        if not _tail_ok(tail):
            continue
        frames.append(Frame(wh, aux, pron, tail, src))
    return frames


# ---------------------------------------------------------------------------
# Embedded wh-complement recombination ("do you know | what this is")
#
# Added after the exp10 probe: matrix-question augmentation fixed the
# memorization collapse but the LD interaction stayed ~0.54, and BLiMP's
# wh_vs_that items are embedded declaratives — the one construction the
# pool barely grows. Recombination here is PREFIX x WH-CLAUSE, grammatical
# by construction: the prefix carries its own subject/tense and the clause
# is finite and self-contained, so no agreement bookkeeping is needed.
# ---------------------------------------------------------------------------

@dataclass
class EmbedFrame:
    prefix: str      # up to and including the embedding verb ("i wonder")
    clause: str      # wh-word onward ("what thomas has")
    term: str        # terminator of the PREFIX's source sentence — the
    src: int = -1    # prefix determines question-hood ("do you know ... ?")


# prefixes may contain these contractions; any other apostrophe is treated
# as quote noise and the frame is skipped
_PREFIX_CONTRACTIONS = frozenset(
    "don't didn't doesn't can't won't wouldn't couldn't shouldn't "
    "i'm you're we're they're she's he's let's that's it's".split()
)
_APOSTROPHE_TOKEN_RE = re.compile(r"\S*'\S*")
# ASCII-lowercase only: kills curly quotes/dashes and subtitle markup that
# the char-class bans can't enumerate
_CLEAN_RE = re.compile(r"^[a-z0-9' -]+$")
_OPENER_WORDS = frozenset(
    "so and but then well now oh hey um uh yeah ok okay just please "
    "actually no yes".split()
)
# verb-presence heuristic for clauses (no POS tagger in this project):
# closed list of auxes + high-frequency verbs, plus -s/-ed/-ing suffixes
_VERBISH = frozenset(
    "is are was were am be been being do does did have has had can could "
    "will would should must may might gonna wanna gotta go goes went gone "
    "say said see saw seen know knew want need mean meant think thought "
    "like love hate tell told get got make made take took put call came "
    "come use hear heard find found buy bought eat ate drink play feel "
    "felt look happen keep kept let give gave run ran".split()
)


def _strip_openers(prefix: str) -> str:
    words = prefix.split()
    while words and (words[0] in _OPENER_WORDS or words[0] == "-"):
        words = words[1:]
    return " ".join(words)


# fragment prefixes ("are to clearly understand", "to know") and glued-on
# clause debris ("matachin and made miguel understand")
_PREFIX_START_BAN = frozenset(
    "is are was were am be been being to as of in that not".split())
_PREFIX_CONJ_RE = re.compile(r"\b(?:and|but|or)\b")
_FILLER_RE = re.compile(r"\b(?:erm|er|um|uh|eh|mm|hm|hmm)\b")
_DASH_TOKEN_RE = re.compile(r"(?:^|\s)-|-(?:\s|$)|\w-(?:\s|$)|(?:^|\s)-\w")
_CLAUSE_END_BAN = frozenset(
    "the a an and but or that this my your his her their".split())


def _prefix_ok(prefix: str, max_words: int = 6) -> bool:
    words = prefix.split()
    if not 1 <= len(words) <= max_words:
        return False
    if not _CLEAN_RE.match(prefix) or "-" in prefix:
        return False
    if words[0] in _PREFIX_START_BAN:
        return False
    if _PREFIX_CONJ_RE.search(prefix) or _FILLER_RE.search(prefix):
        return False
    if _TAIL_WH_RE.search(prefix):
        return False
    for t in _APOSTROPHE_TOKEN_RE.findall(prefix):
        if t not in _PREFIX_CONTRACTIONS:
            return False
    return True


def _clause_ok(clause: str, max_words: int = 14) -> bool:
    words = clause.split()
    if not 3 <= len(words) <= max_words:  # wh + subject + verb minimum
        return False
    if not _CLEAN_RE.match(clause) or "'" in clause:
        return False
    if _TAIL_TYPO_RE.search(clause) or _FILLER_RE.search(clause) \
            or _DASH_TOKEN_RE.search(clause) or "  " in clause:
        return False
    if words[-1] in _CLAUSE_END_BAN:
        return False
    # exactly one wh — the clause's own; a second one is a run-on
    if len(_TAIL_WH_RE.findall(clause)) != 1:
        return False
    # must plausibly contain a verb ("what a lawyer" / "which of the two"
    # are verbless fragments)
    if not any(w in _VERBISH or w.endswith(("ed", "ing"))
               or (len(w) > 3 and w.endswith("s")) for w in words[1:]):
        return False
    return True


def extract_embedded_frames(lines: list[str]) -> list[EmbedFrame]:
    frames = []
    for src, line in enumerate(lines):
        m = syntax.EMBEDDED_WH_RE.search(line)
        if not m:
            continue
        wh_m = syntax._WH_RE.search(line, m.start())
        if not wh_m:
            continue
        # sentence start: just after the previous terminator, if any
        sent_start = 0
        for t in _TERMINATOR_RE.finditer(line, 0, m.start()):
            sent_start = t.end()
        prefix = _strip_openers(line[sent_start: wh_m.start()].strip().lower())
        term_m = _TERMINATOR_RE.search(line, wh_m.start())
        clause = line[wh_m.start(): term_m.start() if term_m
                      else len(line)].strip().lower()
        # the prefix's sentence type decides the terminator ("do you know
        # ... ?" vs "i wonder ... .") — and here prefix and clause share a
        # source sentence, so its terminator belongs to the prefix
        term = term_m.group(0) if term_m and term_m.group(0) in ".?" else "."
        if not _prefix_ok(prefix) or not _clause_ok(clause):
            continue
        frames.append(EmbedFrame(prefix, clause, term, src))
    return frames


def generate_embedded(
    held_lines: list[str],
    blimp_sentences: list[str],
    n_target: int,
    seed: int = 1,
    heldout_frac: float = 0.10,
    n_eval_per_set: int = 1000,
    seen: set[str] | None = None,
) -> tuple[list[str], AugmentStats, dict]:
    """PREFIX x CLAUSE recombination with a CLAUSE-level holdout (the same
    discipline as the matrix holdout, one construction level down — 'no
    leak' at the matrix-frame granularity doesn't guarantee it here).
    Pass `seen` from the matrix pass so the two pools can't overlap."""
    stats = AugmentStats(lines_scanned=len(held_lines))
    frames = extract_embedded_frames(held_lines)
    stats.frames_extracted = len(frames)

    rng = random.Random(seed)
    clauses = sorted({f.clause for f in frames})
    held_clauses = set(rng.sample(clauses,
                                  int(len(clauses) * heldout_frac)))
    trained = [f for f in frames if f.clause not in held_clauses]
    heldout = [f for f in frames if f.clause in held_clauses]
    exclude_line_idxs = sorted({f.src for f in heldout})

    if seen is None:
        seen = {_normalize(l) for l in held_lines}
    blimp = {_normalize(s) for s in blimp_sentences}

    def cross(fs: list[EmbedFrame], limit: int, r: random.Random,
              st: AugmentStats | None) -> list[dict]:
        prefixes = sorted({(f.prefix, f.term) for f in fs})
        cls = sorted({f.clause for f in fs})
        if st is not None:
            st.candidates_raw = len(prefixes) * len(cls)
            st.groups["embedded"] = st.candidates_raw
        out: list[dict] = []
        attempts = 0
        max_attempts = limit * 20
        while len(out) < limit and attempts < max_attempts:
            attempts += 1
            p, term = r.choice(prefixes)
            c = r.choice(cls)
            sent = f"{p} {c} {term}"
            key = _normalize(sent)
            if key in seen or key in blimp:
                continue
            cl = syntax.classify_line(sent)
            if not cl["target"]:
                continue
            seen.add(key)
            out.append({"sentence": sent, "tail": c,
                        "long_distance": cl["long_distance"]})
        return out

    eval_trained = cross(trained, n_eval_per_set, random.Random(seed + 3), None)
    train_recs = cross(trained, n_target, rng, stats)
    eval_heldout = cross(heldout, n_eval_per_set, random.Random(seed + 5), None)

    out = [r["sentence"] for r in train_recs]
    stats.generated = len(out)
    stats.generated_long_distance = sum(r["long_distance"] for r in train_recs)

    holdout_info = {
        "level": "embedded_clause",
        "heldout_frac": heldout_frac,
        "n_clauses_total": len(clauses),
        "n_clauses_heldout": len(held_clauses),
        "heldout_clauses": sorted(held_clauses),
        "exclude_line_idxs": exclude_line_idxs,
        "eval_trained": eval_trained,
        "eval_heldout": eval_heldout,
    }
    return out, stats, holdout_info


def _combos_for(
    frames: list[Frame], n_target: int, per_group_cap: float,
    rng: random.Random, stats: AugmentStats | None = None,
) -> list[tuple[str, str, str, str]]:
    """All valid (wh, aux, pron, tail) recombinations, shuffled, with any
    single (wh, family) group capped to per_group_cap * n_target."""
    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for f in frames:
        fam = _AUX_TO_FAMILY[f.aux]
        # keep the source aux with the tail: a be-family tail is only
        # guaranteed compatible with its own family's aux table
        groups.setdefault((f.wh, fam), []).append((f.aux, f.tail))
    order: list[tuple[str, str, str, str]] = []
    for (wh, fam), auxtails in groups.items():
        tails = sorted({t for _a, t in auxtails})
        combos = [
            (wh, aux, pron, tail)
            for tail in tails
            for aux, prons in FAMILIES[fam].items()
            for pron in prons
            if _combo_ok(aux, pron, tail)
        ]
        rng.shuffle(combos)
        order.extend(combos[: int(n_target * per_group_cap)])
        if stats is not None:
            stats.groups[f"{wh}|{fam}"] = len(combos)
            stats.candidates_raw += len(combos)
    rng.shuffle(order)
    return order


def _realize(
    combos: list[tuple[str, str, str, str]], limit: int,
    seen: set[str], blimp: set[str], stats: AugmentStats | None = None,
) -> list[dict]:
    """Turn combos into validated, deduped sentence records
    ({sentence, tail, long_distance}), mutating `seen`."""
    out: list[dict] = []
    for wh, aux, pron, tail in combos:
        if len(out) >= limit:
            break
        sent = f"{wh} {aux} {pron} {tail} ?"
        key = _normalize(sent)
        if key in seen:
            if stats is not None:
                stats.dropped_dup_original += 1
            continue
        if key in blimp:
            if stats is not None:
                stats.dropped_dup_blimp += 1
            continue
        c = syntax.classify_line(sent)
        if not c["target"]:
            if stats is not None:
                stats.dropped_classify += 1
            continue
        seen.add(key)
        out.append({"sentence": sent, "tail": tail,
                    "long_distance": c["long_distance"]})
    return out


def generate(
    held_lines: list[str],
    blimp_sentences: list[str],
    n_target: int,
    seed: int = 1,
    per_group_cap: float = 0.5,
    heldout_frac: float = 0.10,
    n_eval_per_set: int = 1000,
) -> tuple[list[str], AugmentStats, dict]:
    """Recombine frames into up to n_target new training sentences, holding
    out a frame-level split for the post-training generalization test.

    A random heldout_frac of unique TAILS (frame templates) is excluded
    from recombination, and the source lines those tails came from must be
    excluded from the committed pool (their indices are returned) — so the
    model never sees the held-out frames in any form. Two matched eval
    sets are returned in `holdout`:
      eval_trained: unseen recombinations of TRAINED frames (novel
        sentences, familiar skeletons) — also excluded from the pool
      eval_heldout: recombinations of HELD-OUT frames (novel sentences,
        novel skeletons)
    If the trained-frame set scores much higher per-token log-prob than
    the held-out set, the model memorized skeletons; if the gap is small,
    it generalized the construction.

    Returns (train_sentences, stats, holdout); sentences are lowercase,
    space-tokenized, '?'-terminated — same register as the corpus.
    """
    stats = AugmentStats(lines_scanned=len(held_lines))
    frames = extract_frames(held_lines)
    stats.frames_extracted = len(frames)

    rng = random.Random(seed)
    tails = sorted({f.tail for f in frames})
    held_tails = set(rng.sample(tails, int(len(tails) * heldout_frac)))
    trained_frames = [f for f in frames if f.tail not in held_tails]
    heldout_frames = [f for f in frames if f.tail in held_tails]
    exclude_line_idxs = sorted({f.src for f in heldout_frames})

    seen: set[str] = {_normalize(l) for l in held_lines}
    blimp: set[str] = {_normalize(s) for s in blimp_sentences}

    combos = _combos_for(trained_frames, n_target, per_group_cap, rng, stats)
    # reserve the trained-frame eval set FIRST — realized sentences land in
    # `seen`, so the training pass below can never include them (they must
    # stay unseen by the model for the generalization test to mean anything)
    eval_trained = _realize(combos, n_eval_per_set, seen, blimp)
    train_recs = _realize(combos, n_target, seen, blimp, stats)
    held_combos = _combos_for(heldout_frames, n_eval_per_set, 1.0,
                              random.Random(seed + 7))
    eval_heldout = _realize(held_combos, n_eval_per_set, seen, blimp)

    out = [r["sentence"] for r in train_recs]
    stats.generated = len(out)
    stats.generated_long_distance = sum(r["long_distance"] for r in train_recs)

    holdout = {
        "heldout_frac": heldout_frac,
        "n_tails_total": len(tails),
        "n_tails_heldout": len(held_tails),
        "heldout_tails": sorted(held_tails),
        "exclude_line_idxs": exclude_line_idxs,
        "eval_trained": eval_trained,
        "eval_heldout": eval_heldout,
    }
    return out, stats, holdout

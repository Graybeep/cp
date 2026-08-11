"""Unit tests for the skill modules (plain asserts, no framework).

Run:  python scripts/test_skills.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reopen.config import DATA_DIR, ModelConfig  # noqa: E402
from reopen.model import GPT  # noqa: E402
from reopen.skills import arithmetic as arith  # noqa: E402
from reopen.skills import syntax  # noqa: E402


def test_arith_generator() -> None:
    assert arith.has_carry(15, 16)          # 5+6 carries
    assert arith.has_carry(19, 1)           # 9+1 carries
    assert arith.has_carry(95, 12)          # 9+1 in tens column
    assert not arith.has_carry(23, 45)
    assert not arith.has_carry(1234, 4321)
    assert arith.format_problem(347, 285) == "347+285=632"

    rng = random.Random(0)
    cfg_none = arith.ArithConfig(carrying="none")
    cfg_req = arith.ArithConfig(carrying="required")
    for _ in range(1000):
        a, b = arith.sample_problem(rng, cfg_none)
        assert not arith.has_carry(a, b)
        assert 10 <= a <= 9999 and 10 <= b <= 9999
        a, b = arith.sample_problem(rng, cfg_req)
        assert arith.has_carry(a, b)

    cfg = arith.ArithConfig()
    eval_set = arith.make_eval_set(cfg, 200)
    assert len(set(eval_set)) == 200, "eval problems must be unique"
    carries = sum(arith.has_carry(a, b) for a, b in eval_set)
    assert carries == 100, "mixed eval set must be carry-balanced"

    train = [ln for ln in arith.iter_train_lines(cfg, 2000, set(eval_set))]
    assert len(train) == 2000
    ev = {arith.format_problem(a, b) for a, b in eval_set}
    assert not ev & {ln.strip() for ln in train}, "eval leaked into train"
    for ln in train[:50]:
        lhs, rhs = ln.strip().split("=")
        a, b = lhs.split("+")
        assert int(a) + int(b) == int(rhs)
    print("arith generator OK")


def test_syntax_detector() -> None:
    positives = [
        "What did you see?",                          # matrix object gap
        "Who did Mary say she saw?",                  # long-distance
        "What do you think she saw?",                 # long-distance
        "Which book did John buy yesterday?",
        "I wonder what she bought.",                  # embedded
        "he asked what the boy found there",          # embedded, no caps
        "And then she said: \"What did he bring?\"",  # inside quotes
        "So what did you do?",                        # discourse opener
        "Tell me, what did you see?",                 # comma + wh-pronoun
        "what does she want ?",                       # lowercase line start
        "What was she doing?",              # be-aux + subject + V-ing: kept
        "What did Ben see?",                # -en subject name: kept (do-aux)
    ]
    negatives = [
        "Who saw you?",                    # subject gap (no aux inversion)
        "The cat sat on the mat.",
        "Where did you go?",               # adjunct wh — excluded
        "Why would he do that?",           # adjunct wh — excluded
        "What?",
        "I like what you like",            # 'like' not an embed verb
        # relative clauses (mid-sentence wh):
        "There are things which may be happening now.",
        "My dog, which was very old, slept all day.",
        "The man who did the thing left.",
        "the woman who was holding his hand saw those two trees",
        # subject-gap progressives/passives after be/have:
        "Who was holding the baby?",
        "What was said about me?",
        "Everyone who has seen it agrees.",
    ]
    for s in positives:
        assert syntax.classify_line(s)["target"], f"missed: {s}"
    for s in negatives:
        assert not syntax.classify_line(s)["target"], f"false hit: {s}"

    assert syntax.classify_line("Who did Mary say she saw?")["long_distance"]
    assert syntax.classify_line("What do you think she saw?")["long_distance"]
    assert not syntax.classify_line("What did you see?")["long_distance"]
    print("syntax detector OK")


def test_scoring_shapes() -> None:
    """Random tiny model: arith accuracy ~0, BLiMP accuracy ~0.5."""
    from tokenizers import Tokenizer

    tok_path = DATA_DIR / "tokenized" / "tokenizer.json"
    if not tok_path.exists():
        print("scoring test SKIPPED (tokenizer not built)")
        return
    tok = Tokenizer.from_file(str(tok_path))
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), n_layer=2,
                      n_head=8, n_embd=128, block_size=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(device)

    acfg = arith.ArithConfig()
    probs = arith.make_eval_set(acfg, 64)
    acc = arith.grade_exact_match(model, tok, probs, device=device)
    assert 0.0 <= acc <= 0.05, f"random model arith acc {acc}"

    pairs = [
        ("What did you see?", "What you did see?"),
        ("Who did Mary say she saw?", "Who did Mary say saw she?"),
    ] * 16
    b = syntax.blimp_accuracy(model, tok, pairs, device=device)
    assert 0.0 <= b <= 1.0
    print(f"scoring OK (random-model arith acc {acc:.3f}, blimp {b:.2f})")


def test_vocab_control() -> None:
    from collections import Counter

    from reopen.skills import vocab_frequency as vocab

    fam = vocab.inflection_family("dragon")
    assert {"dragon", "dragons"} <= fam
    assert {"dance", "dances", "danced", "dancing"} \
        <= vocab.inflection_family("dance")
    assert {"stop", "stops", "stopped", "stopping"} \
        <= vocab.inflection_family("stop")
    assert {"carry", "carries", "carried", "carrying"} \
        <= vocab.inflection_family("carry")

    # selection: band respected, inflected-looking forms excluded,
    # deterministic, no family overlap
    from string import ascii_lowercase
    fake_words = [f"w{a}{b}k" for a in ascii_lowercase
                  for b in ascii_lowercase][:300]  # alpha, len 4, safe suffix
    counts = Counter({w: 200 for w in fake_words})
    counts.update({"common": 50_000, "rare": 3,
                   "walked": 200, "running": 200, "quickly": 200,
                   "london": 200})
    cap_counts = Counter({"london": 195})  # proper noun: nearly always caps
    cfg = vocab.VocabControlConfig(n_words=100, min_count=100, max_count=500)
    targets = vocab.select_target_words(counts, cfg, cap_counts)
    assert len(targets) == 100
    assert "common" not in targets and "rare" not in targets
    assert "london" not in targets, "proper nouns must be excluded"
    assert not {"walked", "running", "quickly"} & set(targets)
    assert targets == vocab.select_target_words(counts, cfg, cap_counts), \
        "not deterministic"

    fam_set = vocab.build_family_set(["dragon"])
    assert vocab.line_contains_target("The dragons flew away.", fam_set)
    assert not vocab.line_contains_target("The dog ran away.", fam_set)
    # substring must NOT match ('dragonfly' is a different word)
    assert not vocab.line_contains_target("A dragonfly landed.", fam_set)

    # eval instances: enough left context, and eval lines leave the
    # reinjection stream
    held = [
        "the little green dragon flew over the castle wall\n",
        "dragons\n",  # no left context -> unusable for eval
        "she saw a dragon in the cave yesterday\n",
        "the movie was made by Dragon Pictures in London\n",  # proper noun
    ]
    cfg2 = vocab.VocabControlConfig(n_words=1, min_left_words=3, n_eval=10)
    inst, reinject = vocab.make_eval_instances(held, ["dragon"], cfg2)
    assert all(i["word_base"] == "dragon" for i in inst)
    assert all(i["target"][0].islower() for i in inst), \
        "capitalized (proper-noun) surface forms must not be eval targets"
    assert len(inst) + len(reinject) == len(held)
    for i in inst:
        assert i["context"] + i["target"] not in [r.rstrip("\n") for r in reinject]
    print("vocab control OK")


def test_vocab_cloze_scoring() -> None:
    from tokenizers import Tokenizer

    from reopen.skills import vocab_frequency as vocab

    tok_path = DATA_DIR / "tokenized" / "tokenizer.json"
    if not tok_path.exists():
        print("vocab cloze test SKIPPED (tokenizer not built)")
        return
    tok = Tokenizer.from_file(str(tok_path))
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tok.get_vocab_size(), n_layer=2,
                      n_head=8, n_embd=128, block_size=128)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(cfg).to(device)
    instances = [
        {"context": "the little green ", "target": "dragon",
         "word_base": "dragon"},
        {"context": "she went into the ", "target": "kitchen",
         "word_base": "kitchen"},
    ] * 8
    acc, mean_lp = vocab.cloze_scores(model, tok, instances, device=device)
    assert 0.0 <= acc <= 0.2, f"random model cloze acc {acc}"
    assert mean_lp < 0
    acc2, _lp2, breakdown = vocab.cloze_scores(
        model, tok, instances, device=device, per_word=True)
    assert acc2 == acc
    assert set(breakdown) == {"dragon", "kitchen"}
    assert all(v["n"] == 8 for v in breakdown.values())
    print(f"vocab cloze scoring OK (random-model acc {acc:.3f}, "
          f"mean logprob {mean_lp:.1f})")


if __name__ == "__main__":
    test_arith_generator()
    test_syntax_detector()
    test_vocab_control()
    test_scoring_shapes()
    test_vocab_cloze_scoring()
    print("ALL SKILL TESTS PASSED")

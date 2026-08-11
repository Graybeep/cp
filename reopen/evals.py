"""Eval harness (build priority #4).

Runs at EVERY checkpoint from day one, logging skill accuracy and general
perplexity together — the pairing that distinguishes reopening from
catastrophic forgetting is structural, not bolted on.

What it scores (whatever eval data exists on disk; missing sets -> None):
- arith_accuracy:  teacher-forced exact match on held-out 2-4 digit
                   addition problems (skills/arith/eval.jsonl)
- syntax_accuracy: the wh-LICENSING INTERACTION (long-distance), NOT a raw
                   BLiMP paradigm score. Raw wh_vs_that paradigms are
                   dominated by the unigram frequency prior for "that" —
                   an untrained model scores 0.000 / 1.000 on the two
                   halves purely from word frequency. The interaction
                   (mean of with_gap and no_gap) cancels that main effect,
                   so a frequency-only model lands at exactly chance (0.5)
                   and any excess is real gap-licensing knowledge.
                   syntax_accuracy_local = the same, local distance.
- vocab_control_accuracy / _logprob: cloze recovery of the delayed word
                   band (NEGATIVE CONTROL — expected to show NO critical
                   period; see skills/vocab_frequency.py)

Checkpoint-time evals use a deterministic subset (first n_per_skill
instances) so numbers are comparable across checkpoints and runs; the
post-hoc script (scripts/eval_checkpoints.py) runs full sets.

Contamination note: for skill="none" baselines trained on the full
train.bin, the vocab eval sentences WERE in the training stream, so
vocab numbers are only meaningful for vocab-condition runs (their
streams exclude the eval lines by construction).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import DATA_DIR
from .skills import arithmetic, syntax, vocab_frequency


class EvalHarness:
    def __init__(
        self,
        device: str = "cuda",
        n_per_skill: int | None = 500,
        full_blimp: bool = False,
        data_dir: Path = DATA_DIR,
    ):
        from tokenizers import Tokenizer

        self.device = device
        self.tok = Tokenizer.from_file(
            str(data_dir / "tokenized" / "tokenizer.json")
        )
        n = n_per_skill

        self.arith_problems: list[tuple[int, int]] = []
        arith_path = data_dir / "skills" / "arith" / "eval.jsonl"
        if arith_path.exists():
            for line in arith_path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self.arith_problems.append((d["a"], d["b"]))
            self.arith_problems = self.arith_problems[:n]

        self.blimp: dict[str, list[tuple[str, str]]] = {}
        # the four paradigms forming the two licensing 2x2s
        self.primaries = [name for pair in syntax.LICENSING_PAIRS.values()
                          for name in pair]
        wanted = syntax.BLIMP_PARADIGMS if full_blimp else self.primaries
        for name in wanted:
            p = data_dir / "blimp" / f"{name}.jsonl"
            if p.exists():
                self.blimp[name] = syntax.load_blimp_pairs(p)[:n]

        self.vocab_instances: list[dict] = []
        vocab_path = data_dir / "skills" / "vocab" / "eval.jsonl"
        if vocab_path.exists():
            for line in vocab_path.read_text(encoding="utf-8").splitlines():
                self.vocab_instances.append(json.loads(line))
            self.vocab_instances = self.vocab_instances[:n]

    @torch.no_grad()
    def evaluate(self, model, per_word: bool = False) -> dict:
        """Returns a flat dict for metrics.csv plus nested detail
        (per-paradigm BLiMP accs; per-word vocab breakdown on request).
        Missing eval sets yield None. Leaves the model in eval mode —
        callers that keep training must restore train() themselves."""
        out: dict = {
            "arith_accuracy": None,
            "syntax_accuracy": None,        # licensing interaction, long-distance
            "syntax_accuracy_local": None,  # licensing interaction, local
            "vocab_control_accuracy": None,
            "vocab_control_logprob": None,
            "blimp": {},
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.arith_problems:
                out["arith_accuracy"] = arithmetic.grade_exact_match(
                    model, self.tok, self.arith_problems, device=self.device
                )
            for name, pairs in self.blimp.items():
                out["blimp"][name] = syntax.blimp_accuracy(
                    model, self.tok, pairs, device=self.device
                )
            # headline metrics = bias-corrected licensing interactions
            for dist, (wg, ng) in syntax.LICENSING_PAIRS.items():
                if wg in out["blimp"] and ng in out["blimp"]:
                    key = ("syntax_accuracy" if dist == "long_distance"
                           else "syntax_accuracy_local")
                    out[key] = syntax.wh_licensing_interaction(
                        out["blimp"][wg], out["blimp"][ng]
                    )
            if self.vocab_instances:
                res = vocab_frequency.cloze_scores(
                    model, self.tok, self.vocab_instances,
                    device=self.device, per_word=per_word,
                )
                out["vocab_control_accuracy"] = res[0]
                out["vocab_control_logprob"] = res[1]
                if per_word:
                    out["vocab_per_word"] = res[2]
        return out

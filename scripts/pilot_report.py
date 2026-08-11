"""Week 3 Detect pilot report — applies the PRE-REGISTERED read mechanically.

Pre-registration (fixed 2026-07-16, before the pilot batch finished; see
memory + CLAUDE.md):

SYNTAX (continuation interaction, per-item wins saved by
gap_region_interaction.py --save-items):
- real delay gap: delay0 - delayX >= 0.032 (~2 SE) at delay-75 minimum,
  monotone non-increasing trend (one < 1 SE inversion tolerated), AND
  McNemar p < 0.05 on the paired per-item outcomes (decision criterion;
  the 0.032 bar is the eyeball guard).
- thin-ceiling trigger (build pool v3): all conditions in 0.55-0.59,
  every pairwise delta < 0.032, no monotone trend.
- anything else: ambiguous -> add seeds 2-3 before calling.

ARITH (exact-match, delay-0 ceiling ~0.798, SE ~ 0.013 -> bar 0.036):
- aggregate gap at 2-SE discipline, plus MANDATORY per-difficulty read
  from each run's logs/<run_id>/arith_breakdown.csv.
- Cell dichotomy is SE-AWARE (the easy band, 2-digit-no-carry, is near
  ceiling at delay-0: 0.968 on n=63, SE ~ 0.022 — a relative-drop ratio
  on that denominator can look decisive while resting on noise):
  * easy-band drop < 2 SE_easy AND hard-band drop >= max(2 SE_hard, 0.10
    relative) -> HARDEST-BAND CAP (clean: easy statistically unchanged)
  * both drops >= their 2 SE -> apply the x2 relative-drop ratio test
  * hard-band drop < 2 SE_hard -> no detectable per-cell cap
  * anything else -> INDETERMINATE by exact-match; read the graded
    per-token per-cell signal instead of forcing a label.

Run AFTER: (a) the pilot batch, (b) gap_region_interaction.py --save-items
on the four syntax runs, (c) arith_breakdown.py on the four arith runs.

  python scripts/pilot_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import LOG_DIR  # noqa: E402

ITEMS_DIR = LOG_DIR / "pilot_items"
SYNTAX_RUNS = {d: f"syntax_delay{d}_exp10_size23M_seed1" for d in (0, 25, 50, 75)}
# arith delay-75 is INFEASIBLE under matched exposure at exp20: density
# would be 0.20/0.25 = 0.80 and schedule.py's dominance guard requires
# < 0.5 (max matched delay at exp20 is just under 0.6). delay-55
# (density 0.44) substitutes as the deepest feasible arithmetic delay.
ARITH_RUNS = {d: f"arith_delay{d}_exp20_size23M_seed1" for d in (0, 25, 50, 55)}
SE_BAR_SYNTAX = 0.032
SE_BAR_ARITH = 0.036
LICENSING = {
    "local": ("wh_vs_that_with_gap", "wh_vs_that_no_gap"),
    "long_distance": ("wh_vs_that_with_gap_long_distance",
                      "wh_vs_that_no_gap_long_distance"),
}


def load_items(run_id: str) -> dict | None:
    hits = sorted(ITEMS_DIR.glob(f"{run_id}_step*_items.json"))
    return json.loads(hits[-1].read_text()) if hits else None


def mcnemar_p(wins_a: list[int], wins_b: list[int]) -> float:
    b = sum(1 for x, y in zip(wins_a, wins_b) if x and not y)
    c = sum(1 for x, y in zip(wins_a, wins_b) if y and not x)
    if b + c == 0:
        return 1.0
    return sps.binomtest(b, b + c, 0.5).pvalue


def syntax_section() -> None:
    print("=" * 72)
    print("SYNTAX — continuation interaction (pre-registered read)")
    data = {d: load_items(r) for d, r in SYNTAX_RUNS.items()}
    missing = [SYNTAX_RUNS[d] for d, v in data.items() if v is None]
    if missing:
        print(f"  MISSING per-item files for: {missing}")
        print("  run: python scripts/gap_region_interaction.py "
              f"<run_ids> --save-items {ITEMS_DIR}")
        return
    for dist, (pg, png) in LICENSING.items():
        print(f"\n-- {dist}")
        inter = {}
        for d, v in data.items():
            inter[d] = (sum(v["items"][pg]) / len(v["items"][pg])
                        + sum(v["items"][png]) / len(v["items"][png])) / 2
            print(f"  delay{d:>2}: interaction={inter[d]:.3f}")
        base = inter[0]
        print("  vs delay-0: "
              + "  ".join(f"d{d}:{base - inter[d]:+.3f}" for d in (25, 50, 75)))
        for d in (25, 50, 75):
            ps = [mcnemar_p(data[0]["items"][p], data[d]["items"][p])
                  for p in (pg, png)]
            print(f"  McNemar delay0 vs delay{d}: "
                  f"with_gap p={ps[0]:.4f}  no_gap p={ps[1]:.4f}")
        # mechanical pre-registered calls
        drops = [base - inter[d] for d in (25, 50, 75)]
        vals = [inter[0], inter[25], inter[50], inter[75]]
        monotone = all(vals[i] >= vals[i + 1] - 0.016 for i in range(3))
        gap75 = base - inter[75] >= SE_BAR_SYNTAX
        clustered = (all(0.55 <= v <= 0.59 for v in vals)
                     and max(vals) - min(vals) < SE_BAR_SYNTAX)
        if gap75 and monotone:
            call = "REAL DELAY GAP (confirm with McNemar p-values above)"
        elif clustered:
            call = "THIN-CEILING TRIGGER -> build pool v3"
        else:
            call = "AMBIGUOUS -> add seeds 2-3 before calling"
        print(f"  pre-registered call [{dist}]: {call}")


def arith_section() -> None:
    print("\n" + "=" * 72)
    print("ARITH — exact-match + per-difficulty (pre-registered read)")
    finals = {}
    for d, r in ARITH_RUNS.items():
        m = LOG_DIR / r / "metrics.csv"
        if not m.exists():
            print(f"  MISSING metrics for {r}")
            return
        finals[d] = pd.read_csv(m).iloc[-1]
    for d, row in finals.items():
        print(f"  delay{d:>2}: exact={row.arith_accuracy:.3f}  "
              f"val_ppl={row.val_ppl:.2f}  "
              f"skill_tokens={int(row.skill_tokens_seen):,}")
    base = finals[0].arith_accuracy
    delays = [d for d in finals if d != 0]
    for d in delays:
        gap = base - finals[d].arith_accuracy
        sig = "REAL (>=2 SE)" if gap >= SE_BAR_ARITH else "within noise"
        print(f"  delay0 - delay{d}: {gap:+.3f}  [{sig}]")

    print("\n-- per-difficulty (SE-aware read vs delay-0, per cell)")
    cells = {}
    for d, r in ARITH_RUNS.items():
        p = LOG_DIR / r / "arith_breakdown.csv"
        if not p.exists():
            print(f"  MISSING breakdown for {r} — run arith_breakdown.py {r}")
            return
        df = pd.read_csv(p)
        df = df[df.step == df.step.max()]
        cells[d] = df.groupby(["digits", "carry"]).agg(
            exact=("exact", "mean"),
            n=("exact", "size"),
            tok_correct=("tok_correct", "sum"),
            n_tok=("n_tok", "sum")).assign(
                tok_acc=lambda x: x.tok_correct / x.n_tok)
    base_c = cells[0]

    def cell_stats(d: int, key: tuple) -> tuple[float, float, float]:
        """(absolute drop, its ~SE, relative drop) for one cell."""
        p0, n0 = base_c.exact[key], base_c.n[key]
        p1, n1 = cells[d].exact[key], cells[d].n[key]
        drop = p0 - p1
        se = (p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1) ** 0.5
        rel = drop / p0 if p0 > 0 else float("nan")
        return drop, se, rel

    EASY, HARD = (2, False), (4, True)
    for d in [x for x in cells if x != 0]:
        rel_all = (cells[d].exact / base_c.exact.clip(lower=1e-9)).round(2)
        print(f"  delay{d} exact-match retention by (digits,carry): "
              + "  ".join(f"{ix}={v:.2f}" for ix, v in rel_all.items()))
        tok_rel = (cells[d].tok_acc / base_c.tok_acc.clip(lower=1e-9)).round(2)
        print(f"    per-token retention (graded):        "
              + "  ".join(f"{ix}={v:.2f}" for ix, v in tok_rel.items()))
        e_drop, e_se, e_rel = cell_stats(d, EASY)
        h_drop, h_se, h_rel = cell_stats(d, HARD)
        print(f"    easy(2d,nc): drop={e_drop:+.3f} (SE {e_se:.3f})   "
              f"hard(4d,c): drop={h_drop:+.3f} (SE {h_se:.3f})")
        # gradient signature across ALL 6 cells (supplementary): does the
        # relative drop track baseline difficulty (smooth degradation), or
        # is it a hard-cell outlier (sharp collapse)? n=6 means one noisy
        # cell (esp. the n=63 easy cell) can swing rho — so: EXACT
        # permutation p (all 720 orderings, no asymptotics) + leave-one-out
        # range to expose single-cell influence. Never the mechanical call;
        # a null here = "can't distinguish the two stories at this
        # resolution," NOT "no gradient exists."
        base_err = (1 - base_c.exact).values
        rel_drop = (1 - cells[d].exact / base_c.exact.clip(lower=1e-9)).values
        rho = sps.spearmanr(base_err, rel_drop).statistic
        from itertools import permutations
        perm_rhos = [sps.spearmanr(base_err, [rel_drop[j] for j in pi]).statistic
                     for pi in permutations(range(len(rel_drop)))]
        p_perm = sum(abs(r) >= abs(rho) - 1e-12 for r in perm_rhos) / len(perm_rhos)
        loo = [sps.spearmanr([b for k, b in enumerate(base_err) if k != i],
                             [r for k, r in enumerate(rel_drop) if k != i]
                             ).statistic for i in range(len(rel_drop))]
        verdict = ("gradient-consistent" if rho > 0 and p_perm < 0.05
                   and min(loo) > 0 else
                   "indistinguishable at this resolution (n=6 cells)")
        print(f"    gradient check (all 6 cells, supplementary): "
              f"rho={rho:+.2f}  exact-perm p={p_perm:.3f}  "
              f"leave-one-out range [{min(loo):+.2f}, {max(loo):+.2f}] "
              f"-> {verdict}")
        easy_moved = e_drop >= 2 * e_se
        hard_moved = h_drop >= 2 * h_se
        if not hard_moved:
            call = "no detectable per-cell cap (hard band within noise)"
        elif not easy_moved and h_rel >= 0.10:
            # the only pattern that earns the strong label: easy band
            # statistically UNCHANGED while hard band collapses
            call = ("HARDEST-BAND CAP / clean dissociation (easy "
                    f"statistically unchanged, hard rel drop {h_rel:.0%})")
        elif easy_moved:
            # both bands moved: this is graded impairment, NOT a
            # dissociation — "caps only the hardest cases" would oversell
            call = (f"GRADED IMPAIRMENT, steeper at hard end (easy "
                    f"{e_rel:.0%} and hard {h_rel:.0%} both >=2 SE)"
                    if h_rel >= 2 * e_rel else
                    f"UNIFORM IMPAIRMENT (hard {h_rel:.0%} vs easy "
                    f"{e_rel:.0%})")
        else:
            call = ("INDETERMINATE by exact-match — read the per-token "
                    "line above")
        print(f"    -> {call}")


def main() -> None:
    syntax_section()
    arith_section()


if __name__ == "__main__":
    main()

"""Statistical validation of the resume path (supervisor infrastructure).

WHY: the Weeks 5-7 grid depends on resume being unbiased, so the check
must be a measurement, not an eyeball. Design history (kept auditable):
- v1 gated on max(post) <= max(noise): a single-order-statistic comparison
  across unequal sample sizes — failed spuriously 2026-07-17 (MW p=0.194,
  max 0.00367 vs 0.00265) and was replaced, documented, BEFORE rerunning.
- v2 (MW + KS on pooled diffs) had a design flaw caught before its verdict
  was read: ONE resumed run diffed against all references makes every
  test-side point share a single resume event — non-iid on the tested
  side, the same flaw fixed on the noise side by adding replicas.

v3 (pre-registered 2026-07-17, before any v3 verdict was seen):
1. --replicas straight runs (identical config+seed; every difference is
   GPU nondeterminism) and --resume-trials independent killed+resumed
   runs of the same config (each killed by a short launch timeout).
2. POST = checkpoint steps strictly after the LATEST resume step among
   trials (uniform step set for both sides).
3. PRIMARY (iid at the run level): summary S(x) = mean |d(val_loss)| to
   the references over POST steps (references: to the OTHER references).
   Mann-Whitney {S(trial)} vs {S(reference)}. PASS-A: p > 0.05.
4. SECONDARY (pooled, non-iid acknowledged): KS on pooled per-checkpoint
   diffs, trials-vs-refs against refs-vs-refs. PASS-B: p > 0.05.
5. RANK RULE (pre-registered v3.1, 2026-07-17 — added AFTER the v3 run
   launched but BEFORE any statistics or ranks were visible (the run had
   produced only training-progress lines at commit time, and it crashed
   pre-analysis; the analysis ran with this rule already locked). The
   change is strictly tightening: it can demote a PASS to INCONCLUSIVE,
   never rescue a FAIL. MW at
   n=4-vs-5 bottoms out at p~0.016, so absence of significance alone is
   weak — ranks supplement it): under the null each trial exceeds all
   refs with p=1/6 (expected count ~0.67 of 4), so 1-2 top-ranked trials
   = no signal; >=3 of 4 trials each exceeding ALL references (null prob
   ~0.048) = INCONCLUSIVE -> rerun with more trials, never PASS; all 4 =
   the MW-minimum configuration, caught by PASS-A as FAIL.
   VERDICT: PASS = PASS-A AND PASS-B AND rank rule not tripped.
   Pooled MW + max reported, non-gating.
6. DRIFT DIAGNOSTIC — NEVER GATES. Pre-registered investigation trigger
   (not an auto-FAIL): Spearman(post-step index, pooled mean |d|) with
   exact permutation p < 0.05 and rho > 0 = "drift flag" -> investigate
   before the grid. Anything else = no flag. This diagnostic may not be
   promoted to a gate without a new pre-registration.

Reference runs must be UNINTERRUPTED: partially-trained leftovers are
deleted and retrained; completed ones are reused (idempotent).

Run:  python scripts/validate_resume.py [--replicas 5] [--resume-trials 4]
"""

from __future__ import annotations

import argparse
import re
import shutil
import string
import subprocess
import sys
from itertools import combinations, permutations
from pathlib import Path

import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reopen.config import CHECKPOINT_DIR, LOG_DIR  # noqa: E402

SCRIPTS = Path(__file__).parent
TOTAL_STEPS = 240
CKPT_INTERVAL = 20
BASE_ARGS = ["--total-steps", str(TOTAL_STEPS),
             "--checkpoint-interval", str(CKPT_INTERVAL),
             "--warmup-steps", "20", "--num-workers", "2"]


def run_id_of(name: str) -> str:
    return f"{name}_steps{TOTAL_STEPS}_size23M_seed1"


def is_complete(name: str) -> bool:
    rid = run_id_of(name)
    if not (CHECKPOINT_DIR / rid / f"step_{TOTAL_STEPS:07d}.pt").exists():
        return False
    m = LOG_DIR / rid / "metrics.csv"
    return m.exists() and pd.read_csv(m).step.max() == TOTAL_STEPS


def wipe(name: str) -> None:
    shutil.rmtree(CHECKPOINT_DIR / run_id_of(name), ignore_errors=True)
    shutil.rmtree(LOG_DIR / run_id_of(name), ignore_errors=True)


def val_series(name: str) -> pd.Series:
    df = pd.read_csv(LOG_DIR / run_id_of(name) / "metrics.csv")
    return df.set_index("step").val_loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--resume-trials", type=int, default=4)
    args = ap.parse_args()
    refs = [f"val_noise_{c}" for c in string.ascii_lowercase[:args.replicas]]
    trials = [f"val_resume_{c}"
              for c in string.ascii_lowercase[:args.resume_trials]]

    for name in refs:
        if is_complete(name):
            print(f"== reference {name}: complete, reusing", flush=True)
            continue
        wipe(name)  # a partial reference was interrupted -> not usable
        print(f"== reference {name}: training", flush=True)
        subprocess.run([sys.executable, str(SCRIPTS / "train_baseline.py"),
                        "--run-name", name, *BASE_ARGS],
                       check=True, capture_output=True)

    resume_steps: dict[str, int] = {}
    for name in trials:
        wipe(name)  # trials must each be a FRESH kill+resume event
        print(f"== resume trial {name} (kill + resume)", flush=True)
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "run_supervised.py"),
             "--run-name", name, *BASE_ARGS, "--launch-timeout", "55"],
            check=True, capture_output=True, text=True)
        m = re.search(r"resumed \S+ from step (\d+)", proc.stdout)
        assert m, f"{name} was never killed+resumed — lower --launch-timeout"
        resume_steps[name] = int(m.group(1))
        # persist immediately: if anything crashes after training, the
        # analysis can be redone from artifacts without rerunning trials
        (LOG_DIR / run_id_of(name) / "resume_step.txt").write_text(
            str(resume_steps[name]))
        print(f"   resumed at step {resume_steps[name]}")

    series = {n: val_series(n) for n in refs + trials}
    post_from = max(resume_steps.values())
    post_steps = [s for s in series[refs[0]].index if s > post_from]
    assert len(post_steps) >= 3, (
        f"only {len(post_steps)} post-resume checkpoints after step "
        f"{post_from} — kills landed too late for a meaningful test")
    print(f"\nPOST window: steps > {post_from} "
          f"({len(post_steps)} checkpoints)")

    def mean_diff(x: str, others: list[str], steps) -> float:
        return float(pd.concat(
            [(series[x][steps] - series[o][steps]).abs() for o in others]
        ).mean())

    s_ref = {r: mean_diff(r, [o for o in refs if o != r], post_steps)
             for r in refs}
    s_tri = {t: mean_diff(t, refs, post_steps) for t in trials}
    print("run-level summaries S(x) = mean |d(val_loss)| over POST window:")
    for r, v in s_ref.items():
        print(f"  ref   {r}: {v:.5f}")
    for t, v in s_tri.items():
        rank = sum(v > x for x in s_ref.values())
        print(f"  trial {t}: {v:.5f}  (exceeds {rank}/{len(s_ref)} refs)")
    mw_run = sps.mannwhitneyu(list(s_tri.values()), list(s_ref.values()),
                              alternative="two-sided")
    print(f"PRIMARY  run-level Mann-Whitney: p={mw_run.pvalue:.3f}")

    noise = pd.concat([(series[a][post_steps] - series[b][post_steps]).abs()
                       for a, b in combinations(refs, 2)])
    tested = pd.concat([(series[t][post_steps] - series[r][post_steps]).abs()
                        for t in trials for r in refs])
    ks = sps.ks_2samp(tested, noise)
    mw_pool = sps.mannwhitneyu(tested, noise, alternative="two-sided")
    print(f"SECONDARY pooled KS: p={ks.pvalue:.3f}   "
          f"(pooled MW p={mw_pool.pvalue:.3f}, "
          f"max tested={tested.max():.5f} vs noise={noise.max():.5f} — "
          "reported, non-gating)")

    # drift diagnostic (NEVER gates; pre-registered investigation trigger)
    per_step = [float(pd.Series(
        [abs(series[t][s] - series[r][s]) for t in trials for r in refs]
    ).mean()) for s in post_steps]
    rho = sps.spearmanr(range(len(post_steps)), per_step).statistic
    perms = [sps.spearmanr(range(len(post_steps)),
                           [per_step[j] for j in pi]).statistic
             for pi in permutations(range(len(post_steps)))] \
        if len(post_steps) <= 7 else None
    if perms is not None:
        p_drift = sum(abs(x) >= abs(rho) - 1e-12 for x in perms) / len(perms)
    else:
        p_drift = sps.spearmanr(range(len(post_steps)), per_step).pvalue
    flag = rho > 0 and p_drift < 0.05
    print(f"drift diagnostic (non-gating): rho={rho:+.2f} p={p_drift:.3f}"
          f" -> {'DRIFT FLAG: investigate before grid' if flag else 'no flag'}")
    print("per-step pooled mean |d|: "
          + "  ".join(f"{s}:{v:.5f}" for s, v in zip(post_steps, per_step)))

    n_top = sum(1 for v in s_tri.values()
                if all(v > x for x in s_ref.values()))
    inconclusive = n_top >= 3
    ok = mw_run.pvalue > 0.05 and ks.pvalue > 0.05 and not inconclusive
    if inconclusive and mw_run.pvalue > 0.05 and ks.pvalue > 0.05:
        print(f"\nVERDICT: INCONCLUSIVE — {n_top}/4 trials outrank every "
              "reference (null prob ~0.048); rerun with more trials "
              "before trusting resume")
    else:
        print("\nVERDICT:", "PASS (pre-registered v3.1: run-level MW + "
              "pooled KS + rank rule) — post-resume differences are "
              "statistically indistinguishable from the measured "
              "GPU-nondeterminism floor"
              if ok else
              "FAIL — resume differs from the noise floor; do NOT trust "
              "resume until diagnosed")

    if not ok:
        args.keep = True
        print("(FAIL: keeping artifacts for diagnosis)")
    if not args.keep:
        for name in refs + trials:
            wipe(name)
        print("(validation artifacts cleaned; --keep to retain)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

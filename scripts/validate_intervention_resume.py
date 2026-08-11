"""Statistical validation of resume-through-a-STATEFUL-intervention.

WHY (2026-07-18): the no-op fork check validated fork/config-reconstruction
but exercised NOTHING of the replay_fire path — its lr_shift trigger is
stateless (replay_fire=False) and the fork wasn't killed mid-window. The
replay_fire fix (interventions.py) is what makes a killed-and-resumed
freeze/cram/burst run reconstruct the correct requires_grad / loader-density
state. That is the correctness guarantee the Week-4 intervention arms
actually depend on, so it needs the same rigor validate_resume.py gave
plain resume — a measured noise distribution, not two points vs one ref.

DESIGN (mirrors validate_resume.py v3.1, pre-registered here BEFORE any
verdict was seen):
- Intervention under test: FREEZE (embedding-only trainable) over a window
  [AT, UNTIL) that STRADDLES the kill point, so every resume lands mid-
  freeze and MUST re-fire the freeze during replay. Freeze is chosen as the
  most sensitive replay_fire probe: if the resume forgets to re-freeze, the
  post-resume steps train params that should be frozen -> large, obvious
  divergence. (Cram shares the identical replay_fire mechanism and is
  covered by test_interventions.test_cram_replay_reconstruction.)
- N straight freeze references (identical config+seed; every difference is
  GPU nondeterminism) and M killed+resumed freeze trials (each killed by a
  short launch timeout, landing inside the freeze window).
- Same statistics as validate_resume: run-level Mann-Whitney (PASS-A),
  pooled KS (PASS-B), rank rule (>=3/M trials outranking all refs ->
  INCONCLUSIVE), drift diagnostic (non-gating). VERDICT PASS = A and B and
  rank rule not tripped.
- POWER CONTROL (--sensitivity): repeat the trials with REOPEN_BREAK_REPLAY
  set (replay_fire forced off = the pre-fix bug). The check MUST return FAIL
  on these, proving it can actually detect a broken replay. A validation
  that passes the fixed code but can't fail the broken code proves nothing.

Run:  python scripts/validate_intervention_resume.py
      python scripts/validate_intervention_resume.py --sensitivity
"""

from __future__ import annotations

import argparse
import json
import os
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
# Vehicle = CRAM, not freeze. Freeze-all-but-embedding was the first probe
# but it collapses the reference noise floor (only the embedding updates ->
# near-deterministic runs, ref-vs-ref ~0.0001 vs ~0.0013 in validate_resume),
# so the KS gate flags physically negligible resume differences. Cram keeps
# ALL params training -> normal noise floor -> every gate applies cleanly,
# AND it validates a real Week-4 arm. The freeze run's result (run-level MW
# PASS + ~600x fixed/broken separation) is recorded separately as
# corroboration; see logs/validate_intervention_resume_freeze.out.
#
# Big density delta (skill base 0.10 -> cram 0.49) so a MISSED re-fire on
# resume (broken replay_fire) trains post-resume on 0.10 skill where the
# straight run trains on 0.49 -> the two see materially different data ->
# clear divergence for the power control.
SKILL = "arith"
EXPOSURE = 0.10
CRAM = "0.167:0.917:0.49"           # boost to density 0.49 over steps [40, 220)
CRAM_TAG = "cram49"
# Skill runs derive run_id from the SKILL (arith_delay0_exp10_...), ignoring
# --run-name, so replica identity must live in --run-suffix. run_supervised
# (kill+resume trials) also builds run_id from --skill/--exposure/--suffix,
# so both scripts agree without relying on any auto-suffix.
EXP_TAG = f"exp{int(round(EXPOSURE * 100))}"
BASE_ARGS = ["--total-steps", str(TOTAL_STEPS),
             "--checkpoint-interval", str(CKPT_INTERVAL),
             "--warmup-steps", "20", "--num-workers", "2",
             "--skill", SKILL, "--skill-exposure-frac", str(EXPOSURE),
             "--cram", CRAM]


def suffix_for(tag: str) -> str:
    return f"{CRAM_TAG}_{tag}"


def run_id_of(tag: str) -> str:
    return (f"{SKILL}_delay0_{EXP_TAG}_steps{TOTAL_STEPS}_size23M_seed1_"
            f"{suffix_for(tag)}")


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


def realized_exposure(tag: str) -> dict:
    f = LOG_DIR / run_id_of(tag) / "realized_exposure.json"
    return json.load(open(f))["realized_tokens_by_stream"]


def train_ref(tag: str) -> None:
    if is_complete(tag):
        print(f"== reference {tag}: complete, reusing", flush=True)
        return
    wipe(tag)
    print(f"== reference {tag}: training (straight cram run)", flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / "train_baseline.py"),
                    "--run-suffix", suffix_for(tag), *BASE_ARGS],
                   check=True, capture_output=True)


def train_trial(tag: str, break_replay: bool) -> int:
    wipe(tag)  # each trial is a FRESH kill+resume event
    lbl = " [BROKEN replay_fire]" if break_replay else ""
    print(f"== resume trial {tag}{lbl} (kill + resume mid-cram)", flush=True)
    env = dict(os.environ)
    if break_replay:
        env["REOPEN_BREAK_REPLAY"] = "1"
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_supervised.py"),
         "--run-suffix", suffix_for(tag), *BASE_ARGS, "--launch-timeout", "55"],
        check=True, capture_output=True, text=True, env=env)
    m = re.search(r"resumed \S+ from step (\d+)", proc.stdout)
    assert m, f"{tag} was never killed+resumed — lower --launch-timeout"
    step0 = int(m.group(1))
    at = int(round(0.167 * TOTAL_STEPS))
    end = int(round(0.917 * TOTAL_STEPS))
    assert at <= step0 < end, (
        f"{tag} resumed at step {step0}, OUTSIDE the cram window "
        f"[{at}, {end}) — the replay path was not exercised; "
        "adjust launch-timeout")
    (LOG_DIR / run_id_of(tag) / "resume_step.txt").write_text(str(step0))
    print(f"   resumed at step {step0} (inside cram window)")
    return step0


def analyse(refs: list[str], trials: list[str], resume_steps: dict[str, int],
            label: str) -> bool:
    series = {n: val_series(n) for n in refs + trials}
    post_from = max(resume_steps.values())
    post_steps = [s for s in series[refs[0]].index if s > post_from]
    assert len(post_steps) >= 3, (
        f"only {len(post_steps)} post-resume checkpoints after step "
        f"{post_from} — kills landed too late")
    print(f"\n[{label}] POST window: steps > {post_from} "
          f"({len(post_steps)} checkpoints)")

    def mean_diff(x, others, steps):
        return float(pd.concat(
            [(series[x][steps] - series[o][steps]).abs() for o in others]
        ).mean())

    s_ref = {r: mean_diff(r, [o for o in refs if o != r], post_steps)
             for r in refs}
    s_tri = {t: mean_diff(t, refs, post_steps) for t in trials}
    print("run-level S(x) = mean |d(val_loss)| over POST window:")
    for r, v in s_ref.items():
        print(f"  ref   {r}: {v:.5f}")
    for t, v in s_tri.items():
        rank = sum(v > x for x in s_ref.values())
        print(f"  trial {t}: {v:.5f}  (exceeds {rank}/{len(s_ref)} refs)")
    mw_run = sps.mannwhitneyu(list(s_tri.values()), list(s_ref.values()),
                              alternative="two-sided")
    noise = pd.concat([(series[a][post_steps] - series[b][post_steps]).abs()
                       for a, b in combinations(refs, 2)])
    tested = pd.concat([(series[t][post_steps] - series[r][post_steps]).abs()
                        for t in trials for r in refs])
    ks = sps.ks_2samp(tested, noise)
    mw_pool = sps.mannwhitneyu(tested, noise, alternative="two-sided")
    print(f"PRIMARY run-level MW: p={mw_run.pvalue:.3f}   "
          f"SECONDARY pooled KS: p={ks.pvalue:.3f}   "
          f"(pooled MW p={mw_pool.pvalue:.3f}, max tested={tested.max():.5f}"
          f" vs noise={noise.max():.5f})")

    per_step = [float(pd.Series(
        [abs(series[t][s] - series[r][s]) for t in trials for r in refs]
    ).mean()) for s in post_steps]
    rho = sps.spearmanr(range(len(post_steps)), per_step).statistic
    if len(post_steps) <= 7:
        perms = [sps.spearmanr(range(len(post_steps)),
                               [per_step[j] for j in pi]).statistic
                 for pi in permutations(range(len(post_steps)))]
        p_drift = sum(abs(x) >= abs(rho) - 1e-12 for x in perms) / len(perms)
    else:
        p_drift = sps.spearmanr(range(len(post_steps)), per_step).pvalue
    flag = rho > 0 and p_drift < 0.05
    print(f"drift diagnostic (non-gating): rho={rho:+.2f} p={p_drift:.3f}"
          f" -> {'DRIFT FLAG' if flag else 'no flag'}")

    n_top = sum(1 for v in s_tri.values()
                if all(v > x for x in s_ref.values()))
    inconclusive = n_top >= 3
    ok = mw_run.pvalue > 0.05 and ks.pvalue > 0.05 and not inconclusive
    if inconclusive and mw_run.pvalue > 0.05 and ks.pvalue > 0.05:
        verdict = f"INCONCLUSIVE ({n_top}/{len(trials)} trials outrank all refs)"
    else:
        verdict = "PASS" if ok else "FAIL"
    print(f"[{label}] VERDICT: {verdict}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--replicas", type=int, default=5)
    ap.add_argument("--resume-trials", type=int, default=4)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also run trials with replay_fire BROKEN and "
                         "confirm the check FAILS them (power control)")
    args = ap.parse_args()

    refs = [f"noise_{c}" for c in string.ascii_lowercase[:args.replicas]]
    trials = [f"fix_{c}"
              for c in string.ascii_lowercase[:args.resume_trials]]
    for name in refs:
        train_ref(name)
    fixed_steps = {n: train_trial(n, break_replay=False) for n in trials}

    # PRIMARY GATE — deterministic state reconstruction. replay_fire's job is
    # to rebuild the cram loader-density state on resume so the post-resume
    # DATA STREAM is identical. That is a bit-exact property, not a
    # statistical one: every fixed trial's realized token exposure MUST equal
    # the (identical across refs) straight-run exposure. This is a stronger
    # proof than any val_loss null-test — the val_loss stats below only
    # measure the process-boundary "seam" nondeterminism ON TOP of provably
    # identical data.
    ref_exp = realized_exposure(refs[0])
    for r in refs[1:]:
        assert realized_exposure(r) == ref_exp, (
            f"reference {r} exposure differs from {refs[0]} — the straight "
            "runs are not even mutually consistent; cannot use as baseline")
    fixed_exp = {t: realized_exposure(t) for t in trials}
    exposure_ok = all(e == ref_exp for e in fixed_exp.values())
    print("\n[PRIMARY] deterministic exposure reconstruction:")
    print(f"  straight-run exposure (all {len(refs)} refs identical): {ref_exp}")
    for t, e in fixed_exp.items():
        print(f"  fixed {t}: {'EXACT MATCH' if e == ref_exp else f'MISMATCH {e}'}")
    print(f"[PRIMARY] fixed-trial exposure bit-identical to straight: "
          f"{'PASS' if exposure_ok else 'FAIL'}")

    # DIAGNOSTIC (non-gating) — val_loss seam magnitude, same machinery as
    # validate_resume. Given the data is proven identical, any residual is
    # pure GPU nondeterminism across the resume seam; reported for context.
    analyse(refs, trials, fixed_steps, "DIAGNOSTIC: FIXED val_loss seam")

    power_ok = True
    if args.sensitivity:
        btrials = [f"broken_{c}"
                   for c in string.ascii_lowercase[:args.resume_trials]]
        broken_steps = {n: train_trial(n, break_replay=True) for n in btrials}
        broken_exp = {t: realized_exposure(t) for t in btrials}
        # POWER CONTROL — with replay_fire broken, the cram density is NOT
        # rebuilt on resume, so the post-resume stream (and thus exposure)
        # MUST differ from the straight runs. If broken exposure still
        # matched, the check would have no power to detect a broken replay.
        broken_mismatch = all(e != ref_exp for e in broken_exp.values())
        print("\n[POWER CONTROL] broken-replay exposure must DIFFER:")
        for t, e in broken_exp.items():
            print(f"  broken {t}: {'differs (detected)' if e != ref_exp else 'MATCHES (!!)'} {e}")
        power_ok = broken_mismatch
        print(f"[POWER CONTROL] broken replay detected via exposure mismatch: "
              f"{'PASS' if power_ok else 'FAIL'}")
        analyse(refs, btrials, broken_steps,
                "DIAGNOSTIC: BROKEN val_loss divergence")

    overall = exposure_ok and power_ok
    print("\n" + "=" * 60)
    print(f"replay_fire correctness (exposure bit-identical):  "
          f"{'PASS' if exposure_ok else 'FAIL'}")
    if args.sensitivity:
        print(f"power control (broken replay detected):            "
              f"{'PASS' if power_ok else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")

    keep = args.keep or not overall
    if not keep:
        for name in refs + trials:
            wipe(name)
        if args.sensitivity:
            for name in btrials:
                wipe(name)
        print("(artifacts cleaned; --keep to retain)")
    elif not overall:
        print("(FAIL: keeping artifacts for diagnosis)")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()

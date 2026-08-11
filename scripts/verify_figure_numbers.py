"""Check every number hardcoded in scripts/make_figures.py against run metrics.csv.

A figure is a claim. The figure script transcribes its numbers from the write-up's
§6 owners rather than reading logs at plot time, which is the right call for
reproducibility but means the transcription itself is unverified. This closes that:
it re-reads the step-5000 arithmetic accuracy of every run the figures plot and
diffs it against the constant in the script.

Exit status is nonzero if anything disagrees beyond TOL.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import make_figures as mf  # noqa: E402

TOL = 5e-4   # well under the ~0.005 cross-harness floor quoted in §2.4

# figure constant -> run_id, per size and seed
RUNS = {
    "EARLY": "arith_delay0_exp20_size{size}_seed{seed}",
    "DELAY": "arith_delay50_exp20_size{size}_seed{seed}",
    "TREAT": "arith_delay50_exp20_size{size}_seed{seed}_ivlrshift0at2501",
}

# run_id is the identity of a model, not a label for a configuration
# (week9-stats-backbone.md §5.5.2). `arith_delay50_exp20_size23M_seed1` (0.506) and
# `..._recheck` (0.524) are two *different trained models*; the recheck is the
# canonical Detect null, and 0.506 belongs only to the Week-8 circuit triad, whose C
# arm was forked from that checkpoint's step 2500. Naming the pattern instead of this
# override is how 0.506 would silently end up in a Detect figure.
CANONICAL = {
    ("DELAY", "23M", 1): "arith_delay50_exp20_size23M_seed1_recheck",
}

# the n=1 dose points, 23M seed 1
DOSE_RUNS = {
    25: "arith_delay25_exp20_size23M_seed1",
    50: CANONICAL[("DELAY", "23M", 1)],   # same model as the d50 null above
    55: "arith_delay55_exp20_size23M_seed1",
}


def final_acc(run_id, step=5000):
    path = f"logs/{run_id}/metrics.csv"
    if not os.path.exists(path):
        return None, f"no metrics.csv ({path})"
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    hit = [r for r in rows if int(r["step"]) == step]
    if not hit:
        return None, f"no step {step} row"
    return float(hit[-1]["arith_accuracy"]), None


def main():
    bad, missing, checked = [], [], 0
    for const, pattern in RUNS.items():
        table = getattr(mf, const)
        for size, vals in table.items():
            for i, want in enumerate(vals, start=1):
                run = CANONICAL.get((const, size, i)) or pattern.format(size=size, seed=i)
                got, err = final_acc(run)
                if got is None:
                    missing.append((const, size, i, run, err))
                    continue
                checked += 1
                d = abs(got - want)
                status = "ok " if d <= TOL else "DIFF"
                if d > TOL:
                    bad.append((const, size, i, run, want, got, d))
                print(f"{status} {const:5s} {size:>3s} seed{i}  fig={want:.3f}  "
                      f"log={got:.3f}  d={d:.4f}  {run}")

    for onset, run in DOSE_RUNS.items():
        want = mf.DOSE[onset]
        got, err = final_acc(run)
        if got is None:
            missing.append(("DOSE", "23M", 1, run, err))
            continue
        checked += 1
        d = abs(got - want)
        if d > TOL:
            bad.append(("DOSE", "23M", onset, run, want, got, d))
        print(f"{'ok ' if d <= TOL else 'DIFF'} DOSE  d{onset}       fig={want:.3f}  "
              f"log={got:.3f}  d={d:.4f}  {run}")

    print(f"\nchecked {checked} numbers, {len(bad)} disagree, "
          f"{len(missing)} runs not found")
    for m in missing:
        print(f"  MISSING {m[0]} {m[1]} seed/onset {m[2]}: {m[4]} -> {m[3]}")
    for b in bad:
        print(f"  DIFF {b[0]} {b[1]} {b[2]}: figure {b[4]} vs log {b[5]} "
              f"(delta {b[6]:.4f}) -> {b[3]}")
    return 1 if (bad or missing) else 0


if __name__ == "__main__":
    sys.exit(main())

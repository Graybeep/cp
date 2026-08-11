"""Real power analysis for the syntax size-sweep, answering two questions the
round-number pre-registration hand-waved:

1. SEEDS/SIZE: what N gives ~80% power for the ACTUAL confirm rule
   (Cochran-Armitage trend p<0.05 AND top-size observed rate >= 0.40),
   under several H1 rate scenarios? Also the type-I rate under H0.
2. THRESHOLD: how much does the binary P(high) classification depend on where
   the 0.55-0.57 cutoff falls (i.e. is the rate a stable quantity or an
   artifact of the line)? And does a THRESHOLD-FREE continuous trend test
   (Spearman LD~size) recover the signal without needing a cutoff?
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20260719)
N_SIM = 4000
SIZES = np.array([0.0, 1.0, 2.0])  # ordered size index (11.5M, 23M, 60M)

# Observed structure at 23M: low cluster N(0.534, 0.009); high draw ~0.583.
LOW_MU, LOW_SD = 0.534, 0.009
HIGH_MU, HIGH_SD = 0.583, 0.012

def sample_group(n, rate):
    """n seeds; each is 'high' w.p. rate. Returns continuous LD scores."""
    is_high = rng.random(n) < rate
    lo = rng.normal(LOW_MU, LOW_SD, n)
    hi = rng.normal(HIGH_MU, HIGH_SD, n)
    return np.where(is_high, hi, lo)

def ca_trend_z(counts_hi, counts_n, x=SIZES):
    """Cochran-Armitage trend z (positive = rate rises with size)."""
    N = counts_n.sum(); R = counts_hi.sum()
    if R == 0 or R == N:
        return 0.0
    p = R / N
    T = np.sum(x * (counts_hi - counts_n * p))
    var = p * (1 - p) * (np.sum(counts_n * x**2) - (np.sum(counts_n * x))**2 / N)
    return 0.0 if var <= 0 else T / np.sqrt(var)

def run(N, rates, thr=0.56):
    ca_reject = 0        # CA trend two-sided p<0.05
    confirm40 = 0        # CA one-sided p<0.05 AND top observed rate >= 0.40
    confirm50 = 0        # CA one-sided p<0.05 AND top observed rate >= 0.50
    spear_reject = 0     # continuous Spearman two-sided p<0.05
    for _ in range(N_SIM):
        groups = [sample_group(N, r) for r in rates]
        hi = np.array([(g >= thr).sum() for g in groups])
        nn = np.array([N, N, N])
        z = ca_trend_z(hi, nn)
        p_two = 2 * (1 - stats.norm.cdf(abs(z)))
        p_one = 1 - stats.norm.cdf(z)
        ca_reject += p_two < 0.05
        confirm40 += (p_one < 0.05) and (hi[-1] / N) >= 0.40
        confirm50 += (p_one < 0.05) and (hi[-1] / N) >= 0.50
        allLD = np.concatenate(groups)
        idx = np.repeat(SIZES, N)
        _, ps = stats.spearmanr(idx, allLD)
        spear_reject += ps < 0.05
    f = N_SIM
    return dict(ca=ca_reject/f, c40=confirm40/f, c50=confirm50/f,
               spearman=spear_reject/f)

scenarios = {
    "H0 flat        (.12,.12,.12)": [0.12, 0.12, 0.12],
    "A threshold    (.12,.12,.40)": [0.12, 0.12, 0.40],
    "B graded       (.12,.26,.40)": [0.12, 0.26, 0.40],
    "C strong       (.12,.40,.70)": [0.12, 0.40, 0.70],
}
print(f"{N_SIM} sims/cell. Power = P(reject/confirm). alpha=0.05.\n")
for N in [12, 18, 24, 30, 40]:
    print(f"--- N = {N} seeds/size ---")
    for name, rates in scenarios.items():
        r = run(N, rates)
        tag = "(type-I)" if name.startswith("H0") else ""
        print(f"  {name}: CA_trend={r['ca']:.2f}  confirm@.40={r['c40']:.2f}"
              f"  confirm@.50={r['c50']:.2f}  Spearman={r['spearman']:.2f} {tag}")
    print()

# Threshold sensitivity: does the classified rate wobble with the cutoff?
print("=== Threshold sensitivity of P(high) at the 60M rate, N=24 ===")
for true_rate in [0.12, 0.25, 0.40, 0.70]:
    row = []
    for thr in [0.55, 0.56, 0.57, 0.58]:
        g = np.concatenate([sample_group(24, true_rate) for _ in range(400)])
        row.append((g >= thr).mean())
    print(f"  true_rate={true_rate:.2f}: classified@[.55,.56,.57,.58] = "
          + ", ".join(f"{v:.3f}" for v in row))

# Wilson CI half-width for an observed 0.40 at various N (the user's point).
print("\n=== Wilson 95% CI for observed rate = 0.40 ===")
for N in [12, 18, 24, 30, 40]:
    k = round(0.40 * N)
    lo, hi = stats.binomtest(k, N, 0.4).proportion_ci(method="wilson")
    print(f"  N={N}: {k}/{N}={k/N:.2f}  CI=[{lo:.2f},{hi:.2f}]  "
          f"(overlaps .12? {lo<=0.12}; overlaps .25? {lo<=0.25<=hi})")

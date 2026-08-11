# Week 4 arith reopening pilot — read-out

Common checkpoint: arith_delay50_exp20_size23M_seed1 @ step 2500 (sha 1fa5e0a2...). All arms fork from it.
Anchors (pre-registered): delay-0 0.798@48.19 ppl | d50 null 0.506@50.83 ppl | gap 0.292
Skill gate: PRIMARY >= 0.718, STRICT >= 0.769. Forgetting gate: c-b <= +1.0 ppl.

| arm | exact | recovery% | val_ppl | iv cost c-b | skill | forget |
|-----|-------|-----------|---------|-------------|-------|--------|
| null (source) | 0.506 | +0% | 50.83 | +0.00 | no | OK |
| reset (emb) | 0.586 | +27% | 52.15 | +1.32 | no | FLAG |
| reset+spike | 0.576 | +24% | 51.82 | +0.99 | no | OK |
| spike | 0.536 | +10% | 50.62 | -0.21 | no | OK |
| freeze (emb) | 0.332 | -60% | 51.91 | +1.08 | no | FLAG |
| cram | 0.574 | +23% | 50.98 | +0.15 | no | OK |
| lr_shift (seg 0) | 0.752 | +84% | 49.86 | -0.97 | PRIMARY | OK |

## Per-band exact-match (digits x carry) at step 5000

```
              null (source)  reset (emb)  reset+spike  spike  freeze (emb)  cram  lr_shift (seg 0)
digits carry                                                                                      
2      False          0.762        0.746        0.794  0.714         0.524 0.698             0.921
       True           0.587        0.762        0.730  0.587         0.460 0.667             0.873
3      False          0.587        0.630        0.679  0.582         0.429 0.614             0.815
       True           0.558        0.564        0.583  0.654         0.308 0.551             0.814
4      False          0.490        0.660        0.601  0.553         0.324 0.565             0.727
       True           0.278        0.349        0.395  0.409         0.185 0.445             0.605
```
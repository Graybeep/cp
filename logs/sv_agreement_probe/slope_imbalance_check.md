# Seed-imbalance robustness of the mixture slope tests

Arithmetic mixture is (3, 3, 8) seeds across 11.5M / 23.2M / 60.3M;
syntax mixture is (24, 24, 24) and serves as the balanced reference.
All estimators use the same model-level points; they differ only in what
they assume about residual variance and how sizes are weighted.

## arith — attraction (n = 14)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 3 | +0.448 | 0.106 | 0.254 |
| 23M | 3 | +0.260 | 0.036 | 0.095 |
| 60M | 8 | +0.266 | 0.092 | 0.119 |

SD ratio max/min = 2.91, Levene p = 0.460

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | -0.207 | 0.086 | 0.0333 |
| HC3 robust | -0.207 | 0.105 | 0.0726 |
| WLS 1/s²_group | -0.112 | 0.096 | 0.2620 |
| group-mean OLS (df=1) | -0.237 | 0.177 | 0.4071 |
| stratified bootstrap | -0.208 | CI -0.350, -0.061 | 0.0053 |
| balanced (3,3,3) subsamples | -0.237 (range -0.362…-0.095) | — | median 0.0660, 43% < 0.05 |

## arith — complexity (n = 14)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 3 | +0.155 | 0.064 | 0.254 |
| 23M | 3 | +0.130 | 0.023 | 0.095 |
| 60M | 8 | +0.077 | 0.042 | 0.119 |

SD ratio max/min = 2.76, Levene p = 0.675

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | -0.112 | 0.038 | 0.0122 |
| HC3 robust | -0.112 | 0.053 | 0.0554 |
| WLS 1/s²_group | -0.120 | 0.039 | 0.0099 |
| group-mean OLS (df=1) | -0.110 | 0.013 | 0.0729 |
| stratified bootstrap | -0.112 | CI -0.185, -0.028 | 0.0072 |
| balanced (3,3,3) subsamples | -0.110 (range -0.156…-0.055) | — | median 0.0742, 36% < 0.05 |

## arith — paired (n = 14)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 3 | +0.293 | 0.166 | 0.254 |
| 23M | 3 | +0.131 | 0.059 | 0.095 |
| 60M | 8 | +0.189 | 0.090 | 0.119 |

SD ratio max/min = 2.80, Levene p = 0.390

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | -0.095 | 0.100 | 0.3608 |
| HC3 robust | -0.095 | 0.142 | 0.5142 |
| WLS 1/s²_group | +0.038 | 0.105 | 0.7270 |
| group-mean OLS (df=1) | -0.128 | 0.189 | 0.6216 |
| stratified bootstrap | -0.095 | CI -0.299, +0.100 | 0.3698 |
| balanced (3,3,3) subsamples | -0.128 (range -0.256…+0.003) | — | median 0.3952, 0% < 0.05 |

## syntax — attraction (n = 72)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 24 | +0.430 | 0.071 | 0.032 |
| 23M | 24 | +0.314 | 0.081 | 0.014 |
| 60M | 24 | +0.201 | 0.056 | 0.037 |

SD ratio max/min = 1.46, Levene p = 0.192

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | -0.316 | 0.028 | 0.0000 |
| HC3 robust | -0.316 | 0.026 | 0.0000 |
| WLS 1/s²_group | -0.314 | 0.025 | 0.0000 |
| group-mean OLS (df=1) | -0.316 | 0.030 | 0.0601 |
| stratified bootstrap | -0.316 | CI -0.365, -0.268 | 0.0000 |
| balanced (3,3,3) subsamples | -0.345 (range -0.409…-0.265) | — | median 0.0000, 100% < 0.05 |

## syntax — complexity (n = 72)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 24 | +0.134 | 0.043 | 0.032 |
| 23M | 24 | +0.131 | 0.057 | 0.014 |
| 60M | 24 | +0.145 | 0.055 | 0.037 |

SD ratio max/min = 1.34, Levene p = 0.302

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | +0.016 | 0.021 | 0.4310 |
| HC3 robust | +0.016 | 0.020 | 0.4198 |
| WLS 1/s²_group | +0.015 | 0.020 | 0.4515 |
| group-mean OLS (df=1) | +0.016 | 0.012 | 0.4059 |
| stratified bootstrap | +0.016 | CI -0.021, +0.055 | 0.4043 |
| balanced (3,3,3) subsamples | +0.005 (range -0.045…+0.088) | — | median 0.6403, 0% < 0.05 |

## syntax — paired (n = 72)

| size | n | mean | SD | leverage/pt |
|:--|--:|--:|--:|--:|
| 11M | 24 | +0.296 | 0.103 | 0.032 |
| 23M | 24 | +0.183 | 0.121 | 0.014 |
| 60M | 24 | +0.056 | 0.094 | 0.037 |

SD ratio max/min = 1.29, Levene p = 0.444

| estimator | slope/decade | SE | p |
|:--|--:|--:|--:|
| OLS (published) | -0.332 | 0.042 | 0.0000 |
| HC3 robust | -0.332 | 0.040 | 0.0000 |
| WLS 1/s²_group | -0.332 | 0.039 | 0.0000 |
| group-mean OLS (df=1) | -0.332 | 0.018 | 0.0341 |
| stratified bootstrap | -0.332 | CI -0.407, -0.257 | 0.0000 |
| balanced (3,3,3) subsamples | -0.350 (range -0.484…-0.246) | — | median 0.0000, 100% < 0.05 |

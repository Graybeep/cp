# Parameter counts — recomputed from checkpoint state dicts

Source: `scripts/param_counts.py`. One delay-0 seed-1 arithmetic run per size,
step-5000 checkpoint. Counts are over every tensor in the saved `state_dict`,
so buffers-as-parameters would show up; `wte.weight` and `lm_head.weight` are
split out as the embedding-and-head share.

| tag | run | layers | heads | width | vocab | tensors | total | non-embedding | embed+head | embed share |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 11M | `arith_delay0_exp20_size11M_seed1` | 4 | 8 | 256 | 16384 | 27 | 11,536,640 | 3,148,032 | 8,388,608 | 72.7% |
| 23M | `arith_delay0_exp20_size23M_seed1` | 6 | 8 | 384 | 16384 | 39 | 23,204,736 | 10,621,824 | 12,582,912 | 54.2% |
| 60M | `arith_delay0_exp20_size60M_seed1` | 8 | 10 | 640 | 16384 | 51 | 60,304,000 | 39,332,480 | 20,971,520 | 34.8% |

Total-parameter span 5.23x; non-embedding span 12.49x.

Training config as stored in the same checkpoints:

| tag | ctx | total steps | checkpoint interval | tokens |
|:--|--:|--:|--:|--:|
| 11M | 512 | 5000 | 1250 | 81,920,000 |
| 23M | 512 | 5000 | 100 | 81,920,000 |
| 60M | 512 | 5000 | 1250 | 81,920,000 |

```json
[
  {
    "tag": "11M",
    "run": "arith_delay0_exp20_size11M_seed1",
    "n_tensors": 27,
    "total": 11536640,
    "embed": 8388608,
    "non_embed": 3148032,
    "embed_share": 0.7271274825252413,
    "vocab": 16384,
    "n_embd": 256,
    "n_layer": 4,
    "n_head": 8,
    "block_size": 512,
    "step": 5000,
    "total_steps": 5000,
    "checkpoint_interval": 1250,
    "tokens": 81920000
  },
  {
    "tag": "23M",
    "run": "arith_delay0_exp20_size23M_seed1",
    "n_tensors": 39,
    "total": 23204736,
    "embed": 12582912,
    "non_embed": 10621824,
    "embed_share": 0.5422562014926608,
    "vocab": 16384,
    "n_embd": 384,
    "n_layer": 6,
    "n_head": 8,
    "block_size": 512,
    "step": 5000,
    "total_steps": 5000,
    "checkpoint_interval": 100,
    "tokens": 81920000
  },
  {
    "tag": "60M",
    "run": "arith_delay0_exp20_size60M_seed1",
    "n_tensors": 51,
    "total": 60304000,
    "embed": 20971520,
    "non_embed": 39332480,
    "embed_share": 0.34776333244892543,
    "vocab": 16384,
    "n_embd": 640,
    "n_layer": 8,
    "n_head": 10,
    "block_size": 512,
    "step": 5000,
    "total_steps": 5000,
    "checkpoint_interval": 1250,
    "tokens": 81920000
  }
]
```

"""Parameter counts for the three configurations, read from checkpoint state dicts.

Ground truth for the §2.1 table: counts come from the saved tensors of a trained
model, not from a formula over `config.json` or from `reopen/model.py`. If the code
and the shipped weights ever disagree, the weights are what ran.

One delay-0 seed-1 arithmetic run per size, at step 5000.

Writes logs/param_counts.md.
"""
import json
import os

import torch

RUNS = {
    "11M": "checkpoints/arith_delay0_exp20_size11M_seed1/step_0005000.pt",
    "23M": "checkpoints/arith_delay0_exp20_size23M_seed1/step_0005000.pt",
    "60M": "checkpoints/arith_delay0_exp20_size60M_seed1/step_0005000.pt",
}
OUT = "logs/param_counts.md"

# state-dict key prefixes that are embedding-table or output-projection weights
EMBED_KEYS = ("wte.weight", "lm_head.weight")


def state_dict_of(ckpt):
    for k in ("model", "model_state_dict", "state_dict"):
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    raise KeyError(f"no state dict in checkpoint; keys = {list(ckpt)[:12]}")


def main():
    rows = []
    for tag, path in RUNS.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = state_dict_of(ckpt)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

        total = sum(v.numel() for v in sd.values())
        embed = sum(v.numel() for k, v in sd.items() if k in EMBED_KEYS)
        missing = [k for k in EMBED_KEYS if k not in sd]
        if missing:
            raise KeyError(f"{tag}: expected keys absent: {missing}")

        mcfg = ckpt.get("model_config", {})
        tcfg = ckpt.get("train_config", {})
        rows.append({
            "tag": tag,
            "run": os.path.basename(os.path.dirname(path)),
            "n_tensors": len(sd),
            "total": total,
            "embed": embed,
            "non_embed": total - embed,
            "embed_share": embed / total,
            "vocab": sd["wte.weight"].shape[0],
            "n_embd": sd["wte.weight"].shape[1],
            # layer count from the weights, head count from the stored config
            # (head count is not recoverable from tensor shapes)
            "n_layer": len({k.split(".")[1] for k in sd if k.startswith("blocks.")}),
            "n_head": mcfg.get("n_head"),
            "block_size": mcfg.get("block_size"),
            "step": ckpt.get("step"),
            "total_steps": tcfg.get("total_steps"),
            "checkpoint_interval": tcfg.get("checkpoint_interval"),
            "tokens": (tcfg.get("total_steps", 0) * tcfg.get("micro_batch_size", 0)
                       * tcfg.get("grad_accum_steps", 0) * mcfg.get("block_size", 0)),
        })
        # the stored config must agree with the weights it was saved beside
        assert mcfg.get("n_layer") == rows[-1]["n_layer"], f"{tag}: n_layer mismatch"
        assert mcfg.get("n_embd") == rows[-1]["n_embd"], f"{tag}: n_embd mismatch"
        assert mcfg.get("vocab_size") == rows[-1]["vocab"], f"{tag}: vocab mismatch"

    w = max(r["non_embed"] for r in rows) / min(r["non_embed"] for r in rows)
    t = max(r["total"] for r in rows) / min(r["total"] for r in rows)

    lines = [
        "# Parameter counts — recomputed from checkpoint state dicts",
        "",
        "Source: `scripts/param_counts.py`. One delay-0 seed-1 arithmetic run per size,",
        "step-5000 checkpoint. Counts are over every tensor in the saved `state_dict`,",
        "so buffers-as-parameters would show up; `wte.weight` and `lm_head.weight` are",
        "split out as the embedding-and-head share.",
        "",
        "| tag | run | layers | heads | width | vocab | tensors | total | "
        "non-embedding | embed+head | embed share |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['tag']} | `{r['run']}` | {r['n_layer']} | {r['n_head']} | "
            f"{r['n_embd']} | {r['vocab']} | {r['n_tensors']} | {r['total']:,} | "
            f"{r['non_embed']:,} | {r['embed']:,} | {r['embed_share']:.1%} |"
        )
    lines += [
        "",
        f"Total-parameter span {t:.2f}x; non-embedding span {w:.2f}x.",
        "",
        "Training config as stored in the same checkpoints:",
        "",
        "| tag | ctx | total steps | checkpoint interval | tokens |",
        "|:--|--:|--:|--:|--:|",
    ]
    for r in rows:
        lines.append(f"| {r['tag']} | {r['block_size']} | {r['total_steps']} | "
                     f"{r['checkpoint_interval']} | {r['tokens']:,} |")
    lines += [
        "",
        "```json",
        json.dumps(rows, indent=2),
        "```",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))
    for r in rows:
        print(f"{r['tag']}: total {r['total']:,}  non-emb {r['non_embed']:,}  "
              f"embed {r['embed']:,} ({r['embed_share']:.1%})  "
              f"L={r['n_layer']} H={r['n_head']} d={r['n_embd']} V={r['vocab']}  "
              f"steps={r['total_steps']} ckpt_every={r['checkpoint_interval']} "
              f"tokens={r['tokens']:,}")
    print(f"spans: total {t:.3f}x  non-embedding {w:.3f}x")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

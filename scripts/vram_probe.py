"""VRAM fit check for the ~60M syntax size-sweep config (option B).
Mirrors real training: bf16 autocast, micro_batch=8 x seq=512, grad_accum=4,
fused AdamW step. Reports peak torch.cuda.max_memory_allocated AND reserved,
so 'should fit by arithmetic' becomes 'confirmed fits' before the sweep runs.
"""
import sys, torch
sys.path.insert(0, "D:/EXPERIMENTS/reopen")
from reopen.model import GPT
from reopen.config import ModelConfig

assert torch.cuda.is_available(), "no CUDA"
dev = "cuda"
props = torch.cuda.get_device_properties(0)
total_gb = props.total_memory / 1e9
print(f"GPU: {props.name}  total={total_gb:.2f} GB")

MB, T, ACCUM = 8, 512, 4  # real training values

def probe(n_layer, n_embd, n_head, tag):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = ModelConfig(vocab_size=16384, n_layer=n_layer, n_embd=n_embd,
                      n_head=n_head, block_size=512)
    model = GPT(cfg).to(dev)
    nparams = model.num_params()
    opt = model.configure_optimizer(6e-4, 0.1, (0.9, 0.95), dev)
    # two full optimizer steps so Adam moment buffers are allocated (peak is
    # only realistic after the first step materializes them).
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        for _ in range(ACCUM):
            x = torch.randint(0, cfg.vocab_size, (MB, T), device=dev)
            y = torch.randint(0, cfg.vocab_size, (MB, T), device=dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / 1e9
    peak_resv = torch.cuda.max_memory_reserved() / 1e9
    print(f"[{tag}] {nparams/1e6:.1f}M params  "
          f"peak_allocated={peak_alloc:.2f} GB  peak_reserved={peak_resv:.2f} GB  "
          f"headroom(vs {total_gb:.1f})={total_gb-peak_resv:.2f} GB")
    del model, opt
    torch.cuda.empty_cache()

# Primary ~60M candidate + a slightly larger guard, and the current 23M as a
# sanity anchor (should match the known working footprint).
probe(6, 384, 8,  "23M anchor")
probe(8, 640, 10, "~60M candidate (8L/640d/10h)")

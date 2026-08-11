"""Empirically answer Bug 1(b): when a torch.save is killed mid-write and
leaves a truncated .pt, does torch.load ALWAYS hard-fail, or can it return a
file that passes a key-set check while holding bad tensor data?

Method: take a real project checkpoint, write copies truncated at several
byte lengths (simulating a kill at different points of the save), and for
each try torch.load(weights_only=False). Record: did it load? did it raise?
if it loaded, do the top-level keys match a full load, and are the model
tensors finite / correctly shaped?
"""
import os, sys, traceback
import torch

src = sys.argv[1]
full_size = os.path.getsize(src)
print(f"source: {src}  ({full_size:,} bytes)")

# First, a clean full load for the reference key-set.
ref = torch.load(src, map_location="cpu", weights_only=False)
ref_keys = set(ref.keys())
ref_model_keys = set(ref["model"].keys())
print(f"full load OK. top-level keys={sorted(ref_keys)}  n_model_tensors={len(ref_model_keys)}")
del ref

# Truncation points: bytes REMOVED from the end (a kill leaves a prefix).
# Include near-complete (most adversarial for 'silently loadable') and gross.
removed = [4, 64, 4096, 65536, int(full_size*0.01), int(full_size*0.5)]
tmpdir = os.path.dirname(os.path.abspath(sys.argv[1] if False else __file__))
scratch = os.environ.get("SCRATCH", ".")

with open(src, "rb") as f:
    head = f.read()  # full bytes in memory once

for rem in removed:
    keep = full_size - rem
    tp = os.path.join(scratch, f"trunc_keep_{keep}.pt")
    with open(tp, "wb") as g:
        g.write(head[:keep])
    label = f"removed last {rem:>10,} bytes (kept {keep:,}, {100*keep/full_size:.4f}%)"
    try:
        obj = torch.load(tp, map_location="cpu", weights_only=False)
        # It LOADED. Now: does it pass a naive key-set check?
        keyset_ok = set(obj.keys()) == ref_keys and set(obj["model"].keys()) == ref_model_keys
        # And are the actual tensors intact?
        bad = 0
        for k, v in obj["model"].items():
            if not torch.is_tensor(v) or not torch.isfinite(v).all():
                bad += 1
        print(f"[LOADED]  {label}  keyset_ok={keyset_ok}  bad_tensors={bad}  "
              f"<-- ***SILENT BAD DATA POSSIBLE***" if keyset_ok and bad else
              f"[LOADED]  {label}  keyset_ok={keyset_ok}  bad_tensors={bad}")
    except Exception as e:
        print(f"[RAISED]  {label}  {type(e).__name__}: {str(e)[:80]}")
    finally:
        if os.path.exists(tp):
            os.remove(tp)

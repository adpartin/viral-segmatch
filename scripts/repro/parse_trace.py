import json, glob
from collections import defaultdict

f = glob.glob("models/flu/July_2025/runs/phase1_profile/profile/*.pt.trace.json")[0]
d = json.load(open(f))
ev = d["traceEvents"]

cat_us = defaultdict(float)
op_us = defaultdict(float)
gpu = []  # (start, end, cat)
for e in ev:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    dur = e.get("dur", 0)
    cat_us[cat] += dur
    if cat == "cpu_op":
        op_us[e["name"]] += dur
    if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        ts = e.get("ts", 0)
        gpu.append((ts, ts + dur, cat))

# GPU busy = union of GPU-side intervals; span = first kernel start .. last kernel end
gpu.sort()
busy = 0.0
lo0 = gpu[0][0]; hi0 = gpu[-1][1]
cl = ch = None
for lo, hi, _ in gpu:
    if ch is None: cl, ch = lo, hi
    elif lo <= ch: ch = max(ch, hi)
    else: busy += ch - cl; cl, ch = lo, hi
busy += ch - cl
span = hi0 - lo0
kern = sum(hi - lo for lo, hi, c in gpu if c == "kernel")
memcpy = sum(hi - lo for lo, hi, c in gpu if c == "gpu_memcpy")

ms = lambda x: x / 1000.0
print(f"trace: {f.split('/')[-1]}")
print(f"\n=== GPU timeline over the profiled window ===")
print(f"window span      : {ms(span):8.1f} ms")
print(f"GPU busy (union) : {ms(busy):8.1f} ms   ({100*busy/span:.0f}% of window)")
print(f"GPU IDLE         : {ms(span-busy):8.1f} ms   ({100*(span-busy)/span:.0f}% of window)  <- host-bound signal")
print(f"  of GPU busy: kernels {ms(kern):.1f} ms, H2D/D2H memcpy {ms(memcpy):.1f} ms")
print(f"\n=== time by category (sum of durations) ===")
for c, v in sorted(cat_us.items(), key=lambda x: -x[1]):
    print(f"  {c:16s} {ms(v):8.1f} ms")
print(f"\n=== top CPU ops by total time ===")
for n, v in sorted(op_us.items(), key=lambda x: -x[1])[:12]:
    print(f"  {ms(v):8.1f} ms  {n}")

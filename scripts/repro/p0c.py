import csv, glob, re, statistics as st
from pathlib import Path

ROOT = Path("/lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch")
RUNS = ROOT / "models/flu/July_2025/runs"
LOGS = ROOT / "logs/phase0c"

def epoch_series(fp):
    rows = list(csv.DictReader(open(fp)))
    def col(n): return [float(r[n]) for r in rows if r.get(n) not in (None, '')]
    return col('epoch_time_sec'), col('data_time_sec'), col('compute_time_sec'), col('eval_time_sec')

def elapsed_sec(logfp):
    # parse "Elapsed Time: HH:MM:SS"
    try:
        for line in open(logfp):
            m = re.search(r'Elapsed Time:\s*(\d+):(\d+):(\d+)', line)
            if m:
                h, mi, s = map(int, m.groups()); return h*3600 + mi*60 + s
    except FileNotFoundError:
        pass
    return None

# phase0b K=1 (4-share, single node) baseline for reference
def k1():
    ets = []
    for f in sorted(glob.glob(str(RUNS / "phase0b_share4_g*/training_history.csv"))):
        et, *_ = epoch_series(f)
        if len(et) > 1: ets.append(st.median(et[1:]))
    return st.mean(ets) if ets else None

print(f"{'K (nodes)':>10} {'procs':>6} {'per-epoch median':>18} {'data':>6} {'compute':>8} {'eval':>6} {'startup(s)':>11}")
k1v = k1()
if k1v: print(f"{'1':>10} {4:>6} {k1v:>17.1f}s {'-':>6} {'-':>8} {'-':>6} {'-':>11}   (phase0b 4-share)")

for K in [2, 4, 8]:
    per_ep, datas, comps, evals, starts = [], [], [], [], []
    dirs = sorted(glob.glob(str(RUNS / f"phase0c_{K}n_*_g*/training_history.csv")))
    for f in dirs:
        et, dt, ct, ev = epoch_series(f)
        if len(et) < 2: continue
        per_ep.append(st.median(et[1:])); datas.append(st.median(dt[1:]))
        comps.append(st.median(ct[1:])); evals.append(st.median(ev[1:]))
        # startup = total elapsed - sum(epoch times)
        run = Path(f).parent.name          # phase0c_{K}n_{NODE}_g{g}
        m = re.match(r'phase0c_\d+n_(.+)_g(\d)', run)
        log = LOGS / f"{K}n_{m.group(1)}_g{m.group(2)}.log" if m else None
        el = elapsed_sec(log) if log else None
        if el: starts.append(el - sum(et))
    if not per_ep:
        print(f"{K:>10} — no data"); continue
    procs = K * 4
    su = st.median(starts) if starts else float('nan')
    print(f"{K:>10} {procs:>6} {st.mean(per_ep):>17.1f}s {st.mean(datas):>6.1f} {st.mean(comps):>8.1f} "
          f"{st.mean(evals):>6.1f} {su:>10.0f}s   (n={len(per_ep)} folds)")

print("\nApril production reference: per-epoch 25.4s (28 nodes / 112 procs)")

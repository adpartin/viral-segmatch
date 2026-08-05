import csv, glob, statistics as st
from pathlib import Path

ROOT = Path("/lus/eagle/projects/IMPROVE_Aim1/apartin/viral-segmatch")

def epoch_stats(fp):
    rows = list(csv.DictReader(open(fp)))
    def col(n): return [float(r[n]) for r in rows if r.get(n) not in (None, '')]
    et, dt, ct, ev = col('epoch_time_sec'), col('data_time_sec'), col('compute_time_sec'), col('eval_time_sec')
    med = lambda x: st.median(x) if x else float('nan')
    return len(et), med(et[1:]), med(dt[1:]), med(ct[1:]), med(ev[1:])  # skip warmup epoch 0

# SOLO
f = ROOT / "models/flu/July_2025/runs/phase0b_solo/training_history.csv"
n, solo_et, sdt, sct, sev = epoch_stats(f)
print(f"SOLO      : epochs={n}  total={solo_et:.1f}s  (data {sdt:.2f} / compute {sct:.2f} / eval {sev:.2f})")

# 4-SHARE
sh = []
for f in sorted(glob.glob(str(ROOT / "models/flu/July_2025/runs/phase0b_share4_g*/training_history.csv"))):
    n, et, dt, ct, ev = epoch_stats(f)
    print(f"{Path(f).parent.name}: epochs={n}  total={et:.1f}s  (data {dt:.2f} / compute {ct:.2f} / eval {ev:.2f})")
    sh.append(et)

if sh:
    m = st.mean(sh)
    cont = m / solo_et
    print(f"\nmean 4-share/epoch = {m:.1f}s ; solo = {solo_et:.1f}s")
    print(f"CONTENTION factor  = {cont:.2f}x")
    # throughput (folds-epochs per second)
    thr_solo = 1.0 / solo_et            # 1 fold on the node
    thr_4     = 4.0 / m                  # 4 folds on the node
    thr_ideal = 4.0 / solo_et           # 4 folds, zero contention
    print(f"node throughput: 1-fold={thr_solo:.3f}  4-share={thr_4:.3f}  ideal-4={thr_ideal:.3f} folds-ep/s")
    print(f"packing efficiency (4share/ideal) = {thr_4/thr_ideal*100:.0f}%   (= 1/contention)")
    print(f"4-share vs 1-fold-per-node        = {thr_4/thr_solo:.2f}x")

# dmon
def dmon_stats(fp, gpus):
    if not Path(fp).exists(): return None
    per = {g: [] for g in gpus}
    for line in open(fp):
        line = line.strip()
        if line.startswith('#') or not line: continue
        p = line.split()
        if len(p) >= 5 and p[2] in per:
            try: per[p[2]].append((float(p[3]), float(p[4])))
            except ValueError: pass
    return per

print("\n=== GPU utilization ===")
for label, fp, gpus in [("solo", ROOT/"logs/phase0b/dmon_solo.csv", ['0']),
                        ("4share", ROOT/"logs/phase0b/dmon_4share.csv", ['0','1','2','3'])]:
    per = dmon_stats(fp, gpus)
    if not per: print(f"{label}: dmon missing"); continue
    for g in gpus:
        vals = per[g]
        act = [(s, m) for s, m in vals if s > 0]
        if act:
            sm = [s for s, _ in act]; mem = [m for _, m in act]
            print(f"{label} GPU{g}: sm%% median={st.median(sm):.0f} max={max(sm):.0f} | mem%% median={st.median(mem):.0f}")

import os, glob, pyarrow.parquet as pq
print("CPU", os.cpu_count())
os.system("free -g | head -2; df -h /kaggle/working | tail -1")
for r, d, f in os.walk('/kaggle/input'):
    if len(f) < 5: print(r, f)
    else: print(r, len(f), sorted(f)[:2])
    if r.count('/') > 6: break
p = glob.glob('/kaggle/input/**/*.parquet', recursive=True)[0]
pf = pq.ParquetFile(p)
print("ROWS", pf.metadata.num_rows, "RG", pf.num_row_groups)
print(pf.schema_arrow)
t = pf.read_row_group(0, columns=[c for c in pf.schema_arrow.names][:40]).slice(0, 3).to_pandas()
for c in t.columns:
    v = t[c].iloc[0]
    print("COL", c, type(v).__name__, (len(v) if hasattr(v, '__len__') and not isinstance(v, str) else v))

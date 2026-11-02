import glob, pandas as pd, collections
r = glob.glob('/kaggle/input/**/FeatureAndMetadata_Milling.csv', recursive=True)[0]
d = pd.read_csv(r, sep=';', header=1, low_memory=False)
cols = list(d.columns)
stats = collections.Counter(c.split(' - ')[-1] for c in cols if ' - ' in c)
print("STATS", dict(stats))
print("SENSORCH", sorted({' - '.join(c.split(' - ')[:2]) for c in cols if ' - ' in c}))
meta = [c for c in cols if ' - ' not in c]
print("META", meta)
print(d[meta].head(3).to_string())
key = [c for c in meta if 'ool' in c or 'ample' in c or 'OC' in c or 'ard' in c or 'ycle' in c or 'ype' in c]
print("KEY", key)
for c in meta:
    u = d[c].nunique()
    print("U", c, u, list(d[c].unique()[:8]) if u < 30 else '')

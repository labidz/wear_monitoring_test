import glob, os, pandas as pd
r = glob.glob('/kaggle/input/**/FeatureAndMetadata_Milling.csv', recursive=True)[0]
print('HEAD', open(r).read(400).replace(chr(10),' || '))
d = pd.read_csv(r, sep=None, engine='python')
print("SHAPE", d.shape)
print("COLS", "|".join(d.columns))
meta = [c for c in d.columns if not any(s in c.lower() for s in ["rms", "mean", "std", "kurt", "skew", "peak", "max", "min", "crest", "energy", "freq", "var"])]
print("META", meta)
print(d[meta].drop_duplicates(subset=[c for c in meta if 'tool' in c.lower()][:1] or None).head(20).to_string())
f = sorted(glob.glob('/kaggle/input/**/P*_F*_C*.csv', recursive=True))[0]
with open(f) as fh:
    for i in range(3):
        print("RAW", fh.readline()[:600])

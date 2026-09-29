"""
Miért közeliek a sharpness-átlagok? — szórás és hatásméret.

Futtatás (ahol az ejszakai_emo.csv van):
    python sharp_diag.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

pd.set_option("display.width", 200)

df = pd.read_csv("ejszakai_emo.csv")
val = pd.read_csv("ejszakai_emo_valencia.csv")
val["filename"] = val["path"].map(lambda p: Path(p).name)
df = df.merge(val[["filename", "value"]], on="filename").rename(
    columns={"value": "valence"})
df["csoport"] = pd.qcut(df["valence"], 3,
                        labels=["kellemetlen", "semleges", "kellemes"])

s = df["sharpness_S5"]
print(f"n = {len(df)}\n")
print("1) A SHARPNESS TELJES ELOSZLASA")
print(f"   min {s.min():.2f} | 10% {s.quantile(.1):.2f} | median {s.median():.2f}"
      f" | 90% {s.quantile(.9):.2f} | max {s.max():.2f}   (szoras {s.std():.2f})")

print("\n2) CSOPORTONKENT: atlag vs szoras")
g = df.groupby("csoport", observed=True)["sharpness_S5"]
print(pd.DataFrame({"atlag": g.mean(), "szoras": g.std(), "n": g.size()}).round(3))

print("\n3) HATASMERET (Cohen-d): semleges vs a ket szelso csoport")
sem = df.loc[df.csoport == "semleges", "sharpness_S5"]
sze = df.loc[df.csoport != "semleges", "sharpness_S5"]
pooled = np.sqrt((sem.var() + sze.var()) / 2)
d = (sze.mean() - sem.mean()) / pooled
print(f"   d = {d:.2f}   (0.2 = kicsi, 0.5 = kozepes, 0.8 = nagy)")

r, p = spearmanr(s, df["valence"].abs())
print(f"\n4) MAGYARAZOTT VARIANCIA: r = {r:.3f}  ->  r^2 = {100*r*r:.1f}%")
print("   Vagyis a sharpness a kiemelkedoseg ennyi szazalekat magyarazza.")

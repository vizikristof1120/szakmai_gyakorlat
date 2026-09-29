"""
Diagnosztika az emo_features.csv-hez.

Futtatás (ugyanabban a mappában, ahol az emo_features.csv van):
    python diag.py
"""

import pandas as pd
from scipy.stats import spearmanr

d = pd.read_csv("emo_features.csv")
COLS = ["loudness_N5", "sharpness_S5", "roughness_R5", "crest_db"]

print(f"Sorok: {len(d)}")
print(f"Oszlopok: {list(d.columns)}\n")

# --- 1. A valencia és az arousal ugyanaz-e? ------------------------------
# Ha |r| > 0.9, akkor a két CSV-t valószínűleg ugyanarra az oszlopra
# párosította a kód -> rossz beolvasás.
if "arousal" in d.columns:
    r, p = spearmanr(d["valence"], d["arousal"])
    print(f"1) valencia vs arousal:  r={r:+.3f}  p={p:.4f}")
    if abs(r) > 0.9:
        print("   [!] GYANÚS: a két oszlop szinte azonos -> rossz CSV-párosítás")
    elif abs(r) > 0.7:
        print("   [?] erős összefüggés, de az irodalomban is korrelálnak")
    else:
        print("   OK: külön konstruktumok")
else:
    print("1) nincs arousal oszlop")

# --- 2. Torzít-e a csendkapuzás? -----------------------------------------
# Ha a szegmenshossz korrelál a valenciával, akkor a kapuzás rendszeresen
# másképp bánik a kellemes és a kellemetlen hangokkal.
print()
if "seg_s" in d.columns:
    r, p = spearmanr(d["seg_s"], d["valence"])
    print(f"2) szegmenshossz vs valencia:  r={r:+.3f}  p={p:.4f}")
    print(f"   kapuzott arány: {d['gated'].mean():.2f}")
    print(f"   szegmenshossz: {d['seg_s'].min():.2f} .. {d['seg_s'].max():.2f} s")
    if p < 0.05:
        print("   [!] a kapuzás rendszeresen torzít")

# --- 3. Szintműtermék-e a loudness korrelációja? -------------------------
# Ha a loudness pusztán a felvételi szintet méri, akkor a hangfájlok
# eredeti kivezérlése magyarázza a korrelációt, nem az észlelt hangosság.
print()
print("3) Metrikák egymás közti korrelációja (Spearman):")
print(d[COLS].corr(method="spearman").round(3))
print("\n   Ha a loudness_N5 alig korrelál a többivel, de erősen")
print("   a valenciával, az szintműtermékre utal.")

# --- 4. Alapstatisztikák -------------------------------------------------
print("\n4) Leíró statisztikák:")
cols = COLS + [c for c in ["valence", "arousal"] if c in d.columns]
print(d[cols].describe().round(2))

# --- 5. Hiányzó / gyanús értékek ----------------------------------------
print("\n5) Adatminőség:")
print(f"   NaN-ok: {d[cols].isna().sum().sum()}")
print(f"   nullák: {(d[COLS] == 0).sum().sum()}")

# --- 6. A párosítás ellenőrzése -----------------------------------------
# A legfontosabb: tényleg a helyes valencia tartozik-e a helyes fájlhoz?
print("\n6) Minta a párosításból (ellenőrizd kézzel a Valence.csv-ben!):")
show = ["filename", "valence"] + (["arousal"] if "arousal" in d.columns else [])
print(d[show].head(5).to_string(index=False))
print("   ...")
print(d[show].tail(5).to_string(index=False))
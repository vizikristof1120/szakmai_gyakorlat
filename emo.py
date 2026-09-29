"""
Emo-Soundscapes kiegészítő modul a MoSQITo-teszthez.

Miért hasznos: ez az adathalmaz EMBERI ÉRTÉKELÉSEKET tartalmaz (valencia =
kellemesség), tehát megadja azt a kritériumváltozót, ami eddig hiányzott.
Nem csak "nyugtató hangokat" ad, hanem mérhető célváltozót is.

Használat a fő szkriptből:
    from emo import emo_dataframe
    df_emo = emo_dataframe(EMO_ROOT, n=60, jobs=8)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed


# ---------------------------------------------------------------- felderítés
def discover(root):
    """Megkeresi a valencia-CSV-t és az audiofájlokat, bármilyen almappában."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Nincs ilyen mappa: {root}")

    # A macOS-es ZIP-ekben ott marad a __MACOSX szemétmappa — ki kell szűrni,
    # különben 0 bájtos ál-fájlokat talál.
    def clean(paths):
        return [p for p in paths
                if "__MACOSX" not in p.parts and not p.name.startswith("._")]

    csvs = clean(root.rglob("*.csv"))
    wavs = clean(root.rglob("*.wav")) + clean(root.rglob("*.ogg")) \
        + clean(root.rglob("*.mp3")) + clean(root.rglob("*.flac"))

    val_csv = [c for c in csvs if "valence" in c.name.lower()]
    aro_csv = [c for c in csvs if "arousal" in c.name.lower()]

    print(f"  CSV-k: {len(csvs)}, hangfájlok: {len(wavs)}")
    if not val_csv:
        print(f"  [!] Nem találtam 'Valence' nevű CSV-t. Találtak: "
              f"{[c.name for c in csvs][:10]}")
    return {
        "valence_csv": val_csv[0] if val_csv else None,
        "arousal_csv": aro_csv[0] if aro_csv else None,
        "audio": {p.name: p for p in wavs},
    }


def load_ratings(csv_path, audio_map):
    """Beolvassa a rating-CSV-t, oszlopneveket automatikusan felismerve."""
    # Sok ilyen CSV fejléc nélküli — mindkettőt megpróbáljuk.
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2 or df.iloc[:, 1].dtype == object:
        df = pd.read_csv(csv_path, header=None)

    # Fájlnév-oszlop: az, amelyikben a legtöbb érték egyezik a hangfájlokkal.
    names = set(audio_map)
    stems = {Path(n).stem for n in names}
    best, best_hit = None, 0
    for c in df.columns:
        vals = df[c].astype(str)
        hit = sum((v in names) or (Path(v).stem in stems) for v in vals)
        if hit > best_hit:
            best, best_hit = c, hit
    if best is None or best_hit == 0:
        raise ValueError(f"Nem találtam fájlnév-oszlopot itt: {csv_path}")

    # Érték-oszlop: az első numerikus, ami nem a fájlnév-oszlop.
    num = [c for c in df.columns
           if c != best and pd.api.types.is_numeric_dtype(df[c])]
    if not num:
        raise ValueError(f"Nem találtam numerikus oszlopot itt: {csv_path}")

    out = pd.DataFrame({
        "fname": df[best].astype(str),
        "value": pd.to_numeric(df[num[0]], errors="coerce"),
    }).dropna()

    # Az Emo-Soundscapes páros összehasonlításból készült: a "rating" fájlokban
    # gyakran RANGSORSZÁM van 1..1213 között, nem -1..+1 közötti érték.
    # Felismerés: egész számok, 1-től indulnak, a maximum >> 1.
    v = out["value"]
    looks_like_rank = (v.min() >= 0.5 and v.max() > 10
                       and np.allclose(v, v.round()))
    if looks_like_rank:
        # Az 1-es rang a legkellemetlenebb, a legnagyobb a legkellemesebb
        # (a fájlnevek ezt megerősítik: nature/quiet alul, mechanical felül).
        # Lineáris leképezés -1..+1 tartományra.
        out["value"] = -1.0 + 2.0 * (v - v.min()) / (v.max() - v.min())
        print(f"  {csv_path.name}: RANGSOR felismerve ({v.min():.0f}..{v.max():.0f}) "
              f"-> átskálázva -1..+1 tartományra")
    else:
        print(f"  {csv_path.name}: {len(out)} sor, "
              f"tartomány {v.min():.2f} .. {v.max():.2f}")
    print(f"     {best_hit} fájlnév-egyezés")
    return out


def resolve(fname, audio_map):
    if fname in audio_map:
        return audio_map[fname]
    stem = Path(fname).stem
    for n, p in audio_map.items():
        if Path(n).stem == stem:
            return p
    return None


# ------------------------------------------------------- rétegzett mintavétel
def stratified_sample(df, n, bins=10, seed=42):
    """A valencia teljes tartományából egyenletesen mintavételez.

    Fontos: véletlen mintavétel a középső értékeket zsúfolná tele, és pont a
    szélsőségek (nagyon kellemes / nagyon kellemetlen) maradnának ki, amiktől
    a korreláció értelmezhető lenne.
    """
    df = df.copy()
    df["bin"] = pd.qcut(df["value"], q=bins, labels=False, duplicates="drop")
    per = max(1, n // df["bin"].nunique())
    out = (df.groupby("bin", group_keys=False)
             .apply(lambda g: g.sample(min(per, len(g)), random_state=seed)))
    return out.head(n).reset_index(drop=True)


# ------------------------------------------------------------------ fő belépő
def emo_dataframe(root, extract_fn, n=60, jobs=1, only_originals=True):
    """Visszaad egy DataFrame-et: jellemzők + valencia (+ arousal, ha van)."""
    print(f"\n-> Emo-Soundscapes felderítése: {root}")
    d = discover(root)
    if d["valence_csv"] is None:
        return pd.DataFrame()

    val = load_ratings(d["valence_csv"], d["audio"])

    # A 1213 klipből 613 MESTERSÉGESEN KEVERT. Az eredeti 600-at érdemes
    # használni, mert a keverés megváltoztatja az akusztikai karaktert.
    if only_originals:
        before = len(val)
        val = val[~val["fname"].str.contains("mix", case=False, na=False)]
        if len(val) < before:
            print(f"  kevert klipek kiszűrve: {before} -> {len(val)}")

    val["path"] = val["fname"].map(lambda f: resolve(f, d["audio"]))
    val = val.dropna(subset=["path"])
    print(f"  párosított fájl: {len(val)}")
    if val.empty:
        return pd.DataFrame()

    sample = stratified_sample(val, n).reset_index(drop=True)
    print(f"-> {len(sample)} klip feldolgozása (rétegzett a valencia mentén)...")

    res = Parallel(n_jobs=jobs, verbose=5)(
        delayed(extract_fn)(str(p), "emo_soundscape")
        for p in sample["path"]
    )
    # FONTOS: index szerint párosítunk, nem pozíció szerint. Ha egy fájl
    # hibára fut és None-t ad, a pozíciós vágás minden értéket elcsúsztatna.
    keep = [i for i, r in enumerate(res) if r is not None]
    if not keep:
        return pd.DataFrame()
    feats = pd.DataFrame([res[i] for i in keep]).reset_index(drop=True)
    feats["valence"] = sample["value"].values[keep]

    if d["arousal_csv"]:
        try:
            aro = load_ratings(d["arousal_csv"], d["audio"])
            m = dict(zip(aro["fname"].map(lambda f: Path(f).stem), aro["value"]))
            feats["arousal"] = [m.get(Path(f).stem, np.nan)
                                for f in feats["filename"]]
        except Exception as e:
            print(f"  [!] arousal kihagyva: {e}")
    return feats


# ------------------------------------------------------------- kiértékelés
def report(feats, cols):
    """Spearman-korreláció metrika vs. valencia — EZ a lényeg."""
    from scipy.stats import spearmanr
    if feats.empty:
        print("  [!] nincs adat")
        return

    print("\n--- METRIKA vs. EMBERI ÉRTÉKELÉS (Spearman) ---")
    for target in ["valence", "arousal"]:
        if target not in feats or feats[target].isna().all():
            continue
        print(f"\n  {target} (n={feats[target].notna().sum()}):")
        for c in cols:
            m = feats[[c, target]].dropna()
            if len(m) < 5:
                continue
            r, p = spearmanr(m[c], m[target])
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            print(f"    {c:16s} r={r:+.3f}  p={p:.4f} {sig}")

    print("\n--- PROFILOK VALENCIA-HARMADONKÉNT ---")
    f = feats.dropna(subset=["valence"]).copy()
    if len(f) < 6 or f["valence"].nunique() < 3:
        print(f"  (kihagyva: túl kevés minta, n={len(f)})")
    else:
        f["csoport"] = pd.qcut(f["valence"], 3, duplicates="drop",
                               labels=["kellemetlen", "semleges", "kellemes"])
        print(f.groupby("csoport", observed=True)[cols].mean().round(3))
    print("\n  (a valencia -1 = kellemetlen, +1 = kellemes)")


# ============================================================================
# ÖNÁLLÓ FUTTATÁS
#
#   python emo.py                      # 60 klip, 1 szál
#   python emo.py --n 60 --jobs 8      # ajánlott
#   python emo.py --root "C:/Users/vizik/Downloads/Emo-Soundscapes"
#
# Az eredmény emo_features.csv-be mentődik.
# ============================================================================
if __name__ == "__main__":
    import argparse
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            message="invalid value encountered in divide")

    from TESTING import extract_features, FEATURE_COLS

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\vizik\Downloads\Emo-Soundscapes")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    feats = emo_dataframe(args.root, extract_features, n=args.n, jobs=args.jobs)
    if not feats.empty:
        feats.to_csv("emo_features.csv", index=False)
        print("-> Mentve: emo_features.csv")
        report(feats, FEATURE_COLS)
"""
Éjszakai futás — felügyelet nélkül, megszakítható, folytatható.

MIT CSINÁL

  1. szakasz (EMO):  mind a 600 eredeti Emo-Soundscapes klip, LUFS-kiegyenlítve,
     tonalitással kiegészítve.
     -> Ez adja a FÜGGETLEN MEGERŐSÍTŐ MINTÁT az U-alak hipotézishez.
        A korábbi 60 klip a "felfedező" minta, a maradék ~540 a "megerősítő".
        Ezzel a post-hoc sejtésből igazolt (vagy cáfolt) eredmény lesz.

  2. szakasz (ESC):  a teljes ESC-50 készlet, LUFS-kiegyenlítve, tonalitással.
     -> Teszteli, hogy a tonalitás megjavítja-e a leggyengébb osztályokat
        (duda, harang, motor), és hogy a normalizálás változtat-e a 36%-on.

BIZTONSÁG FELÜGYELET NÉLKÜLI FUTÁSHOZ
  - minden fájl soronként mentődik -> áramszünet/Ctrl+C esetén nem vész el semmi
  - újraindításkor onnan folytatja, ahol abbahagyta (a kész fájlokat kihagyja)
  - egyetlen hibás fájl sem állítja meg a futást, a hiba naplóba kerül
  - naplófájl időbélyegekkel: ejszakai_naplo.txt

FUTTATÁS

    python ejszakai.py --jobs 8                 # mindkét szakasz
    python ejszakai.py --jobs 8 --stage emo     # csak az Emo-Soundscapes
    python ejszakai.py --analyze                # reggel: kiértékelés (másodpercek)

  Becsült idő ~10 s/fájl mellett, 8 szálon: EMO ~100 perc, ESC ~140 perc.
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# A sajat modulokbol -- igy biztosan ugyanazok a jellemzodefiniciok
from TESTING import load_mono_48k, active_segment, SHARP_SKIP, TARGET_CLASSES
from normalizal import normalize_lufs, to_pa
from mosqito.sq_metrics import (loudness_zwtv, sharpness_din_tv, roughness_dw,
                                tnr_ecma_st, pr_ecma_st)

EMO_CSV = Path("ejszakai_emo.csv")
ESC_CSV = Path("ejszakai_esc.csv")
LOG = Path("ejszakai_naplo.txt")
CHUNK = 20          # ennyi fajlonkent ment -> ennyit lehet legfeljebb elveszteni

FEATS = ["loudness_N5", "sharpness_S5", "roughness_R5", "crest_db",
         "tnr_max_db", "pr_max_db", "ton_db_count"]


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ===========================================================================
# JELLEMZOK (a megszokottak + tonalitas)
# ===========================================================================
def extract(path, label=""):
    """Egy fajl -> jellemzok. Barmilyen hiba eseten None-t ad, nem dob."""
    try:
        sig, fs = load_mono_48k(path)
        sig, lufs = normalize_lufs(sig, fs)
        if not np.isfinite(lufs):
            return None
        sig = to_pa(sig, normalized=True)
        seg, info = active_segment(sig, fs)

        N, _, _, _ = loudness_zwtv(seg, fs)
        S, _ = sharpness_din_tv(seg, fs, skip=SHARP_SKIP)
        R, _, _, _ = roughness_dw(seg, fs)

        # Tonalitas: ECMA-418-1 tone-to-noise ratio es prominence ratio.
        # Ha nincs kiugro tonalis komponens, ures tombot ad -> 0.0
        # KALIBRACIOFUGGETLEN: aranyalapu, ezert a szint nem befolyasolja.
        def tone(fn):
            try:
                _, v, _, _ = fn(seg, fs, prominence=True)
                v = np.atleast_1d(np.asarray(v, dtype=float))
                v = v[np.isfinite(v)]
                return (float(np.max(v)), int(v.size)) if v.size else (0.0, 0)
            except Exception:
                return (0.0, 0)

        tnr, n_tnr = tone(tnr_ecma_st)
        pr, _ = tone(pr_ecma_st)

        rms = float(np.sqrt(np.mean(seg ** 2)))
        crest = 20 * np.log10(np.max(np.abs(seg)) / rms) if rms > 0 else 0.0

        return {
            "loudness_N5": float(np.nanpercentile(N, 95)),
            "sharpness_S5": float(np.nanpercentile(S, 95)),
            "roughness_R5": float(np.nanpercentile(R, 95)),
            "crest_db": float(crest),
            "tnr_max_db": tnr,
            "pr_max_db": pr,
            "ton_db_count": n_tnr,
            "eredeti_lufs": float(lufs),
            "seg_s": info["seg_s"],
            "label": label,
            "filename": Path(path).name,
        }
    except Exception as e:
        log(f"   [HIBA] {Path(path).name}: {type(e).__name__}: {e}")
        return None


def run_chunked(tasks, out_csv, jobs):
    """Darabokban dolgozik es minden darab utan ment -> folytathato."""
    kesz = set()
    if out_csv.exists():
        try:
            kesz = set(pd.read_csv(out_csv)["filename"])
            log(f"   folytatas: {len(kesz)} fajl mar kesz, kihagyom")
        except Exception:
            log("   [!] a meglevo CSV olvashatatlan, ujrakezdem")

    todo = [(p, l) for p, l in tasks if Path(p).name not in kesz]
    if not todo:
        log("   nincs tennivalo, minden fajl kesz")
        return
    log(f"   {len(todo)} fajl feldolgozasra var")

    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        res = Parallel(n_jobs=jobs)(delayed(extract)(p, l) for p, l in batch)
        rows = [r for r in res if r is not None]
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_csv, mode="a", header=not out_csv.exists(),
                      index=False)
        done = min(i + CHUNK, len(todo))
        el = time.time() - t0
        hatra = el / max(done, 1) * (len(todo) - done)
        log(f"   {done}/{len(todo)}  ({100*done/len(todo):.0f}%)  "
            f"eltelt {el/60:.0f} perc, hatra ~{hatra/60:.0f} perc")


# ===========================================================================
# 1. SZAKASZ — EMO-SOUNDSCAPES (mind a 600 eredeti klip)
# ===========================================================================
def stage_emo(root, jobs):
    log("=== 1. SZAKASZ: Emo-Soundscapes, mind a 600 eredeti klip ===")
    from emo import discover, load_ratings, resolve
    from normalizal import rank_direction

    d = discover(root)
    if d["valence_csv"] is None:
        log("[!] Nincs Valence CSV, a szakasz kimarad.")
        return

    val = load_ratings(d["valence_csv"], d["audio"])
    val = val[~val["fname"].str.contains("mix", case=False, na=False)]
    sign = rank_direction(val)
    val = val.copy()
    val["value"] = sign * val["value"]
    val["path"] = val["fname"].map(lambda f: resolve(f, d["audio"]))
    val = val.dropna(subset=["path"])
    val.to_csv("ejszakai_emo_valencia.csv", index=False)
    log(f"   {len(val)} klip, valencia elmentve")

    run_chunked([(str(p), "emo") for p in val["path"]], EMO_CSV, jobs)


# ===========================================================================
# 2. SZAKASZ — ESC-50 (teljes keszlet, tonalitassal)
# ===========================================================================
def stage_esc(jobs):
    log("=== 2. SZAKASZ: ESC-50 teljes keszlet, tonalitassal ===")
    data = Path("test_data/office_sounds")
    log(f"   adatmappa: {data.resolve()}")
    if not data.exists():
        log("[!] NINCS MEG az ESC-50 adatmappa, a szakasz kimarad.")
        log("    Futtasd a szkriptet abbol a mappabol, ahol a TESTING.py-t "
            "is futtattad (ott van a test_data mappa).")
        return
    hianyzo = [c for c in TARGET_CLASSES if not (data / c).exists()]
    if hianyzo:
        log(f"[!] hianyzo osztalymappak: {hianyzo}")
    tasks = []
    for cat in TARGET_CLASSES:
        tasks += [(str(f), cat) for f in sorted((data / cat).glob("*.wav"))]
    log(f"   {len(tasks)} fajl, {len(TARGET_CLASSES)} osztaly")
    run_chunked(tasks, ESC_CSV, jobs)


# ===========================================================================
# KIERTEKELES — reggel, masodpercek alatt
# ===========================================================================
def p_star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def analyze_emo():
    if not EMO_CSV.exists():
        print(f"\nEMO: nincs eredmenyfajl ({EMO_CSV.resolve()})")
        return
    print("\n" + "=" * 70)
    print("EMO-SOUNDSCAPES — AZ U-ALAK MEGEROSITESE FUGGETLEN MINTAN")
    print("=" * 70)
    df = pd.read_csv(EMO_CSV)
    val = pd.read_csv("ejszakai_emo_valencia.csv")
    val["filename"] = val["path"].map(lambda p: Path(p).name)
    df = df.merge(val[["filename", "value"]], on="filename", how="inner")
    df = df.rename(columns={"value": "valence"}).dropna(subset=["valence"])

    # A korabbi 60 klip = felfedezo minta; a tobbi = fuggetlen megerosito minta
    disc = set()
    for f in ["emo_normalizalt.csv", "emo_features.csv"]:
        if Path(f).exists():
            try:
                disc |= set(pd.read_csv(f)["filename"])
            except Exception:
                pass
    df["minta"] = np.where(df["filename"].isin(disc), "felfedezo", "megerosito")
    n_m = (df["minta"] == "megerosito").sum()
    print(f"  osszesen {len(df)} klip  "
          f"({len(df)-n_m} felfedezo, {n_m} FUGGETLEN megerosito)\n")

    for nev, sub in [("TELJES MINTA", df),
                     ("FUGGETLEN MEGEROSITO MINTA", df[df.minta == "megerosito"])]:
        if len(sub) < 20:
            continue
        print(f"--- {nev} (n={len(sub)}) ---")
        print(f"{'metrika':<16}{'linearis':>20}{'U-alak (|valencia|)':>26}")
        for c in FEATS:
            if c not in sub or sub[c].nunique() < 3:
                continue
            r1, p1 = spearmanr(sub[c], sub["valence"])
            r2, p2 = spearmanr(sub[c], sub["valence"].abs())
            print(f"{c:<16}  r={r1:+.3f} p={p1:.4f}{p_star(p1):<4}"
                  f"    r={r2:+.3f} p={p2:.4f}{p_star(p2)}")
        print()

    print("--- PROFILOK VALENCIA-HARMADONKENT (teljes minta) ---")
    f = df.copy()
    f["csoport"] = pd.qcut(f["valence"], 3, duplicates="drop",
                           labels=["kellemetlen", "semleges", "kellemes"])
    print(f.groupby("csoport", observed=True)[FEATS].mean().round(3))
    print("\n  Ha a 'semleges' oszlop a legalacsonyabb -> U-alak megerositve.")


def analyze_esc():
    if not ESC_CSV.exists():
        print("\n" + "=" * 70)
        print(f"ESC-50: nincs eredmenyfajl ({ESC_CSV.resolve()})")
        print("  A 2. szakasz nem futott le. Az okot az ejszakai_naplo.txt")
        print("  '2. SZAKASZ' sora utan talalod. Inditas kulon:")
        print("      python ejszakai.py --jobs 8 --stage esc")
        print("=" * 70)
        return
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import classification_report

    print("\n" + "=" * 70)
    print("ESC-50 — SEGIT-E A TONALITAS?")
    print("=" * 70)
    df = pd.read_csv(ESC_CSV).dropna(subset=FEATS)
    y = df["label"]
    present = [c for c in TARGET_CLASSES if c in set(y)]
    n_split = int(min(5, y.value_counts().min()))
    if n_split < 2:
        print("  tul keves minta")
        return

    base = ["loudness_N5", "sharpness_S5", "roughness_R5", "crest_db"]
    cv = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    print(f"  {len(df)} minta, {len(present)} osztaly, {n_split}-fold CV\n")

    eredmeny = {}
    for nev, cols in [("tonalitas NELKUL (4 jellemzo)", base),
                      ("tonalitassal (7 jellemzo)", FEATS)]:
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        pred = cross_val_predict(clf, df[cols], y, cv=cv)
        acc = (pred == y).mean()
        eredmeny[nev] = (acc, pred, cols)
        print(f"  {nev:34s} pontossag = {acc:.3f}")

    a1 = eredmeny["tonalitas NELKUL (4 jellemzo)"][0]
    a2 = eredmeny["tonalitassal (7 jellemzo)"][0]
    print(f"\n  valtozas: {a2-a1:+.3f}  "
          f"({'a tonalitas segit' if a2 > a1 else 'nem segit'})")

    acc, pred, cols = eredmeny["tonalitassal (7 jellemzo)"]
    print("\n--- OSZTALYONKENT (tonalitassal) ---")
    print(classification_report(y, pred, labels=present,
                                target_names=present, zero_division=0))

    clf = DecisionTreeClassifier(max_depth=6, random_state=42).fit(df[FEATS], y)
    print("--- JELLEMZOK FONTOSSAGA ---")
    for n_, imp in sorted(zip(FEATS, clf.feature_importances_),
                          key=lambda x: -x[1]):
        print(f"  {n_:16s} {imp:.3f}  {'#'*int(imp*45)}")


# ===========================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=r"C:\Users\vizik\Downloads\Emo-Soundscapes")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--stage", choices=["emo", "esc", "mind"], default="mind")
    ap.add_argument("--analyze", action="store_true",
                    help="csak kiertekeles a meglevo CSV-kbol")
    a = ap.parse_args()

    if a.analyze:
        analyze_emo()
        analyze_esc()
        sys.exit(0)

    log(f"### INDULAS  stage={a.stage}  jobs={a.jobs}")
    t0 = time.time()
    try:
        if a.stage in ("emo", "mind"):
            stage_emo(a.root, a.jobs)
        if a.stage in ("esc", "mind"):
            stage_esc(a.jobs)
    except KeyboardInterrupt:
        log("### MEGSZAKITVA — az eddigi eredmenyek mentve, "
            "ujrainditaskor folytatja")
        sys.exit(0)
    except Exception:
        log("### VARATLAN HIBA:\n" + traceback.format_exc())
        log("### az eddigi eredmenyek mentve maradtak")
        sys.exit(1)

    log(f"### KESZ  ({(time.time()-t0)/60:.0f} perc)")
    analyze_emo()
    analyze_esc()
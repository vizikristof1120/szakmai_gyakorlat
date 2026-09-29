import sys
import shutil
import argparse
import subprocess
from math import gcd
from pathlib import Path

# ==========================================
# 1. CSOMAGOK
# ==========================================
# FONTOS: a pip-név és az import-név eltérhet (scikit-learn -> sklearn),
# ezért párban tároljuk, különben minden futásnál újratelepítené.
REQUIRED_PACKAGES = [
    ("mosqito", "mosqito"),
    ("soundfile", "soundfile"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("joblib", "joblib"),
]


def install_dependencies():
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            print(f"[{pip_name}] nincs telepítve. Telepítés...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


install_dependencies()

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from mosqito.sq_metrics import loudness_zwtv, sharpness_din_tv, roughness_dw

# ==========================================
# 2. ADATHALMAZ
# ==========================================
DATA_DIR = Path("test_data/office_sounds")
ESC50_DIR = Path("ESC-50-temp")
ESC50_REPO = "https://github.com/karolpiczak/ESC-50.git"
CSV_OUT = Path("mosqito_features.csv")

# FIGYELEM: az ESC-50 aláhúzással tárolja a többszavas kategóriákat,
# és az ajtókopogás neve door_wood_knock (nem door_knock).
TARGET_CLASSES = [
    # iroda / belső tér
    "keyboard_typing", "mouse_click", "door_wood_knock", "clock_tick",
    "vacuum_cleaner", "toilet_flush",
    # emberi
    "coughing", "sneezing", "breathing", "footsteps",
    # háttér / természet
    "rain", "wind", "thunderstorm", "crickets", "chirping_birds",
    # külső / gépi
    "helicopter", "car_horn", "engine", "train", "church_bells", "airplane",
    # nyugtató kontroll
    "sea_waves", "water_drops", "pouring_water", "crackling_fire", "frog", "insects",
    # zavaró kontroll (a sziréna és az ébresztő tonális!)
    "siren", "clock_alarm", "crying_baby", "glass_breaking", "chainsaw",
]

def prepare_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print(f"-> Tesztadatok már megvannak: {DATA_DIR}")
        return

    print("-> ESC-50 letöltése (~600 MB, eltarthat pár percig)...")
    if not ESC50_DIR.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", ESC50_REPO, str(ESC50_DIR)], check=True
        )

    df = pd.read_csv(ESC50_DIR / "meta" / "esc50.csv")
    audio_dir = ESC50_DIR / "audio"

    # Csendes hiba elleni védelem: elgépelt kategórianév eddig
    # egyszerűen nulla fájlt eredményezett, hibaüzenet nélkül.
    missing = set(TARGET_CLASSES) - set(df["category"].unique())
    if missing:
        raise ValueError(
            f"Nincs ilyen ESC-50 kategória: {sorted(missing)}\n"
            f"Elérhető nevek: {sorted(df['category'].unique())}"
        )

    filtered = df[df["category"].isin(TARGET_CLASSES)]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"-> {len(filtered)} fájl másolása {len(TARGET_CLASSES)} kategóriába...")
    for _, row in filtered.iterrows():
        cat_folder = DATA_DIR / row["category"]
        cat_folder.mkdir(exist_ok=True)
        shutil.copy(audio_dir / row["filename"], cat_folder / row["filename"])

    shutil.rmtree(ESC50_DIR, ignore_errors=True)
    print("-> Adatelőkészítés kész.")


# ==========================================
# 3. JELLEMZŐK
# ==========================================
TARGET_FS = 48000    # a MoSQITo ezt várja; enélkül minden híváskor újramintavételez
MIN_DUR = 0.6        # s — ennél rövidebb szegmensen a metrikák instabilak
PAD = 0.05           # s — párnázás az aktív szakasz körül
GATE_DB = 10.0       # dB a háttér fölött → aktívnak számít
FRAME_MS = 20.0      # ms — a szintbecslés keretmérete
SHARP_SKIP = 0.2     # s — a sharpness_din_tv indulási tranziensének levágása

FEATURE_COLS = ["loudness_N5", "sharpness_S5", "roughness_R5", "crest_db"]


def load_mono_48k(path):
    """Egyszeri beolvasás + egyszeri reszamplálás 48 kHz-re."""
    sig, fs = sf.read(path, always_2d=False)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)          # monóvá keverés, nem csak az első csatorna
    sig = np.asarray(sig, dtype=np.float64)
    if fs != TARGET_FS:
        g = gcd(int(fs), TARGET_FS)
        sig = resample_poly(sig, TARGET_FS // g, int(fs) // g)
    return sig, TARGET_FS


def active_segment(sig, fs, thr_db=GATE_DB, frame_ms=FRAME_MS):
    """Összefüggő aktív szakaszt vág ki (első..utolsó aktív keret).

    Szándékosan NEM fűz össze különálló darabokat: az mesterséges ugrásokat
    okozna, amit a roughness hamis érdességnek látna.
    Ha nincs kiemelkedő esemény (eső, szél), a teljes jelet adja vissza.
    """
    w = max(1, int(frame_ms / 1000 * fs))
    n_fr = max(1, (len(sig) - w) // w + 1)
    lev = np.array([
        20 * np.log10(np.sqrt(np.mean(sig[i * w:i * w + w] ** 2)) + 1e-12)
        for i in range(n_fr)
    ])
    bg = np.percentile(lev, 10)
    act = np.where(lev > bg + thr_db)[0]

    info = {"bg_db": float(bg), "orig_s": len(sig) / fs}
    if act.size == 0:
        info.update(seg_s=len(sig) / fs, gated=False)
        return sig, info

    pad = int(PAD * fs)
    a = max(0, act[0] * w - pad)
    b = min(len(sig), (act[-1] + 1) * w + pad)

    need = int(MIN_DUR * fs)
    if b - a < need:
        extra = need - (b - a)
        a = max(0, a - extra // 2)
        b = min(len(sig), b + (need - (b - a)))
        a = max(0, b - need)

    seg = sig[a:b]
    info.update(seg_s=len(seg) / fs, gated=True)
    return seg, info


def extract_features(file_path, label):
    """Időben változó (_tv) metrikák, 95. percentilissel aggregálva.

    Az _st (stacionárius) változatok tranziens hangokra hibás értéket adnak.
    A 95. percentilis az 5%-os túllépési érték (N5/S5/R5) — ez a
    pszichoakusztika bevett statisztikája, nem az átlag.
    """
    try:
        sig, fs = load_mono_48k(file_path)
        seg, info = active_segment(sig, fs)

        N, _, _, _ = loudness_zwtv(seg, fs)
        S, _ = sharpness_din_tv(seg, fs, skip=SHARP_SKIP)
        R, _, _, _ = roughness_dw(seg, fs)

        rms = float(np.sqrt(np.mean(seg ** 2)))
        crest_db = 20 * np.log10(np.max(np.abs(seg)) / rms) if rms > 0 else 0.0

        return {
            "loudness_N5": float(np.percentile(N, 95)),
            "sharpness_S5": float(np.percentile(S, 95)),
            "roughness_R5": float(np.percentile(R, 95)),
            "crest_db": float(crest_db),
            "seg_s": info["seg_s"],
            "gated": info["gated"],
            "label": label,
            "filename": Path(file_path).name,
        }
    except Exception as e:
        print(f"   [HIBA] {Path(file_path).name}: {type(e).__name__}: {e}")
        return None


# ==========================================
# 4. KIÉRTÉKELÉS
# ==========================================
def compute_features(limit=None, jobs=1):
    tasks = []
    for cat in TARGET_CLASSES:
        files = sorted((DATA_DIR / cat).glob("*.wav"))
        if limit:
            files = files[:limit]
        tasks += [(str(f), cat) for f in files]

    print(f"\n-> {len(tasks)} fájl feldolgozása {jobs} szálon...")
    print("   (a roughness_dw a leglassabb, legyen türelem)")
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(extract_features)(f, c) for f, c in tasks
    )
    results = [r for r in results if r is not None]
    return pd.DataFrame(results)


def evaluate(df):
    print(f"\nFeldolgozott minta: {len(df)}")

    # Diagnosztika: hol talált eseményt a csendkapuzás?
    print("\n--- CSENDKAPUZÁS DIAGNOSZTIKA ---")
    diag = df.groupby("label").agg(
        kapuzott_arany=("gated", "mean"),
        atlag_hossz_s=("seg_s", "mean"),
    )
    print(diag.round(2))
    print("  (kapuzott_arany ~0 folytonos hangnál normális: eső, szél, zizegés)")

    print("\n--- PSZICHOAKUSZTIKAI PROFILOK (N5 / S5 / R5) ---")
    summary = df.groupby("label")[FEATURE_COLS].mean().sort_values("sharpness_S5")
    print(summary.round(3))

    # A sharpness csak 1.75 acum fölött járul hozzá a Zwicker-féle
    # psychoacoustic annoyance-hez — érdemes látni, mely osztályok lépik át.
    above = summary[summary["sharpness_S5"] > 1.75]
    print(f"\n  1.75 acum fölött ({len(above)}/{len(summary)} osztály): "
          f"{', '.join(above.index)}")

    X = df[FEATURE_COLS]
    y = df["label"]
    present = [c for c in TARGET_CLASSES if c in set(y)]

    # A foldok száma nem lehet nagyobb a legkisebb osztály méreténél
    # (kis --limit értéknél ez különben ValueError-t dobna).
    min_count = y.value_counts().min()
    n_splits = int(min(5, min_count))
    if n_splits < 2:
        print("\n[!] Osztályonként legalább 2 minta kell a keresztvalidációhoz. "
              "Növeld a --limit értékét.")
        return

    clf = DecisionTreeClassifier(max_depth=6, random_state=42)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    print(f"\n--- OSZTÁLYOZÁSI PONTOSSÁG ({n_splits}-Fold CV) ---")
    # labels= NÉLKÜL a sklearn ABC-sorrendbe rendez, de a target_names
    # a mi sorrendünkben lenne -> a címkék némán felcserélődnének!
    print(classification_report(
        y, y_pred, labels=present, target_names=present, zero_division=0
    ))

    print("--- TÉVESZTÉSI MÁTRIX ---")
    cm = confusion_matrix(y, y_pred, labels=present)
    print(pd.DataFrame(cm, index=[f"V:{c}" for c in present],
                       columns=[f"J:{c}" for c in present]))

    clf.fit(X, y)
    print("\n--- JELLEMZŐK FONTOSSÁGA ---")
    for name, imp in sorted(zip(FEATURE_COLS, clf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"  {name:16s} {imp:.3f}  {'#' * int(imp * 40)}")
    print("  (ha a crest_db dominál, a pszichoakusztikai metrikák keveset adtak hozzá)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="max. fájl osztályonként (gyors próbához, pl. 5)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="párhuzamos folyamatok száma (pl. 4)")
    ap.add_argument("--from-csv", action="store_true",
                    help="a mentett CSV-ből számol, nem újra")
    args = ap.parse_args()

    if args.from_csv:
        if not CSV_OUT.exists():
            sys.exit(f"Nincs meg: {CSV_OUT} — futtasd előbb --from-csv nélkül.")
        df = pd.read_csv(CSV_OUT)
        print(f"-> Betöltve: {CSV_OUT}")
    else:
        prepare_dataset()
        df = compute_features(limit=args.limit, jobs=args.jobs)
        df.to_csv(CSV_OUT, index=False)
        print(f"-> Mentve: {CSV_OUT}")

    evaluate(df)


if __name__ == "__main__":
    main()
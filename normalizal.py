"""
Szintkiegyenlítés (LUFS-normalizálás) és a korrelációk újraszámolása.

MIÉRT: a mostani eredményed legnagyobb gyengesége, hogy a loudness_N5 erős
korrelációja (r = -0.785) lehet puszta MŰTERMÉK. Az Emo-Soundscapes klipek
Freesoundról jönnek, különböző emberektől, különböző kivezérléssel. Így a
loudness nem az észlelt hangosságot méri, hanem azt, KI MILYEN SZINTEN VETTE FEL.

MIT CSINÁL: minden klipet azonos észlelt szintre hoz (ITU-R BS.1770 / LUFS),
majd ugyanazokra a klipekre újraszámolja a metrikákat, és EGYMÁS MELLETT
mutatja a korrelációt normalizálás ELŐTT és UTÁN.

ÉRTELMEZÉS:
  - ha egy korreláció a normalizálás után is megmarad -> VALÓDI, spektrális
    vagy időbeli tartalom hordozza
  - ha összeomlik -> a felvételi szint csinálta, nem a hang karaktere

Futtatás:
    python normalizal.py --n 60 --jobs 8
    python normalizal.py --root "C:\\Users\\vizik\\Downloads\\Emo-Soundscapes"

Szükséges:  pip install pyloudnorm
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln
from scipy.stats import spearmanr
from joblib import Parallel, delayed

# A MoSQITo sharpness-e 0/0 osztast jelez a csendes kereteken, de utana
# lenullazza oket -- ismert, artalmatlan. Csak EZT az egyet nemitjuk.
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")

from TESTING import load_mono_48k, active_segment, SHARP_SKIP
from mosqito.sq_metrics import loudness_zwtv, sharpness_din_tv, roughness_dw

P_REF = 20e-6
TARGET_LUFS = -23.0    # EBU R128 ajanlas
TARGET_SPL = 65.0      # dB SPL, tipikus irodai beszedszint
COLS = ["loudness_N5", "sharpness_S5", "roughness_R5", "crest_db"]


# ===========================================================================
# 1. SZINTKIEGYENLITES
# ===========================================================================
def normalize_lufs(sig, fs, target_lufs=TARGET_LUFS):
    """ITU-R BS.1770 szerinti szintkiegyenlites.

    A LUFS K-sulyozott, es beepitett kapuzasa van (a csendes reszeket
    kihagyja a mereesbol), ezert alkalmasabb, mint az egyszeru RMS.
    Visszaad: (normalizalt jel, eredeti LUFS)
    """
    # A merőnek legalabb block_size (0.4 s) hosszu jel kell
    if len(sig) < int(0.5 * fs):
        return sig, np.nan
    meter = pyln.Meter(fs)
    lufs = meter.integrated_loudness(sig)
    if not np.isfinite(lufs):          # nema vagy majdnem nema klip
        return sig, np.nan
    gain = 10 ** ((target_lufs - lufs) / 20)
    return sig * gain, lufs


def to_pa(sig, normalized):
    """Digitalis jel -> pascal.

    Normalizalt jelnel a lekepezes minden klipnel AZONOS, mert a LUFS
    mar kiegyenlitette oket: a target_lufs szint felel meg target_spl-nek.
    Normalizalatlan jelnel nincs kalibracio -- ez epp a problema, amit
    vizsgalunk, ezert a nyers erteket adjuk tovabb (a MoSQITo igy ugy
    ertelmezi, mintha a csucs 1 Pa lenne, azaz ~91 dB SPL).
    """
    if not normalized:
        return sig
    return sig * P_REF * 10 ** ((TARGET_SPL - TARGET_LUFS) / 20)


# ===========================================================================
# 2. JELLEMZOK
# ===========================================================================
def features(path, normalized):
    """Ugyanaz a jellemzokeszlet, normalizalassal vagy anelkul."""
    try:
        sig, fs = load_mono_48k(path)
        lufs = np.nan
        if normalized:
            sig, lufs = normalize_lufs(sig, fs)
            if not np.isfinite(lufs):
                return None
        sig = to_pa(sig, normalized)

        seg, info = active_segment(sig, fs)
        N, _, _, _ = loudness_zwtv(seg, fs)
        S, _ = sharpness_din_tv(seg, fs, skip=SHARP_SKIP)
        R, _, _, _ = roughness_dw(seg, fs)

        rms = float(np.sqrt(np.mean(seg ** 2)))
        crest = 20 * np.log10(np.max(np.abs(seg)) / rms) if rms > 0 else 0.0

        return {
            "loudness_N5": float(np.nanpercentile(N, 95)),
            "sharpness_S5": float(np.nanpercentile(S, 95)),
            "roughness_R5": float(np.nanpercentile(R, 95)),
            "crest_db": float(crest),
            "eredeti_lufs": float(lufs),
            "seg_s": info["seg_s"],
            "filename": Path(path).name,
        }
    except Exception as e:
        print(f"   [HIBA] {Path(path).name}: {type(e).__name__}: {e}")
        return None


# ===========================================================================
# 3. A RANGSOR IRANYANAK AUTOMATIKUS MEGALLAPITASA
# ===========================================================================
def rank_direction(df):
    """Az Emo-Soundscapes fajlnevekben benne van a Schafer-kategoria
    (nature, quiet, human, society, mechanical, indicator).

    Ebbol EMPIRIKUSAN eldontheto, melyik iranyba no a kellemesseg --
    nem kell feltetelezni. Visszaad: +1 (nagy rang = kellemes) vagy -1.
    """
    kellemes = df["fname"].str.contains("nature|quiet", case=False, na=False)
    kellemetlen = df["fname"].str.contains("mechanical|indicator", case=False,
                                           na=False)
    if kellemes.sum() < 5 or kellemetlen.sum() < 5:
        print("  [!] a fajlnevekbol nem allapithato meg az irany, "
              "feltetelezes: nagy rang = kellemes")
        return 1

    m_kell = df.loc[kellemes, "value"].mean()
    m_nem = df.loc[kellemetlen, "value"].mean()
    print(f"  iranyellenorzes ({kellemes.sum()} nature/quiet vs "
          f"{kellemetlen.sum()} mechanical/indicator):")
    print(f"     atlagertek {m_kell:+.3f}  vs  {m_nem:+.3f}")
    if m_kell < m_nem:
        print("  -> ALACSONY rang = kellemes, tehat a valencia = -rang")
        return -1
    print("  -> MAGAS rang = kellemes, tehat a valencia = +rang")
    return 1


# ===========================================================================
# 4. FO FUTAS
# ===========================================================================
def main(root, n, jobs):
    from emo import discover, load_ratings, stratified_sample, resolve

    print(f"\n-> Emo-Soundscapes: {root}")
    d = discover(root)
    if d["valence_csv"] is None:
        print("[!] Nincs Valence CSV.")
        return

    val = load_ratings(d["valence_csv"], d["audio"])
    val = val[~val["fname"].str.contains("mix", case=False, na=False)]

    # Az iranyt megmerjuk, nem feltetelezzuk
    sign = rank_direction(val)
    val = val.copy()
    val["value"] = sign * val["value"]

    val["path"] = val["fname"].map(lambda f: resolve(f, d["audio"]))
    val = val.dropna(subset=["path"])
    sample = stratified_sample(val, n).reset_index(drop=True)
    paths = [str(p) for p in sample["path"]]
    print(f"-> {len(paths)} klip, mindegyik KETSZER (elotte + utana)...")

    out = {}
    for label, norm in [("ELOTTE (nyers)", False),
                        ("UTANA (LUFS-kiegyenlitett)", True)]:
        print(f"\n-> {label}")
        res = Parallel(n_jobs=jobs, verbose=5)(
            delayed(features)(p, norm) for p in paths)
        keep = [i for i, r in enumerate(res) if r is not None]
        df = pd.DataFrame([res[i] for i in keep]).reset_index(drop=True)
        df["valence"] = sample["value"].values[keep]
        out[label] = df

    # ---------------- eredmenyek ----------------
    before, after = out["ELOTTE (nyers)"], out["UTANA (LUFS-kiegyenlitett)"]
    after.to_csv("emo_normalizalt.csv", index=False)
    before.to_csv("emo_nyers.csv", index=False)
    print("\n-> Mentve: emo_nyers.csv, emo_normalizalt.csv")

    print("\n" + "=" * 64)
    print("EREDETI SZINTEK SZORASA (ez a problema forrasa)")
    print("=" * 64)
    lu = after["eredeti_lufs"].dropna()
    print(f"  a klipek eredeti LUFS-szintje: {lu.min():.1f} .. {lu.max():.1f}")
    print(f"  terjedelem: {lu.max()-lu.min():.1f} dB, szoras: {lu.std():.1f} dB")
    r, p = spearmanr(after["eredeti_lufs"], after["valence"],
                     nan_policy="omit")
    print(f"\n  eredeti felveteli szint vs. valencia:  r={r:+.3f}  p={p:.4f}")
    if p < 0.05:
        print("  [!] A FELVETELI SZINT MAGA korrelal a valenciaval.")
        print("      Ez kozvetlen bizonyitek a mutermekre.")

    print("\n" + "=" * 64)
    print("KORRELACIO A VALENCIAVAL - ELOTTE vs. UTANA")
    print("=" * 64)
    print(f"{'metrika':<16}{'elotte':>10}{'utana':>10}{'valtozas':>11}   ertekeles")
    print("-" * 64)
    for c in COLS:
        r1, p1 = spearmanr(before[c], before["valence"])
        r2, p2 = spearmanr(after[c], after["valence"])
        s1 = "***" if p1 < .001 else "**" if p1 < .01 else "*" if p1 < .05 else ""
        s2 = "***" if p2 < .001 else "**" if p2 < .01 else "*" if p2 < .05 else ""
        # FONTOS: az elojelvaltas nem "megmaradas". Ha a korrelacio atfordul,
        # az azt jelenti, hogy a nyers ertek a valodi hatas ELLENKEZOJET
        # mutatta -- a felveteli szint elfedte es meg is forditotta.
        flip = (r1 * r2) < 0
        if p1 < .05 and p2 < .05 and flip:
            verd = "ELOJELET VALTOTT -> a szint forditotta meg!"
        elif p1 < .05 and p2 < .05:
            gyeng = abs(r2) < abs(r1) * 0.6
            verd = ("MEGMARADT, de gyengult -> reszben szint"
                    if gyeng else "MEGMARADT -> valodi")
        elif p1 < .05 and p2 >= .05:
            verd = "ELTUNT -> szintmutermek volt"
        elif p1 >= .05 and p2 < .05:
            verd = "MEGJELENT -> a szint rejtette"
        else:
            verd = "egyik sem szignifikans"
        print(f"{c:<16}{r1:+7.3f}{s1:<3}{r2:+7.3f}{s2:<3}{r2-r1:+11.3f}   {verd}")
    print("\n  *** p<0.001   ** p<0.01   * p<0.05")

    print("\n" + "=" * 64)
    print("PROFILOK VALENCIA-HARMADONKENT (normalizalas utan)")
    print("=" * 64)
    f = after.dropna(subset=["valence"]).copy()
    if len(f) >= 6:
        f["csoport"] = pd.qcut(f["valence"], 3, duplicates="drop",
                               labels=["kellemetlen", "semleges", "kellemes"])
        print(f.groupby("csoport", observed=True)[COLS].mean().round(3))
        print("\n  Ellenorzes: a loudness_N5 oszlopnak most mar kozel")
        print("  azonosnak kell lennie a harom csoportban -- ha nem az,")
        print("  a maradek elteres a spektrumbol jon, nem a felveteli szintbol.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\vizik\Downloads\Emo-Soundscapes")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    main(a.root, a.n, a.jobs)

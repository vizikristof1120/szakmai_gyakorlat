"""
Validáció — reprodukálja-e az eszköz a MoSQITo referenciametrikákat?

A KÉRDÉS
  Az eddigi metrikáidat a MoSQITo teljes sávszélességű hangból számolta.
  Az eszköz viszont 1/3-oktávsávos, N keret/s adatot fog adni. Ez nem ugyanaz.
  Ez a szkript ugyanazokon a klipeken összeveti a kettőt.

  DÖNTÉSI KÜSZÖB (előre rögzítve, mielőtt megnézzük az eredményt):
     r > 0.90  -> az eszköz reprodukálja, használható
     0.70-0.90 -> gyengült, de rangsorolásra még jó
     r < 0.70  -> NEM reprodukálható; vagy az eszközön kell számolni,
                  vagy ki kell hagyni a modellből

A VÁLTOZATOK
  Több eszközbeállítást is végigpróbál, hogy legyen mire hivatkozni a
  hardveres egyeztetésen:
     fps 50 / 100 / 200        - mennyi keret/s kell?
     csúcstartás nélkül        - számolható-e crest factor?
     bekapcsolt AGC            - mit ront el, ha nem kapcsolható ki?

FUTTATÁS
    python validal.py --jobs 8                 # minden változat
    python validal.py --jobs 8 --limit 100     # gyors próba
    python validal.py --analyze                # kiértékelés a meglévő CSV-kből

  Megszakítható és folytatható, mint az ejszakai.py.
  Szükséges fájlok ugyanabban a mappában:
     eszkoz_szim.py, ejszakai_emo.csv, ejszakai_emo_valencia.csv
"""

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from scipy.stats import spearmanr
from joblib import Parallel, delayed

from eszkoz_szim import Eszkoz, P_REF

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

REF_CSV = Path("ejszakai_emo.csv")            # MoSQITo referenciaértékek
PATH_CSV = Path("ejszakai_emo_valencia.csv")  # a hangfájlok elérési útja
LOG = Path("validal_naplo.txt")
CHUNK = 25

TARGET_LUFS = -23.0   # ugyanaz, mint a normalizal.py-ban
TARGET_SPL = 65.0

# szimulált metrika -> referencia oszlop
PAROK = [("loudness_N5", "loudness_N5"),
         ("sharpness_S5", "sharpness_S5"),
         ("roughness_R5", "roughness_R5"),
         ("crest_db", "crest_db"),
         ("ton_max_db", "pr_max_db")]

# (nev, fps, peak_hold, agc)
VALTOZATOK = [("fps100", 100, True, False),
              ("fps50", 50, True, False),
              ("fps200", 200, True, False),
              ("fps100_nincs_csucs", 100, False, False),
              ("fps100_agc", 100, True, True)]


def log(m):
    line = f"[{datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    try:
        LOG.open("a", encoding="utf-8").write(line + "\n")
    except Exception:
        pass


def egy_fajl(path, fps, ph, agc):
    """Egy klip végigfuttatása a szimulált eszközön."""
    try:
        sig, fs = sf.read(path, always_2d=False)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        sig = np.asarray(sig, float)
        if len(sig) < int(0.5 * fs):
            return None
        # Ugyanaz a szintkiegyenlítés, mint a referenciánál (normalizal.py)
        lufs = pyln.Meter(fs).integrated_loudness(sig)
        if not np.isfinite(lufs):
            return None
        sig = sig * 10 ** ((TARGET_LUFS - lufs) / 20)
        sig = sig * P_REF * 10 ** ((TARGET_SPL - TARGET_LUFS) / 20)

        e = Eszkoz(fps=fps, peak_hold=ph, agc=agc)
        if fs != 48000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(fs), 48000)
            sig = resample_poly(sig, 48000 // g, int(fs) // g)
            fs = 48000
        # JAVITVA: az AGC-t itt kell alkalmazni. Korabban a _frames kozvetlen
        # hivasa megkerulte a wav_to_frames-t, ahol az AGC fut -> az agc=True
        # valtozat valojaban AGC nelkul futott.
        if agc:
            sig = e._agc(sig, fs)
        frames = e._frames(sig, fs)

        m = e.metrics(frames)
        m["filename"] = Path(path).name
        return m
    except Exception as ex:
        log(f"   [HIBA] {Path(path).name}: {type(ex).__name__}: {ex}")
        return None


def futtat(nev, fps, ph, agc, paths, jobs):
    out = Path(f"validal_{nev}.csv")
    kesz = set()
    if out.exists():
        try:
            kesz = set(pd.read_csv(out)["filename"])
        except Exception:
            pass
    todo = [p for p in paths if Path(p).name not in kesz]
    log(f"=== {nev}: fps={fps} csucstartas={ph} agc={agc} — "
        f"{len(todo)} fajl ({len(kesz)} kesz) ===")
    if not todo:
        return
    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        res = Parallel(n_jobs=jobs)(
            delayed(egy_fajl)(p, fps, ph, agc) for p in batch)
        rows = [r for r in res if r is not None]
        if rows:
            pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(),
                                      index=False)
        done = min(i + CHUNK, len(todo))
        el = time.time() - t0
        log(f"   {done}/{len(todo)} ({100*done/len(todo):.0f}%) — "
            f"hatra ~{el/max(done,1)*(len(todo)-done)/60:.0f} perc")


def verdikt(r):
    """A dontes a korrelacio ERossegen mulik, nem az elojelen: egy kozelito
    metrika lehet forditott iranyu es meg mindig hasznalhato (pl. skalazva)."""
    if not np.isfinite(r):
        return "nem szamolhato (nincs szoras vagy tul keves adat)"
    jel = " [FORDITOTT IRANY]" if r < 0 else ""
    a = abs(r)
    if a > 0.90:
        return "REPRODUKALHATO" + jel
    if a > 0.70:
        return "gyengult, rangsorolasra jo" + jel
    return "NEM REPRODUKALHATO" + jel


def kiertekel():
    if not REF_CSV.exists():
        print(f"Nincs meg: {REF_CSV.resolve()}")
        return
    ref = pd.read_csv(REF_CSV)
    print("\n" + "=" * 74)
    print("REPRODUKALJA-E AZ ESZKOZ A REFERENCIAT?")
    print("=" * 74)
    print("  Spearman-korrelacio a MoSQITo referencia es a szimulalt")
    print("  eszkozadatbol szamolt metrika kozott, klipenkent parositva.\n")

    osszes = {}
    for nev, fps, ph, agc in VALTOZATOK:
        f = Path(f"validal_{nev}.csv")
        if not f.exists():
            continue
        sim = pd.read_csv(f)
        d = sim.merge(ref, on="filename", suffixes=("_sim", "_ref"))
        if len(d) < 10:
            continue
        sor = {}
        for s_col, r_col in PAROK:
            a = d[s_col + "_sim"] if s_col + "_sim" in d else d.get(s_col)
            b = d[r_col + "_ref"] if r_col + "_ref" in d else d.get(r_col)
            if a is None or b is None:
                continue
            ok = np.isfinite(a) & np.isfinite(b)
            # legalább 5 érvényes pár kell, és mindkét oldalon legyen szórás
            sor[s_col] = (spearmanr(a[ok], b[ok])[0]
                          if ok.sum() >= 5 and a[ok].nunique() > 1
                          and b[ok].nunique() > 1 else np.nan)
        osszes[nev] = sor
        print(f"--- {nev}  (n={len(d)}) ---")
        for k, v in sor.items():
            csik = "#" * int(max(v, 0) * 30) if np.isfinite(v) else ""
            print(f"    {k:14s} r={v:+.3f}  {csik:<30} {verdikt(v)}")
        print()

    if len(osszes) > 1:
        print("=" * 74)
        print("OSSZEHASONLITAS (Spearman r)")
        print("=" * 74)
        print(pd.DataFrame(osszes).round(3).to_string())
        print("\n  Amit ki kell olvasni:")
        print("   - fps50 vs fps100 vs fps200: melyik keretsebesseg kell?")
        print("   - nincs_csucs: a crest_db eltunik -> csucstartas kell")
        print("   - agc: mennyit ront a modulacio-alapu metrikakon")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()

    if a.analyze:
        kiertekel()
        return

    if not PATH_CSV.exists():
        print(f"Nincs meg: {PATH_CSV.resolve()}")
        return
    v = pd.read_csv(PATH_CSV)
    paths = [str(p) for p in v["path"] if Path(str(p)).exists()]
    if not paths:
        print("Egyik hangfajl sem talalhato a megadott utvonalakon.")
        print(f"Minta ut: {v['path'].iloc[0]}")
        return
    if a.limit:
        paths = paths[:a.limit]
    log(f"### INDULAS — {len(paths)} klip, {len(VALTOZATOK)} valtozat, "
        f"jobs={a.jobs}")

    t0 = time.time()
    try:
        for nev, fps, ph, agc in VALTOZATOK:
            futtat(nev, fps, ph, agc, paths, a.jobs)
    except KeyboardInterrupt:
        log("### MEGSZAKITVA — az eddigi eredmenyek mentve")
        return
    except Exception:
        log("### HIBA:\n" + traceback.format_exc())
        return
    log(f"### KESZ ({(time.time()-t0)/60:.0f} perc)")
    kiertekel()


if __name__ == "__main__":
    main()
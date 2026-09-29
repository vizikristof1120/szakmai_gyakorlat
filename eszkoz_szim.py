"""
Eszközszimulátor — a mérési lánc hardver nélkül.

MIÉRT EZ A FELÉPÍTÉS
  A program bemenete nem hangfájl, hanem az a formátum, amit az eszköz majd ad:
  1/3-oktávsávos keretek N keret/s sebességgel. Ezt most WAV-ból állítjuk elő.
  Amikor megjön a hardver, csak a bemeneti forrást kell kicserélni — a
  metrikaszámítás, az eseménydetektálás és a validáció változatlan marad.

  Így a teljes lánc megírható és validálható MOST, mérések előtt.

A LÁNC
  WAV ──> [A] FFT-keretek ──> [B] 1/3-oktáv aggregálás ──> [C] metrikák
                                        │
                                        └─ ennyi megy át Bluetooth-on

  A referencia a MoSQITo teljes sávszélességű hangból számolt értéke.
  A kérdés: a [C] mennyire közelíti a referenciát?

HASZNÁLAT
    from eszkoz_szim import Eszkoz
    e = Eszkoz(fps=100, peak_hold=True)
    keretek = e.wav_to_frames("hang.wav")
    m = e.metrics(keretek)

MEGÁLLAPÍTÁS: IDŐ- ÉS FREKVENCIAFELBONTÁS ÜTKÖZÉSE
    Ez a szimulátor FFT-ablakokból képez sávokat, ahogy a legtöbb egyszerű
    hardvermegvalósítás. Az ablak 2 keret hosszú, tehát a frekvenciafelbontás
    a keretsebességgel romlik:

        25 keret/s  ->  80 ms ablak  ->  12 Hz felbontás
       100 keret/s  ->  20 ms ablak  ->  50 Hz felbontás
       200 keret/s  ->  10 ms ablak  -> 100 Hz felbontás

    Egy 1/3-oktávsáv szélessége a középfrekvencia ~23%-a, tehát a 160 Hz-es
    sáv csak ~37 Hz széles. 100 keret/s mellett az ennél keskenyebb (alsó)
    sávokat NEM lehet feloldani — a mérés ott több dB-t téved.

    Vagyis az időfelbontás (impulzusok, moduláció) és a frekvenciafelbontás
    (alsó sávok) egymás ellen dolgozik EGYETLEN FFT-ablakméret mellett.

    MEGOLDÁS a hardveren: ne egyetlen FFT-ablak legyen, hanem
    (a) valódi 1/3-oktáv SZŰRŐBANK (IEC 61260), ahol minden sávnak saját
        időállandója van — így nincs ütközés, vagy
    (b) többfelbontású FFT: az alsó sávok hosszú ablakból, a felsők rövidből.

    Ezt érdemes a hardveres kollégával egyeztetni, MIELŐTT az FFT-lánc
    véglegesedik.
"""

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="invalid value encountered in divide")

# A MoSQITo belső, SZABVÁNYOS útja: 1/3-oktáv szintek -> hangosság.
# (A loudness_zwst_freq NEM jó ide: egyenletes frekvenciaosztást feltételez,
#  a mi sávjaink viszont logaritmikusak.)
from mosqito.sq_metrics.loudness.loudness_zwst._main_loudness import _main_loudness
from mosqito.sq_metrics.loudness.loudness_zwst._calc_slopes import _calc_slopes
from mosqito.sq_metrics import sharpness_din_from_loudness
from mosqito.utils import amp2db

P_REF = 20e-6
TARGET_FS = 48000

# ISO 532-1 B módszere pontosan 28 sávot használ: 25 Hz - 12.5 kHz.
# Ugyanez a MoSQITo noct_spectrum(fmin=24, fmax=12600) kimenete, ezért a
# szimulátor kimenete közvetlenül összevethető a referenciával.
CENTERS = np.array([25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
                    315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500,
                    3150, 4000, 5000, 6300, 8000, 10000, 12500], float)


class Eszkoz:
    """A tervezett mérőeszköz szimulációja.

    fps        : keret/s. A korábbi mérés szerint 100 az ajánlott
                 (a szinthiba 2 dB alatt marad minden felfutási időnél).
    peak_hold  : ad-e az eszköz keretenkénti csúcsértéket. Ha nem, a
                 crest factor nem számolható — ezt a validáció megmutatja.
    agc        : az automatikus erősítésszabályzás szimulációja. Ha a
                 hardveren nem kapcsolható ki, ez mutatja meg, mit rontana el.
    """

    def __init__(self, fps=100, peak_hold=True, agc=False, centers=CENTERS,
                 metric_fps=20, field_type="free"):
        self.fps = fps
        self.metric_fps = metric_fps
        self.field_type = field_type
        self.peak_hold = peak_hold
        self.agc = agc
        self.centers = np.asarray(centers, float)
        # sávhatárok: +/- 1/6 oktáv a középfrekvencia körül
        self.lo = self.centers * 2 ** (-1 / 6)
        self.hi = self.centers * 2 ** (+1 / 6)

    # ---------------------------------------------------------------- [A]+[B]
    def wav_to_frames(self, path, spl_ref=None):
        """WAV -> (sávkeretek, csúcsok). A sávkeretek Pa-ban, RMS.

        spl_ref : ha megadod, erre a dB SPL-re skálázza a jelet. Enélkül a
                  fájl nyers amplitúdóját használja (kalibrálatlan eset).
        """
        sig, fs = sf.read(path, always_2d=False)
        if sig.ndim > 1:
            sig = sig.mean(axis=1)
        sig = np.asarray(sig, float)
        if fs != TARGET_FS:
            g = gcd(int(fs), TARGET_FS)
            sig = resample_poly(sig, TARGET_FS // g, int(fs) // g)
        fs = TARGET_FS

        if spl_ref is not None:
            rms = np.sqrt(np.mean(sig ** 2))
            if rms > 0:
                sig = sig / rms * P_REF * 10 ** (spl_ref / 20)

        if self.agc:
            sig = self._agc(sig, fs)

        return self._frames(sig, fs)

    def _agc(self, sig, fs, tau=0.5, target_db=-20):
        """Egyszerű AGC: kisimítja a szintingadozást.

        Ez rontja el a moduláció-alapú metrikákat (érdesség, ingadozáserősség).
        """
        w = max(1, int(0.02 * fs))
        env = np.sqrt(np.convolve(sig ** 2, np.ones(w) / w, "same")) + 1e-12
        a = np.exp(-1 / (tau * fs / w))
        sm = np.copy(env)
        for i in range(1, len(sm)):
            sm[i] = a * sm[i - 1] + (1 - a) * env[i]
        gain = (10 ** (target_db / 20)) / sm
        return sig * np.clip(gain, 0.1, 10)

    def _frames(self, sig, fs):
        """FFT-keretek -> 1/3-oktáv sávenergiák (ezt adja át az eszköz)."""
        hop = max(1, int(round(fs / self.fps)))
        win = hop * 2
        n = max(0, (len(sig) - win) // hop + 1)
        if n < 2:
            return np.zeros((0, len(self.centers))), np.zeros(0)

        w = np.hanning(win)
        # a Hann-ablak teljesítménykorrekciója, hogy az RMS helyes maradjon
        corr = np.sqrt(np.mean(w ** 2))
        f = np.fft.rfftfreq(win, 1 / fs)
        idx = [np.where((f >= a) & (f < b))[0] for a, b in zip(self.lo, self.hi)]

        bands = np.zeros((n, len(self.centers)))
        peaks = np.zeros(n)
        for i in range(n):
            seg = sig[i * hop:i * hop + win]
            peaks[i] = np.max(np.abs(seg)) if self.peak_hold else np.nan
            X = np.fft.rfft(seg * w)
            # egyoldalas teljesítménysűrűség -> sávonkénti RMS
            P = (np.abs(X) ** 2) * 2 / (win ** 2) / (corr ** 2)
            P[0] /= 2
            for b, ix in enumerate(idx):
                if ix.size:
                    bands[i, b] = np.sqrt(P[ix].sum())
        return bands, peaks

    # -------------------------------------------------------------------- [C]
    def metrics(self, frames, aktiv_kuszob_db=10.0):
        """Metrikák KIZÁRÓLAG a sávadatból — ahogy a valóságban is lenne."""
        bands, peaks = frames
        if bands.shape[0] < 2:
            return {k: np.nan for k in
                    ["loudness_N5", "sharpness_S5", "roughness_R5",
                     "crest_db", "ton_max_db", "aktiv_s"]}

        # aktív szakasz: keretszint a háttér (10. percentilis) fölött
        lev = 20 * np.log10(np.sqrt((bands ** 2).sum(axis=1)) / P_REF + 1e-12)
        bg = np.percentile(lev, 10)
        act = np.where(lev > bg + aktiv_kuszob_db)[0]
        if act.size < 2:
            act = np.arange(bands.shape[0])
        a, b = act[0], act[-1] + 1
        B = bands[a:b]

        # --- hangosság és élesség: SZABVÁNYOS út (ISO 532-1 / DIN 45692
        #     eleve 1/3-oktáv szintekből dolgozik) ---
        # A hangosság stacionárius számítás, ezért ritkább rácson is pontos:
        # minden step-edik keretet vesszük (metric_fps). Ez tizedeli a futásidőt.
        step = max(1, int(round(self.fps / self.metric_fps)))
        sub = B[::step]
        N, S = [], []
        if sub.size and sub.sum() > 0:
            # ISO 532-1 B: sáv-RMS -> dB -> fő hangosság -> meredekségek.
            # Oszloponként (nfreq, nseg) alakban, ahogy a szabvány kéri.
            spec_db = amp2db(sub.T, ref=P_REF)
            Nm = _main_loudness(spec_db, self.field_type)
            n_, n_spec = _calc_slopes(Nm)
            N = np.atleast_1d(n_).ravel().tolist()
            S = np.atleast_1d(
                sharpness_din_from_loudness(n_, n_spec, weighting="din")
            ).ravel().tolist()

        # --- érdesség KÖZELÍTÉS: a sávburkolók modulációja ---
        #     A szabványos érdesség 20-300 Hz-es modulációt mér, ehhez
        #     ~600 keret/s kellene. Ez itt csak az elérhető tartományt fogja.
        R = self._roughness_proxy(B)

        # --- crest factor: csak csúcstartással ---
        rms = np.sqrt(np.mean((bands[a:b] ** 2).sum(axis=1)))
        crest = (20 * np.log10(np.max(peaks[a:b]) / rms)
                 if self.peak_hold and rms > 0 else np.nan)

        # --- tonalitás KÖZELÍTÉS: sávkiemelkedés a szomszédokhoz képest ---
        ton = self._tonality_proxy(B)

        return {"loudness_N5": float(np.percentile(N, 95)) if N else np.nan,
                "sharpness_S5": float(np.percentile(S, 95)) if S else np.nan,
                "roughness_R5": R,
                "crest_db": crest,
                "ton_max_db": ton,
                "aktiv_s": (b - a) / self.fps}

    def _roughness_proxy(self, B):
        """Modulációs energia a sávburkolókban, a roughness súlyozásához
        hasonló, 70 Hz körül csúcsosodó ablakkal (amennyi belefér)."""
        if B.shape[0] < 8:
            return np.nan
        out = []
        freqs = np.fft.rfftfreq(B.shape[0], 1 / self.fps)
        # a Zwicker-fele erdesseg sulyozasa ~70 Hz-en maximalis
        wgt = np.exp(-((np.log2(np.maximum(freqs, 1e-6) / 70.0)) ** 2) / 2)
        wgt[freqs < 5] = 0
        for b in range(B.shape[1]):
            env = B[:, b]
            m = env.mean()
            if m <= 0:
                continue
            sp = np.abs(np.fft.rfft((env - m) * np.hanning(len(env))))
            out.append((sp * wgt).sum() / (m * len(env)))
        return float(np.sum(out)) if out else np.nan

    def _tonality_proxy(self, B):
        """Mennyire emelkedik ki egy sáv a két szomszédjához képest [dB].
        A szabványos TNR keskenysávú spektrumot igényel — ez csak közelítés."""
        if B.shape[1] < 3:
            return np.nan
        L = 20 * np.log10(B / P_REF + 1e-12)
        # A nulla energiájú sávok -240 dB-t adnának, amitől a szomszéd-
        # különbség értelmetlenül nagy lesz. Keretenként 80 dB-es dinamikára
        # vágjuk a szinteket.
        L = np.maximum(L, L.max(axis=1, keepdims=True) - 80.0)
        nb = (L[:, :-2] + L[:, 2:]) / 2.0
        return float(np.percentile((L[:, 1:-1] - nb).max(axis=1), 95))


# ---------------------------------------------------------------------------
def savszelesseg(fps, n_sav=len(CENTERS), bajt=2, peak_hold=True):
    """Mennyi adatot jelent ez percenként?"""
    per_frame = n_sav * bajt + (2 if peak_hold else 0)
    return fps * per_frame * 60 / 1e6   # MB/perc


if __name__ == "__main__":
    for fps in (25, 50, 100, 200):
        print(f"  {fps:4d} keret/s -> {savszelesseg(fps):.2f} MB/perc")

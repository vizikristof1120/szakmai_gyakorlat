"""Mekkora keretsebesseg kell? -- szimulacio hardver nelkul."""
import numpy as np
P_REF=20e-6; FS=48000

def band_envelope(sig, fs, frame_rate):
    """Szimulalja a hardvert: FFT-keretek -> savenergia-burkolo adott keretsebesseggel."""
    hop = int(round(fs/frame_rate)); win = hop*2
    n = (len(sig)-win)//hop
    return np.array([np.sqrt(np.mean(sig[i*hop:i*hop+win]**2)) for i in range(n)]), frame_rate

# ---------- 1. IMPULZUS: megmarad-e a csucs es a felfutas? ----------
def impulse(rise_ms=300, decay_ms=40, bg=45, peak=75):
    n=int(1.5*FS); t=np.arange(n)/FS
    rng=np.random.default_rng(0)
    sig=rng.standard_normal(n); sig/=np.sqrt(np.mean(sig**2)); sig*=P_REF*10**(bg/20)
    t0=int(0.6*FS); m=int((rise_ms+decay_ms*4)/1000*FS)
    tt=np.arange(m)/FS
    env=np.minimum(tt/(rise_ms/1000),1)*np.exp(-np.maximum(tt-rise_ms/1000,0)/(decay_ms/1000))
    burst=rng.standard_normal(m)*env; burst/=np.max(np.abs(burst))
    sig[t0:t0+m]+=burst*P_REF*10**(peak/20)*np.sqrt(2)
    return sig

print("=== 1. IMPULZUSDETEKTALAS (300 ms felfutas, 75 dB csucs, 45 dB hatter) ===")
sig=impulse()
true_peak=20*np.log10(np.max(np.abs(sig))/P_REF)
print(f"  valodi csucs: {true_peak:.1f} dB\n")
print(f"{'keret/s':>8} | {'mert csucs':>11} | {'alabecsles':>11} | {'hatter folott':>13} | detektalhato?")
for fr in [10,25,50,100,200,400,800]:
    env,_=band_envelope(sig,FS,fr)
    db=20*np.log10(env/P_REF+1e-12)
    bg=np.percentile(db,10); pk=np.max(db)
    print(f"{fr:8d} | {pk:8.1f} dB | {true_peak-pk:8.1f} dB | {pk-bg:10.1f} dB | "
          f"{'IGEN' if pk-bg>=10 else 'NEM (<10 dB)'}")

# ---------- 2. MODULACIO: visszanyerheto-e a lüktetés? ----------
print("\n=== 2. MODULACIO VISSZANYERESE (100% AM, 1 kHz vivo) ===")
def am(fmod,spl=60,dur=3.0):
    t=np.arange(int(dur*FS))/FS
    x=np.sin(2*np.pi*1000*t)*(1+np.sin(2*np.pi*fmod*t))
    x/=np.sqrt(np.mean(x**2)); return x*P_REF*10**(spl/20)

print(f"{'keret/s':>8} | " + " | ".join(f"{f:>5.0f}Hz" for f in [2,4,8,16,32,64]))
for fr in [10,25,50,100,200,400]:
    row=[]
    for fmod in [2,4,8,16,32,64]:
        env,_=band_envelope(am(fmod),FS,fr)
        env=env-np.mean(env)
        if len(env)<8: row.append("  --  "); continue
        sp=np.abs(np.fft.rfft(env*np.hanning(len(env))))
        f=np.fft.rfftfreq(len(env),1/fr)
        det=f[np.argmax(sp[1:])+1]
        ok = abs(det-fmod)/fmod < 0.15 and fmod < fr/2
        row.append(f" {'OK':>4} " if ok else f" {'X':>4} ")
    print(f"{fr:8d} | " + "|".join(row))
print("\n  OK = a modulacios frekvencia helyesen visszanyerheto")

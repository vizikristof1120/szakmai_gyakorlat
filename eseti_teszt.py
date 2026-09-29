import pandas as pd, numpy as np
from scipy.stats import spearmanr
d = pd.read_csv("emo_normalizalt.csv")
for c in ["sharpness_S5","roughness_R5"]:
    lin,_ = spearmanr(d[c], d.valence)
    kvad,_ = spearmanr(d[c], d.valence.abs())   # tavolsag a semlegestol
    print(f"{c:15s} lineáris r={lin:+.3f}   |valencia| r={kvad:+.3f}")
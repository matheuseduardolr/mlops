import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 300
lats = np.random.normal(-23.55, 0.15, n)
lngs = np.random.normal(-46.63, 0.15, n)

df = pd.DataFrame({"lat": lats, "lng": lngs})
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/points.csv", index=False)
print(f"Dados gerados: {n} pontos em data/raw/points.csv")

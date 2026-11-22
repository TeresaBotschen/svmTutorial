""" Erstellung der beiden Datasets fuer svm_tutorial.ipynb.

data/linear_data.csv     zwei linear separierbare Punktwolken, Gaussian gesampled
data/nonlinear_data.csv  zwei ringfoermige Punktwolken, nicht linear separierbar

Labels sind als -1 / +1 kodiert.
"""

# (1) Dependencies
from pathlib import Path

import numpy as np
import pandas as pd

# (2) Dateizugang
OUT = Path(__file__).parent / "data"
OUT.mkdir(exist_ok=True)

# (3) Fixieren des random seeds, um jedes Mal dieselben Daten zu samplen
rng = np.random.default_rng(7)

# (4) Linear separierbare Daten. 
#
# n = 60 Datenpunkte
# Klasse 1: um Zentrum -1.6 / -1.4 verteilt
# Klasse 2: um Zentrum 1.6 / 1.5 verteilt
# Ziel: separierte Punktwolken entlang der Diagonale von links oben nach rechts unten (um 0)
#
# Kovarianz: 0.9: wie weit sind die Punkte um ihr Zentrum verteilt
# Kovarianz: 0.35: niedrige/hohe x Werte gehen mit niedrigen/hohen y Werten einher
#                  (negativ: niedrige Werte gehen mit hohen Werten einher)
#                  (0: keine Relation der x/y Werte)
#
n = 60
cov = np.array([[0.9, 0.35], [0.35, 0.9]])
class_1 = rng.multivariate_normal([-1.6, -1.4], cov, n)
class_2 = rng.multivariate_normal([1.6, 1.5], cov, n)
# Konkateniere die beiden Punktwolken aneinander und erstelle die zugehoerigen labels 
features = np.vstack([class_1, class_2])
labels = np.r_[-np.ones(n), np.ones(n)].astype(int)

# (4.1) Erzwinge die Separierbarkeit.
# Berechne fuer jeden Punkt die Entfernung zur Mitte, normalisiere auf den Faktor 1
# (da der Einheitsvektor Laenge sqrt(1^2 + 1^2) hat)
distances = features @ np.array([1.0, 1.0]) / np.sqrt(2)
# Grenzbereich um die Diagonale (links oben bis rechts unten) in der keine Punkte sein duerfen
hard_margin = 0.35
# Bewege alle Punkte, die innerhalb des Grenzbereiches sind, daraus hinaus
features[(labels == -1) & (distances > -hard_margin)] -= ((distances[(labels == -1) & (distances > -hard_margin)] + hard_margin + rng.uniform(0, 0.4, ((labels == -1) & (distances > -hard_margin)).sum()))[:, None] * np.array([1, 1]) / np.sqrt(2))
features[(labels == 1) & (distances < hard_margin)] += ((hard_margin - distances[(labels == 1) & (distances < hard_margin)] + rng.uniform(0, 0.4, ((labels == 1) & (distances < hard_margin)).sum()))[:, None] * np.array([1, 1]) / np.sqrt(2))

# (4.2) Vermische die Punkte und speichere sie ab
idx = rng.permutation(len(labels))
pd.DataFrame({"x1": features[idx, 0].round(4), "x2": features[idx, 1].round(4), "label": labels[idx]}).to_csv(
    OUT / "linear_data.csv", index=False
)

# (5) Ringfoermige Punktwolken, eine innere und eine aeussere (in und out)
#     Jeweils 150 Punkte
#     Eine Wolke ist um den Radius 1.0 verteilt, die andere um den Radius 2.4
#     Wir samplen zum einen den Radius, zum anderen zwischen 0 und 360 Grad
n_in, n_out = 150, 150
r_in = rng.normal(1.0, 0.28, n_in)
r_out = rng.normal(2.4, 0.32, n_out)
t_in = rng.uniform(0, 2 * np.pi, n_in)
t_out = rng.uniform(0, 2 * np.pi, n_out)

# (5.1) Umrechnung von Radius/Grad Zahl zu x/y Koordinate
# (x,y = Radius * cos(Gradzahl) , Radius * sin(Gradzahl))
features = np.vstack([
    np.c_[r_in * np.cos(t_in), r_in * np.sin(t_in)],
    np.c_[r_out * np.cos(t_out), r_out * np.sin(t_out)],
])
# Wie oben Kodierung der Klassen in 1 und -1
labels = np.r_[np.ones(n_in), -np.ones(n_out)].astype(int)

# (5.2) Vermische die Punkte und speichere sie ab
idx = rng.permutation(len(labels))
pd.DataFrame({"x1": features[idx, 0].round(4), "x2": features[idx, 1].round(4), "label": labels[idx]}).to_csv(
    OUT / "nonlinear_data.csv", index=False
)
print("wrote", OUT / "linear_data.csv", "and", OUT / "nonlinear_data.csv")

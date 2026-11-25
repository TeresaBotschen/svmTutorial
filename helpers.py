# Visualisierung
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import numpy as np

# Farben
COLOR_C_1 = "#0002c9"        # Klasse 1: blau
COLOR_C_2 = "#eb6834"        # Klasse 2: orange

BLACK = "#0b0b0b"
GREY = "#52514e"
LIGHT_GREY = "#e4e3df"

# Farbverlauf zur Visualisierung des Abstandes der Klassen
BG_CMAP = LinearSegmentedColormap.from_list("bg", ["#b7d3f6", "#f6f5f2", "#f8c7b2"])


# Standard fuer alle Plots: weisser hintergrund,
# kein top / right frame, Grid Darstellung, keine Legende
plt.rcParams.update({
    "figure.dpi": 100, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GREY, "axes.labelcolor": BLACK, "axes.titlesize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": LIGHT_GREY, "grid.linewidth": 0.8,
    "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})

# Darstellung der Klassen:
# - Klasse 1: Blau, Kreis
# - Klasse 2: Orange, Viereck
#
# ax: matplotlib axes Objekt
# feature_matrix: 2 Werte (Spalte) per Datenpunkt (Reihe)
# label_vector: fuer jeden Datenpunkt in der feature matrix ein Label (-1 oder 1)
# marker_size = Groesse der Datenpunkte
# alpha: Transparenz der visualisierten Datenpunkte
def scatter_classes(ax, feature_matrix, label_vector, marker_size=36, alpha=1.0):
    # Vorbereitung fuer den scatter plot:
    # Selektiere alle Datenpunkte der jeweiligen Klasse, teile sie in x1 und x2 Koordinate auf
    # und transponiere 
    class_1 = feature_matrix[label_vector == -1]
    class_1_x1 = class_1[:,0].T
    class_1_x2 = class_1[:,1].T

    class_2 = feature_matrix[label_vector == 1]
    class_2_x1 = class_2[:,0].T
    class_2_x2 = class_2[:,1].T

    # Scatter Plot fuer beide Klassen
    ax.scatter(class_1_x1, class_1_x2, c=COLOR_C_1, marker="o", s=marker_size, edgecolors="#fcfcfb", linewidths=1, alpha=alpha, label="class −1", zorder=3)
    ax.scatter(class_2_x1, class_2_x2,  c=COLOR_C_2, marker="s", s=marker_size, edgecolors="#fcfcfb", linewidths=1, alpha=alpha, label="class +1", zorder=3)
    
    # Label Bezeichnung
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


# Plotten des Outputs der SVM, der Decision Boundary und der Support Vektoren
#
# ax: matplotlib axes Objekt
# model: sklearn.svm.SVC Objekt (handle zur SVM)
# feature_matrix: 2 Werte (Spalte) per Datenpunkt (Reihe)
# label_vector: fuer jeden Datenpunkt in der feature matrix ein Label (-1 oder 1)
# title: Titel des Plots
# show_sv: Visualisiere die Support Vektoren oder nicht
# res: pixel - Aufloesung des Plots
def plot_svm(ax, model, feature_matrix, labels, title="", show_sv=True, res=300):
    # (1) Extraktion des Wertebereiches der Features um den zu zeichnenden Bereich festzulegen
    x_min, x_max = feature_matrix[:, 0].min(), feature_matrix[:, 0].max()
    y_min, y_max = feature_matrix[:, 1].min(), feature_matrix[:, 1].max()

    # (2) Darstellung der Decision Function der SVM
    # Erstelle ein grid an X und Y Koordinaten innerhalb des Wertebereiches
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, res), np.linspace(y_min, y_max, res))

    # (2.1) Auswertung der Decision Function fuer jeden Punkt (NICHT Datenpunkt) im Grid
    #       Wir erreichen eine Farbgebung des Grids, bei der mit steigender Distanz 
    #       zur decision boundary der Hintergrund staerker Richtung Blau oder Orange gefaerbt ist.
    #       Entfernung von Extremwerten um den Farbverlauf nicht zu stoeren
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # 95th percentile of |f| as a cap, so a few extreme values do not wash out the colour scale
    lim = np.percentile(np.abs(Z), 95)

    # (2.2) Wir verwenden die Color Map um die Konfidenz (Abstand zur Decision Boundary) der 
    # Datenpunkte zur Decision Boundary darzustellen
    ax.contourf(xx, yy, np.clip(Z, -lim, lim), levels=30, cmap=BG_CMAP, alpha=0.9, zorder=0)

    # (2.3) Darstellung der Decision Boundary und der Margins.
    #       Durchgezogene Linie fuer die Decision Boundary, gestrichelte Linie fuer die Margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[GREY, BLACK, GREY],
               linestyles=["--", "-", "--"], linewidths=[1.2, 2, 1.2], zorder=1)


    # (3) Darstellung der Datepunkte auf dem Grid
    scatter_classes(ax, feature_matrix, labels)

    # (4) Darstellung der Support Vektoren - wenn verfuegbar
    if show_sv and hasattr(model, "support_vectors_"):
        sv = model.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=150, facecolors="none", edgecolors=BLACK, linewidths=1.4, zorder=4, label="support vectors")
    
    # (5) Formatierung: Deaktivierung des default Grids, Titel und plot Dimension
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.grid(False)

# Legende fuer die Matplotlib Plots
# include_sv: fuege Support Vektor Legende hinzu
def legend_handles(include_sv=True):
    # Legende fuer die beiden Klassen (-1 und 1), die margins und die decision boundary
    h = [Line2D([], [], marker="o", ls="", color=COLOR_C_1, label="class −1"),
         Line2D([], [], marker="s", ls="", color=COLOR_C_2, label="class +1"),
         Line2D([], [], color=BLACK, lw=2, label="decision boundary  f(x)=0"),
         Line2D([], [], color=GREY, lw=1.2, ls="--", label="margin  f(x)=±1")]
    
    # optional: Legende fuer die Support Vektoren
    if include_sv:
        h.append(Line2D([], [], marker="o", ls="", mfc="none", mec=BLACK, ms=11, label="support vectors"))
    return h
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Chemins — adapte ces 3 lignes à ton environnement ─────────────────────
CSV_PUMA = "results_puma.csv"      # chemin vers ton CSV PUMA
CSV_CYTO = "results_cytoDark.csv"  # chemin vers ton CSV CytoDArk
OUT      = "figures/"              # dossier de sortie (créé automatiquement)
# ──────────────────────────────────────────────────────────────────────────
os.makedirs(OUT, exist_ok=True)

# ── Palette cohérente par famille ──────────────────────────────────────────
COLORS = {
    "Cellpose-SAM": "#2196F3",   # bleu
    "VISTA-2D":     "#4CAF50",   # vert
    "StarDist":     "#FF9800",   # orange
    "CellVit":      "#9C27B0",   # violet
}
BASELINE_ALPHA = 0.45
FT_ALPHA       = 1.0

# ── Chargement ─────────────────────────────────────────────────────────────
puma  = pd.read_csv(CSV_PUMA)
cyto  = pd.read_csv(CSV_CYTO)

# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — PUMA : Baseline vs Fine-tuned — barres groupées PQ / DQ / SQ
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
fig.suptitle("PUMA Dataset — Baseline vs Fine-tuned per Model", fontsize=14, fontweight='bold')

metrics = [("PQ", "Panoptic Quality (PQ) %"),
           ("DQ", "Detection Quality (DQ) %"),
           ("SQ", "Segmentation Quality (SQ) %")]

# Sélection : 1 baseline + 1 FT par famille (exclut CellVit qui n'a pas de FT)
puma_plot = puma[puma["Famille"] != "CellVit"].copy()
families  = ["Cellpose-SAM", "VISTA-2D", "StarDist"]
x         = np.arange(len(families))
w         = 0.35

for ax, (metric, ylabel) in zip(axes, metrics):
    baselines, finetuned = [], []
    for fam in families:
        sub = puma_plot[puma_plot["Famille"] == fam]
        b = sub[sub["Modele"].str.contains("Baseline", case=False)][metric].values
        f = sub[~sub["Modele"].str.contains("Baseline", case=False)][metric].values
        baselines.append(b[0] if len(b) else 0)
        finetuned.append(f[0] if len(f) else 0)

    bars_b = ax.bar(x - w/2, baselines, w, label="Baseline",
                    color=[COLORS[f] for f in families], alpha=BASELINE_ALPHA,
                    edgecolor='white', linewidth=0.8)
    bars_f = ax.bar(x + w/2, finetuned, w, label="Fine-tuned",
                    color=[COLORS[f] for f in families], alpha=FT_ALPHA,
                    edgecolor='white', linewidth=0.8)

    # Valeurs sur les barres
    for bar in list(bars_b) + list(bars_f):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                f"{h:.1f}", ha='center', va='bottom', fontsize=8)

    # Delta annotation
    for i, (b, f) in enumerate(zip(baselines, finetuned)):
        delta = f - b
        ax.annotate(f"Δ+{delta:.1f}", xy=(x[i] + w/2, f + 2),
                    ha='center', fontsize=7.5, color='#333333',
                    fontweight='bold')

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(families, fontsize=9)
    ax.set_ylim(45, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# CellVit ajouté en annotation séparée
axes[0].axhline(y=74.69, color=COLORS["CellVit"], linestyle=':', linewidth=1.5,
                label=f"CellVit baseline: 74.69%")
axes[1].axhline(y=89.50, color=COLORS["CellVit"], linestyle=':', linewidth=1.5)
axes[2].axhline(y=83.27, color=COLORS["CellVit"], linestyle=':', linewidth=1.5)

# Légende unifiée
legend_patches = [
    mpatches.Patch(facecolor=COLORS[f], label=f, alpha=FT_ALPHA) for f in families
] + [
    mpatches.Patch(facecolor=COLORS["CellVit"], label="CellVit (no FT)", alpha=FT_ALPHA),
    mpatches.Patch(facecolor='gray', label="Baseline", alpha=BASELINE_ALPHA),
    mpatches.Patch(facecolor='gray', label="Fine-tuned", alpha=FT_ALPHA),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(OUT + "fig1_puma_baseline_vs_ft.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig1_puma_baseline_vs_ft.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PUMA : tableau récapitulatif F1 / Precision / Recall
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
ax.axis('off')

# Données triées par F1
rows_data = []
for _, row in puma.iterrows():
    rows_data.append([
        row["Famille"], row["Modele"],
        f"{row['SQ']:.2f}", f"{row['DQ']:.2f}", f"{row['PQ']:.2f}",
        f"{row['Precision']:.2f}", f"{row['Recall']:.2f}", f"{row['F1_Score']:.2f}",
        str(int(row['TP'])), str(int(row['FP'])), str(int(row['FN']))
    ])

col_labels = ["Famille", "Modèle", "SQ%", "DQ%", "PQ%",
              "Precision%", "Recall%", "F1%", "TP", "FP", "FN"]

table = ax.table(cellText=rows_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.6)

# Couleurs lignes header
for j in range(len(col_labels)):
    table[0, j].set_facecolor('#2C3E50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Couleur alternée + highlight meilleur PQ
best_pq_idx = max(range(len(rows_data)), key=lambda i: float(rows_data[i][4]))
for i, row in enumerate(rows_data):
    fam = row[0]
    for j in range(len(col_labels)):
        if i == best_pq_idx:
            table[i+1, j].set_facecolor('#E8F5E9')
        elif i % 2 == 0:
            table[i+1, j].set_facecolor('#F8F9FA')
        else:
            table[i+1, j].set_facecolor('#FFFFFF')
        c = COLORS.get(fam, '#888888')
        table[i+1, 0].set_facecolor(c)
        table[i+1, 0].set_text_props(color='white', fontweight='bold')

ax.set_title("PUMA — Résultats complets (IoU ≥ 0.5)", fontsize=12,
             fontweight='bold', pad=20)
plt.savefig(OUT + "fig2_puma_table.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig2_puma_table.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CytoDArk : PQ vs Résolution — courbes par modèle (meilleur FT)
# ════════════════════════════════════════════════════════════════════════════
# On prend le meilleur FT de chaque famille (le plus performant toutes dims)
# et la baseline, sur le zoom 40x (plus challengeant)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("CytoDArk0 — PQ vs Image Dimension (zoom 20x left, 40x right)",
             fontsize=13, fontweight='bold')

DIM_ORDER_20 = [256, 512, 1024]
DIM_ORDER_40 = [256, 512, 1024, 2048]

# Mapping noms FT par famille
BEST_FT = {
    "Cellpose-SAM": "Cellpose_FT_Complet",
    "VISTA-2D":     "Vista_FT_CYTO",
    "StarDist":     "StarDist_FT_Cyto",
}
BASELINE_NAME = {
    "Cellpose-SAM": "Cellpose_Baseline",
    "VISTA-2D":     "Vista_baseline",   # ← attention minuscule dans ton CSV
    "StarDist":     "StarDist_Baseline",
}
for ax, zoom, dims in [(axes[0], "20x", DIM_ORDER_20),
                        (axes[1], "40x", DIM_ORDER_40)]:
    for fam in ["Cellpose-SAM", "VISTA-2D", "StarDist"]:
        color = COLORS[fam]
        sub   = cyto[(cyto["Famille"] == fam) & (cyto["Zoom"] == zoom)]

        # Baseline
        bl_name = BASELINE_NAME[fam]
        bl = sub[sub["Modele"] == bl_name].copy()
        bl["Dimension"] = bl["Dimension"].astype(int)
        bl = bl.sort_values("Dimension")
        if not bl.empty:
            ax.plot(bl["Dimension"], bl["PQ"],
                    color=color, alpha=BASELINE_ALPHA,
                    linestyle='--', marker='o', markersize=5,
                    label=f"{fam} Baseline")

        # Fine-tuned
        ft_name = BEST_FT[fam]
        ft = sub[sub["Modele"] == ft_name].copy()
        ft["Dimension"] = ft["Dimension"].astype(int)
        ft = ft.sort_values("Dimension")
        if not ft.empty:
            ax.plot(ft["Dimension"], ft["PQ"],
                    color=color, alpha=FT_ALPHA,
                    linestyle='-', marker='s', markersize=6,
                    label=f"{fam} FT", linewidth=2)

    # CellVit — tracé comme une vraie courbe sur les deux panels
    cv_sub = cyto[(cyto["Famille"] == "CellVit") & (cyto["Zoom"] == zoom)].copy()
    if not cv_sub.empty:
        cv_sub["Dimension"] = cv_sub["Dimension"].astype(int)
        cv_sub = cv_sub.sort_values("Dimension")
        # Filtre sur les dims disponibles pour ce zoom
        cv_sub = cv_sub[cv_sub["Dimension"].isin(dims)]
        if not cv_sub.empty:
            ax.plot(cv_sub["Dimension"], cv_sub["PQ"],
                    color=COLORS["CellVit"], alpha=FT_ALPHA,
                    linestyle=':', marker='D', markersize=5,
                    label="CellVit Baseline", linewidth=1.5)

    ax.set_xlabel("Dimension (px)", fontsize=10)
    ax.set_ylabel("PQ (%)", fontsize=10)
    ax.set_title(f"Zoom {zoom}", fontsize=11)
    ax.set_xticks(dims)
    ax.set_ylim(40, 85)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Légende commune
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(OUT + "fig3_cytodark_pq_vs_dim.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig3_cytodark_pq_vs_dim.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — CytoDArk : Gain FT (ΔPQ) par modèle et zoom
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 6))

families_cyto = ["Cellpose-SAM", "VISTA-2D", "StarDist"]
zoom_combos   = [("20x", 256), ("20x", 512), ("20x", 1024),
                 ("40x", 256), ("40x", 512), ("40x", 1024), ("40x", 2048)]
x_labels      = [f"{z}\n{d}px" for z, d in zoom_combos]
x_pos         = np.arange(len(zoom_combos))
w             = 0.25

for i, fam in enumerate(families_cyto):
    deltas = []
    for zoom, dim in zoom_combos:
        sub = cyto[(cyto["Famille"] == fam) & (cyto["Zoom"] == zoom) &
                   (cyto["Dimension"].astype(str) == str(dim))]
        bl_pq = sub[sub["Modele"] == BASELINE_NAME[fam]]["PQ"].values
        ft_pq = sub[sub["Modele"] == BEST_FT[fam]]["PQ"].values
        delta = (ft_pq[0] - bl_pq[0]) if (len(bl_pq) and len(ft_pq)) else 0
        deltas.append(delta)

    offset = (i - 1) * w
    bars = ax.bar(x_pos + offset, deltas, w,
                  color=COLORS[fam], alpha=0.85,
                  label=fam, edgecolor='white')
    for bar, d in zip(bars, deltas):
        if d > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"+{d:.1f}", ha='center', va='bottom', fontsize=7.5)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=8.5)
ax.set_ylabel("ΔPQ (Fine-tuned − Baseline) %", fontsize=10)
ax.set_title("CytoDArk0 — Gain du Fine-tuning par Modèle, Zoom et Résolution",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, frameon=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "fig4_cytodark_delta_pq.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig4_cytodark_delta_pq.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Analyse TP/FP/FN : profil d'erreurs par modèle sur PUMA
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("PUMA — Profil d'erreurs par modèle (TP / FP / FN)",
             fontsize=13, fontweight='bold')

for ax, subset_label, title in [
    (axes[0], "Baseline", "Modèles Baseline"),
    (axes[1], "FT",       "Modèles Fine-tunés")
]:
    rows = []
    labels_plot = []
    for fam in ["Cellpose-SAM", "VISTA-2D", "StarDist"]:
        sub = puma[puma["Famille"] == fam]
        if subset_label == "Baseline":
            row = sub[sub["Modele"].str.contains("Baseline", case=False)]
        else:
            row = sub[~sub["Modele"].str.contains("Baseline", case=False)]
        if not row.empty:
            rows.append(row.iloc[0])
            labels_plot.append(fam)

    # CellVit baseline sur les deux panels
    cv_row = puma[puma["Famille"] == "CellVit"]
    if not cv_row.empty and subset_label == "Baseline":
        rows.append(cv_row.iloc[0])
        labels_plot.append("CellVit")

    x = np.arange(len(rows))
    w = 0.25
    tps = [r["TP"] for r in rows]
    fps = [r["FP"] for r in rows]
    fns = [r["FN"] for r in rows]

    b1 = ax.bar(x - w,   tps, w, label="TP", color="#4CAF50", alpha=0.85)
    b2 = ax.bar(x,       fps, w, label="FP", color="#F44336", alpha=0.85)
    b3 = ax.bar(x + w,   fns, w, label="FN", color="#FF9800", alpha=0.85)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 30,
                    str(int(h)), ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=10)
    ax.set_ylabel("Nombre de cellules", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT + "fig5_puma_error_profile.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig5_puma_error_profile.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Radar chart : comparaison globale des modèles FT sur PUMA
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.patches import FancyArrowPatch

categories = ['SQ', 'DQ', 'PQ', 'Precision', 'Recall', 'F1_Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(60, 100)
ax.set_yticks([65, 75, 85, 95])
ax.set_yticklabels(['65', '75', '85', '95'], fontsize=8)
ax.grid(color='grey', alpha=0.3)

# Modèles FT + CellVit
plot_models = [
    ("Cellpose-SAM", "Cellpose_FT_PUMA"),
    ("VISTA-2D",     "Vista_FT_PUMA"),
    ("StarDist",     "StarDist_FT_PUMA"),
    ("CellVit",      "CellVit_Baseline"),
]

for fam, model_name in plot_models:
    row = puma[puma["Modele"] == model_name]
    if row.empty:
        continue
    row = row.iloc[0]
    values = [row[c] for c in categories]
    values += values[:1]
    ax.plot(angles, values, linewidth=2, color=COLORS[fam], label=model_name)
    ax.fill(angles, values, alpha=0.08, color=COLORS[fam])

ax.set_title("PUMA — Profil global des modèles fine-tunés\n(et CellVit baseline)",
             fontsize=12, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9, frameon=False)
plt.tight_layout()
plt.savefig(OUT + "fig6_puma_radar.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig6_puma_radar.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — CytoDArk best config résumé — heatmap PQ par modèle × config
# ════════════════════════════════════════════════════════════════════════════
# Meilleur PQ FT par famille × (zoom, dim)
import matplotlib.colors as mcolors

pivot_data = {}
configs = []
for zoom in ["20x", "40x"]:
    dims = [256, 512, 1024] if zoom == "20x" else [256, 512, 1024, 2048]
    for dim in dims:
        cfg = f"{zoom}/{dim}"
        configs.append(cfg)
        for fam in families_cyto:
            sub = cyto[(cyto["Famille"] == fam) &
                       (cyto["Zoom"] == zoom) &
                       (cyto["Dimension"].astype(str) == str(dim)) &
                       (cyto["Modele"] == BEST_FT[fam])]
            val = sub["PQ"].values[0] if not sub.empty else np.nan
            pivot_data.setdefault(fam, []).append(val)

matrix = np.array([pivot_data[f] for f in families_cyto])

fig, ax = plt.subplots(figsize=(13, 4))
im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=65, vmax=80)

ax.set_xticks(range(len(configs)))
ax.set_xticklabels(configs, rotation=35, ha='right', fontsize=9)
ax.set_yticks(range(len(families_cyto)))
ax.set_yticklabels(families_cyto, fontsize=10)

for i in range(len(families_cyto)):
    for j in range(len(configs)):
        val = matrix[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                    fontsize=8.5, fontweight='bold',
                    color='white' if val < 70 else 'black')

plt.colorbar(im, ax=ax, label="PQ (%)", shrink=0.8)
ax.set_title("CytoDArk0 — Heatmap PQ Fine-tunés (zoom × dimension)",
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT + "fig7_cytodark_heatmap_pq.png", dpi=200, bbox_inches='tight')
plt.close()
print("✓ fig7_cytodark_heatmap_pq.png")

print("\nTous les graphes générés avec succès !")
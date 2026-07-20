"""
generate_figures_puma.py
────────────────────────
Génère les figures comparatives à partir des masques .npy
produits par les scripts d'inférence.

À lancer APRÈS les 3 scripts d'inférence :
    infer_cellpose_puma.py    (pasteur_env)
    infer_vista_puma.py       (vista)
    infer_stardist_puma.py    (stardist_env)

Usage (n'importe quel env avec numpy/matplotlib/opencv, depuis ~/Pima/) :
    python3 generate_figures_puma.py

Sorties dans inference_outputs/ :
    comparison_all_models.png       — 6 panels : image + GT + 3 FT + overlay
    comparison_contours.png         — 4 panels contours côte à côte (pour rapport)
    comparison_vista_baseline_ft.png — 6 panels baseline vs FT avec erreurs
    comparison_vista_report.png     — 4 panels version rapport
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.io import imread

# ── Chemins ──────────────────────────────────────────────────────────────────
IMAGE_PATH  = "puma_data_converted/val/images/training_set_metastatic_roi_025.tif"
GT_PATH     = "puma_data_converted/val/masks/training_set_metastatic_roi_025_nuclei_mask.tiff"
GT_VIS_PATH = "puma_masks_visual/training_set_metastatic_roi_025_nuclei_vis.png"
NPY_DIR     = "inference_outputs"
OUT_DIR     = "inference_outputs2"

# ── Couleurs des contours (RGB) ───────────────────────────────────────────────
COLORS = {
    "GT":          (255, 255, 255),
    "CellposeSAM": (255, 165,   0),   # orange
    "VISTA":       ( 80, 230,   0),   # vert vif
    "StarDist":    (  0, 160, 255),   # bleu ciel
}


# ════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ════════════════════════════════════════════════════════════════════════════

def safe_u8(img):
    """
    Convertit en uint8 RGB proprement.
    - Gère les images 16-bit (max > 255)
    - Retire le canal alpha si présent (RGBA -> RGB)
    - Normalise par le percentile 99 pour éviter que quelques pixels
      saturés écrasent le contraste de toute l'image
    """
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[:, :, :3]   # RGBA -> RGB
    img = img.astype(np.float32)
    # Normalisation percentile : plus robuste que /max() sur images 16-bit
    p_low  = np.percentile(img, 1)
    p_high = np.percentile(img, 99)
    if p_high > p_low:
        img = (img - p_low) / (p_high - p_low)
    else:
        img = img / (img.max() + 1e-8)
    img = np.clip(img, 0, 1) * 255
    return img.astype(np.uint8)


def mask_to_colored(mask):
    np.random.seed(42)
    n      = int(mask.max()) + 1
    colors = np.random.randint(60, 230, (n, 3), dtype=np.uint8)
    colors[0] = [15, 15, 15]
    return colors[mask.astype(int)]


def draw_contours(base_img, mask, color_rgb, thickness=1):
    """Dessine les contours de toutes les instances sur l'image."""
    # cv2 = BGR
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    out = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        binary  = (mask == cell_id).astype(np.uint8)
        ctrs, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, ctrs, -1, color_bgr, thickness)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def draw_error_contours(base_img, pred_mask, gt_mask, iou_threshold=0.5):
    """
    Colorie les cellules par catégorie d'erreur :
      Vert   = TP (bien détecté)
      Rouge  = FP (fausse alarme)
      Orange = FN (cellule ratée)
    """
    out_bgr  = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    pred_ids = [i for i in np.unique(pred_mask) if i != 0]
    gt_ids   = [i for i in np.unique(gt_mask)   if i != 0]
    matched_pred, matched_gt = set(), set()

    for pid in pred_ids:
        pb = pred_mask == pid
        best_iou, best_gid = 0, None
        for gid in gt_ids:
            if gid in matched_gt:
                continue
            gb  = gt_mask == gid
            iou = np.logical_and(pb, gb).sum() / (np.logical_or(pb, gb).sum() + 1e-8)
            if iou > best_iou:
                best_iou, best_gid = iou, gid
        if best_iou >= iou_threshold and best_gid is not None:
            matched_pred.add(pid)
            matched_gt.add(best_gid)

    # TP vert / FP rouge
    for pid in pred_ids:
        binary  = (pred_mask == pid).astype(np.uint8)
        ctrs, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        color = (0, 200, 80) if pid in matched_pred else (0, 30, 220)
        cv2.drawContours(out_bgr, ctrs, -1, color, 2)

    # FN orange
    for gid in gt_ids:
        if gid in matched_gt:
            continue
        binary  = (gt_mask == gid).astype(np.uint8)
        ctrs, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out_bgr, ctrs, -1, (0, 140, 255), 2)

    tp = len(matched_pred)
    fp = len(pred_ids) - tp
    fn = len(gt_ids)   - len(matched_gt)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), tp, fp, fn


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ════════════════════════════════════════════════════════════════════════════

print("Chargement image et GT...")
img = imread(IMAGE_PATH)
if img.ndim == 3 and img.shape[-1] == 4:
    img = img[:, :, :3]
img_u8 = safe_u8(img)

gt_raw = cv2.imread(GT_PATH, cv2.IMREAD_UNCHANGED)
if gt_raw is not None:
    if gt_raw.ndim > 2:
        gt_raw = gt_raw[:, :, 0]
    gt_mask = gt_raw.astype(np.int32)
else:
    print("  [!] GT non trouvé")
    gt_mask = np.zeros(img.shape[:2], dtype=np.int32)

gt_vis     = imread(GT_VIS_PATH) if os.path.exists(GT_VIS_PATH) else None
gt_display = gt_vis if gt_vis is not None else mask_to_colored(gt_mask)
print(f"  GT cells : {gt_mask.max()}")

print("\nChargement des masques .npy...")
masks = {}
for key, filename in [
    ("cellpose",        "masks_cellpose_ft.npy"),
    ("vista_ft",        "masks_vista_ft.npy"),
    ("vista_baseline",  "masks_vista_baseline.npy"),
    ("stardist",        "masks_stardist_ft.npy"),
]:
    path = os.path.join(NPY_DIR, filename)
    if os.path.exists(path):
        masks[key] = np.load(path).astype(np.int32)
        print(f"  ✓ {filename} — {masks[key].max()} cells")
    else:
        print(f"  [!] {filename} non trouvé — zéros utilisés")
        masks[key] = np.zeros(img.shape[:2], dtype=np.int32)

err_legend = [
    mpatches.Patch(color='#00C850', label='TP — correct'),
    mpatches.Patch(color='#DC1E1E', label='FP — false alarm'),
    mpatches.Patch(color='#FF8C00', label='FN — missed cell'),
]


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — TOUS LES MODÈLES FT (6 panels)
# ════════════════════════════════════════════════════════════════════════════

print("\nFigure 1 — tous les modèles FT (6 panels)...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Qualitative Comparison — PUMA (training_set_metastatic_roi_025)\n"
    "Fine-tuned models vs Ground Truth",
    fontsize=13, fontweight='bold'
)

axes[0, 0].imshow(img_u8)
axes[0, 0].set_title("① Original H&E Image", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_display)
axes[0, 1].set_title(f"② Ground Truth\n({gt_mask.max()} cells)", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_to_colored(masks["cellpose"]))
axes[0, 2].set_title(f"③ Cellpose-SAM FT\n({masks['cellpose'].max()} cells)", fontsize=11, fontweight='bold')
axes[0, 2].axis('off')

axes[1, 0].imshow(mask_to_colored(masks["vista_ft"]))
axes[1, 0].set_title(f"④ VISTA-2D FT\n({masks['vista_ft'].max()} cells)", fontsize=11, fontweight='bold')
axes[1, 0].axis('off')

axes[1, 1].imshow(mask_to_colored(masks["stardist"]))
axes[1, 1].set_title(f"⑤ StarDist FT\n({masks['stardist'].max()} cells)", fontsize=11, fontweight='bold')
axes[1, 1].axis('off')

# Panel ⑥ — contours superposés
overlay = img_u8.copy()
overlay = draw_contours(overlay, gt_mask,             COLORS["GT"],          thickness=3)
overlay = draw_contours(overlay, masks["cellpose"],   COLORS["CellposeSAM"], thickness=2)
overlay = draw_contours(overlay, masks["vista_ft"],   COLORS["VISTA"],       thickness=2)
overlay = draw_contours(overlay, masks["stardist"],   COLORS["StarDist"],    thickness=2)

axes[1, 2].imshow(overlay)
axes[1, 2].set_title("⑥ Contour Overlay\n(GT + all FT models)", fontsize=11, fontweight='bold')
axes[1, 2].axis('off')

legend = [
    mpatches.Patch(color=np.array(COLORS["GT"])          / 255, label="Ground Truth"),
    mpatches.Patch(color=np.array(COLORS["CellposeSAM"]) / 255, label="Cellpose-SAM FT"),
    mpatches.Patch(color=np.array(COLORS["VISTA"])       / 255, label="VISTA-2D FT"),
    mpatches.Patch(color=np.array(COLORS["StarDist"])    / 255, label="StarDist FT"),
]
axes[1, 2].legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.85)

plt.tight_layout()
out = os.path.join(OUT_DIR, "comparison_all_models.png")
plt.savefig(out, dpi=250, bbox_inches='tight')
plt.close()
print(f"✓ {out}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — CONTOURS SEULEMENT (4 panels, version rapport)
# ════════════════════════════════════════════════════════════════════════════

print("Figure 2 — contours 4 panels (rapport)...")

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle(
    "Cell Segmentation Contours — PUMA (training_set_metastatic_roi_025)",
    fontsize=13, fontweight='bold'
)

for ax, (mask, color_key, title) in zip(axes, [
    (gt_mask,           "GT",          f"Ground Truth ({gt_mask.max()} cells)"),
    (masks["cellpose"], "CellposeSAM", f"Cellpose-SAM FT ({masks['cellpose'].max()} cells)"),
    (masks["vista_ft"], "VISTA",       f"VISTA-2D FT ({masks['vista_ft'].max()} cells)"),
    (masks["stardist"], "StarDist",    f"StarDist FT ({masks['stardist'].max()} cells)"),
]):
    panel = draw_contours(img_u8, mask, COLORS[color_key], thickness=2)
    ax.imshow(panel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
out = os.path.join(OUT_DIR, "comparison_contours.png")
plt.savefig(out, dpi=250, bbox_inches='tight')
plt.close()
print(f"✓ {out}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — VISTA BASELINE VS FT (6 panels avec erreurs)
# ════════════════════════════════════════════════════════════════════════════

print("Figure 3 — Vista baseline vs FT (6 panels)...")

err_base, tp_b, fp_b, fn_b = draw_error_contours(img_u8, masks["vista_baseline"], gt_mask)
err_ft,   tp_f, fp_f, fn_f = draw_error_contours(img_u8, masks["vista_ft"],       gt_mask)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "VISTA-2D: Baseline vs Fine-tuned on PUMA\n"
    "Green=TP  |  Red=FP  |  Orange=FN",
    fontsize=13, fontweight='bold'
)

axes[0, 0].imshow(img_u8)
axes[0, 0].set_title("① Original H&E Image", fontsize=11, fontweight='bold')
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_display)
axes[0, 1].set_title(f"② Ground Truth\n({gt_mask.max()} cells)", fontsize=11, fontweight='bold')
axes[0, 1].axis('off')

axes[0, 2].imshow(mask_to_colored(masks["vista_baseline"]))
axes[0, 2].set_title(
    f"③ VISTA Baseline\n({masks['vista_baseline'].max()} cells predicted)",
    fontsize=11, fontweight='bold'
)
axes[0, 2].axis('off')

axes[1, 0].imshow(mask_to_colored(masks["vista_ft"]))
axes[1, 0].set_title(
    f"④ VISTA FT PUMA\n({masks['vista_ft'].max()} cells predicted)",
    fontsize=11, fontweight='bold'
)
axes[1, 0].axis('off')

axes[1, 1].imshow(err_base)
axes[1, 1].set_title(
    f"⑤ Baseline Errors\nTP={tp_b}  FP={fp_b}  FN={fn_b}",
    fontsize=11, fontweight='bold'
)
axes[1, 1].axis('off')
axes[1, 1].legend(handles=err_legend, loc='lower right', fontsize=8, framealpha=0.85)

axes[1, 2].imshow(err_ft)
axes[1, 2].set_title(
    f"⑥ FT PUMA Errors\nTP={tp_f}  FP={fp_f}  FN={fn_f}",
    fontsize=11, fontweight='bold'
)
axes[1, 2].axis('off')
axes[1, 2].legend(handles=err_legend, loc='lower right', fontsize=8, framealpha=0.85)

plt.tight_layout()
out = os.path.join(OUT_DIR, "comparison_vista_baseline_ft.png")
plt.savefig(out, dpi=250, bbox_inches='tight')
plt.close()
print(f"✓ {out}")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — VISTA BASELINE VS FT (4 panels, version rapport)
# ════════════════════════════════════════════════════════════════════════════

print("Figure 4 — Vista baseline vs FT rapport (4 panels)...")

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
fig.suptitle(
    "VISTA-2D Baseline vs Fine-tuned — PUMA\n"
    "Impact of domain adaptation on cell detection",
    fontsize=13, fontweight='bold'
)

for ax, (panel, title) in zip(axes, [
    (img_u8,    "Original Image"),
    (gt_display, f"Ground Truth ({gt_mask.max()} cells)"),
    (err_base,  f"Baseline  TP={tp_b} | FP={fp_b} | FN={fn_b}"),
    (err_ft,    f"FT PUMA   TP={tp_f} | FP={fp_f} | FN={fn_f}"),
]):
    ax.imshow(panel)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')

fig.legend(
    handles=err_legend, loc='lower center', ncol=3,
    bbox_to_anchor=(0.5, -0.04), fontsize=10, frameon=False
)
plt.tight_layout()
out = os.path.join(OUT_DIR, "comparison_vista_report.png")
plt.savefig(out, dpi=250, bbox_inches='tight')
plt.close()
print(f"✓ {out}")

print("\nToutes les figures générées dans :", OUT_DIR)
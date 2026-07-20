"""
generate_gradcam_figures_v2.py
──────────────────────────────
Genere les figures GradCAM sans la colonne heatmap centrale.
Format : Original/GT a gauche | GradCAM + contours a droite

Usage (depuis ~/Pima/) :
    python3 generate_gradcam_figures_v2.py --name cyto_boss \
        --gt cytoDArk_split/20x/256x256/test/ID14_Aud_Cortex_Tursiops_14_masks.tiff
"""

import os, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NPY_DIR = "gradcam_outputs"
OUT_DIR = "gradcam_outputs"

MODEL_COLORS = {
    "CellposeSAM": (255, 165,   0),
    "VISTA-2D":    ( 80, 230,   0),
    "StarDist":    (  0, 160, 255),
}


def load_npy(path):
    return np.load(path) if os.path.exists(path) else None


def safe_u8(img):
    if img.max() > 255:
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip((img.astype(np.float32) - p1) / (p99 - p1 + 1e-8), 0, 1) * 255
    return img.astype(np.uint8)


def load_gt(gt_path):
    if gt_path is None or not os.path.exists(gt_path):
        return None
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        return None
    if gt.ndim > 2:
        gt = gt[:, :, 0]
    return gt.astype(np.int32)


def overlay_heatmap_on_image(img_u8, cam, alpha_img=0.5, alpha_heat=0.5):
    norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * norm), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_u8, alpha_img, heatmap_rgb, alpha_heat, 0)


def draw_contours(base_img, mask, color_rgb, thickness=2):
    color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
    out = cv2.cvtColor(base_img, cv2.COLOR_RGB2BGR)
    for cid in np.unique(mask):
        if cid == 0: continue
        binary = (mask == cid).astype(np.uint8)
        ctrs, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, color_bgr, thickness)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def mask_to_colored(mask):
    np.random.seed(42)
    n = int(mask.max()) + 1
    colors = np.random.randint(60, 230, (n, 3), dtype=np.uint8)
    colors[0] = [15, 15, 15]
    return colors[mask.astype(int)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--gt",   default=None)
    args = parser.parse_args()
    name = args.name

    img_raw = load_npy(os.path.join(NPY_DIR, f"img_{name}.npy"))
    if img_raw is None:
        print(f"[!] img_{name}.npy introuvable")
        return

    img_u8 = safe_u8(img_raw)
    if img_u8.ndim == 3 and img_u8.shape[-1] > 3:
        img_u8 = img_u8[:, :, :3]

    cams = {
        "CellposeSAM": load_npy(os.path.join(NPY_DIR, f"cam_cellpose_{name}.npy")),
        "VISTA-2D":    load_npy(os.path.join(NPY_DIR, f"cam_vista_{name}.npy")),
        "StarDist":    load_npy(os.path.join(NPY_DIR, f"cam_stardist_{name}.npy")),
    }
    masks = {
        "CellposeSAM": load_npy(os.path.join(NPY_DIR, f"masks_cellpose_{name}.npy")),
        "VISTA-2D":    load_npy(os.path.join(NPY_DIR, f"masks_vista_{name}.npy")),
        "StarDist":    load_npy(os.path.join(NPY_DIR, f"masks_stardist_{name}.npy")),
    }
    gt_mask = load_gt(args.gt)
    available = [m for m in cams if cams[m] is not None]
    print(f"Modeles disponibles : {available}")

    # ── FIGURE 1 — Version rapport : 2 colonnes par modele (GT | GradCAM+contours) ──
    # Disposition : 1 ligne par modele, 2 panels chacun
    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(12, 5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"GradCAM Analysis — {name}\n"
        "Left: Ground Truth | Right: GradCAM activation + predicted contours",
        fontsize=13, fontweight='bold'
    )

    for row, model_name in enumerate(available):
        cam   = cams[model_name]
        mask  = masks[model_name]
        color = MODEL_COLORS[model_name]

        # Panel gauche — GT coloré ou image avec contours GT
        if gt_mask is not None:
            gt_display = draw_contours(img_u8, gt_mask, (255, 255, 255), thickness=2)
            axes[row, 0].imshow(gt_display)
            axes[row, 0].set_title(
                f"Ground Truth ({gt_mask.max()} cells)",
                fontsize=11, fontweight='bold'
            )
        else:
            axes[row, 0].imshow(img_u8)
            axes[row, 0].set_title("Original Image", fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')
        # Etiquette modele sur le cote gauche
        axes[row, 0].set_ylabel(model_name, fontsize=12, fontweight='bold', rotation=90,
                                labelpad=10, va='center')

        # Panel droit — GradCAM superpose + contours de prediction
        overlay = overlay_heatmap_on_image(img_u8, cam, alpha_img=0.45, alpha_heat=0.45)
        if mask is not None:
            overlay = draw_contours(overlay, mask, color, thickness=2)
        n_cells = int(mask.max()) if mask is not None else 0
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"{model_name} — GradCAM + predicted contours ({n_cells} cells)",
            fontsize=11, fontweight='bold'
        )
        axes[row, 1].axis('off')

    plt.tight_layout()
    out1 = os.path.join(OUT_DIR, f"gradcam_report_v2_{name}.png")
    plt.savefig(out1, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"✓ {out1}")

    # ── FIGURE 2 — 1 ligne, n+2 colonnes : Original | GT | modele1 | modele2 | modele3 ──
    fig, axes = plt.subplots(1, n + 2, figsize=(5 * (n + 2), 6))
    fig.suptitle(
        f"GradCAM Comparison — {name}\n"
        "GradCAM activation + predicted contours vs Ground Truth",
        fontsize=13, fontweight='bold'
    )

    # Colonne 0 : image originale
    axes[0].imshow(img_u8)
    axes[0].set_title("Original Image", fontsize=10, fontweight='bold')
    axes[0].axis('off')

    # Colonne 1 : GT une seule fois
    if gt_mask is not None:
        gt_display = draw_contours(img_u8, gt_mask, (255, 255, 255), thickness=2)
        axes[1].imshow(gt_display)
        axes[1].set_title(
            f"Ground Truth\n({gt_mask.max()} cells)",
            fontsize=10, fontweight='bold'
        )
    else:
        axes[1].imshow(img_u8)
        axes[1].set_title("Original Image", fontsize=10)
    axes[1].axis('off')

    # Colonnes suivantes : GradCAM + contours par modele
    for col, model_name in enumerate(available, start=2):
        cam   = cams[model_name]
        mask  = masks[model_name]
        color = MODEL_COLORS[model_name]
        n_cells = int(mask.max()) if mask is not None else 0

        overlay = overlay_heatmap_on_image(img_u8, cam, alpha_img=0.45, alpha_heat=0.45)
        if mask is not None:
            overlay = draw_contours(overlay, mask, color, thickness=2)
        axes[col].imshow(overlay)
        axes[col].set_title(
            f"{model_name}\nGradCAM + contours ({n_cells} cells)",
            fontsize=10, fontweight='bold'
        )
        axes[col].axis('off')

    plt.tight_layout()
    out2 = os.path.join(OUT_DIR, f"gradcam_comparison_v2_{name}.png")
    plt.savefig(out2, dpi=220, bbox_inches='tight')
    plt.close()
    print(f"✓ {out2}")
    print("\nTermine !")


if __name__ == "__main__":
    main()
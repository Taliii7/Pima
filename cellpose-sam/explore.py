"""
explore_architecture.py
-----------------------
Outil de diagnostic pour explorer l'architecture du modèle CellposeSAM
et identifier les meilleurs points d'accroche pour les hooks XAI.

Usage :
    python3 explore_architecture.py --model models/cellposeSam_puma
"""

import sys, os, argparse
import torch
import numpy as np
from cellpose import models

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'common'))
from utils import deviceChoice


def explore_model(model_path=None):
    device = deviceChoice()
    is_gpu = device.type != 'cpu'

    print("Chargement du modèle…")
    if model_path and os.path.exists(model_path):
        try:
            model = models.CellposeModel(gpu=is_gpu, pretrained_model=model_path, device=device)
        except KeyError:
            model = models.CellposeModel(gpu=is_gpu, device=device)
            sd = torch.load(model_path, map_location=device)
            if isinstance(sd, dict):
                for k in ('model_state_dict', 'state_dict', 'net', 'model'):
                    if k in sd:
                        sd = sd[k]; break
            model.net.load_state_dict(sd, strict=False)
    else:
        model = models.CellposeModel(gpu=is_gpu, device=device)

    net = model.net

    print("\n" + "=" * 70)
    print("  ARCHITECTURE COMPLÈTE — tous les modules nommés")
    print("=" * 70)

    # Catégorisation des modules
    attention_blocks = []
    encoder_blocks   = []
    neck_blocks      = []
    decoder_blocks   = []
    other_blocks     = []

    all_named = list(net.named_modules())

    for name, module in all_named:
        class_name = module.__class__.__name__
        n_params   = sum(p.numel() for p in module.parameters(recurse=False))
        n_params_total = sum(p.numel() for p in module.parameters())

        info = f"  {name:<55} | {class_name:<30} | params={n_params_total:>10,}"

        if 'attn' in name.lower() or 'attention' in name.lower():
            attention_blocks.append((name, module, info))
        elif 'encoder' in name.lower() and 'neck' not in name.lower():
            encoder_blocks.append((name, module, info))
        elif 'neck' in name.lower():
            neck_blocks.append((name, module, info))
        elif 'decoder' in name.lower() or 'output' in name.lower():
            decoder_blocks.append((name, module, info))
        elif name:  # exclut le module racine vide
            other_blocks.append((name, module, info))

    def print_group(label, group, max_show=None):
        print(f"\n{'─'*70}")
        print(f"  {label} ({len(group)} modules)")
        print(f"{'─'*70}")
        items = group if max_show is None else group[:max_show]
        for _, _, info in items:
            print(info)
        if max_show and len(group) > max_show:
            print(f"  … et {len(group) - max_show} autres")

    print_group("🔷 BLOCS D'ATTENTION (Transformer / ViT)", attention_blocks, max_show=30)
    print_group("🟩 BLOCS ENCODEUR", encoder_blocks, max_show=20)
    print_group("🟧 NECK (projection encodeur→décodeur)", neck_blocks)
    print_group("🟥 DÉCODEUR / SORTIES", decoder_blocks, max_show=20)

    # ── Recommandations XAI ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RECOMMANDATIONS POUR LES HOOKS XAI")
    print("=" * 70)

    reco = []

    # Dernier bloc d'attention numéroté (le plus profond = le plus sémantique)
    numbered_attn = [
        (n, m) for n, m in all_named
        if n.split('.')[-1].isdigit()
        and ('block' in n.lower() or 'layer' in n.lower() or 'encoder' in n.lower())
    ]
    if numbered_attn:
        best_name, best_mod = numbered_attn[-1]
        reco.append(("Dernier bloc numéroté (haute sémantique)", best_name, best_mod))

    # Dernier bloc d'attention
    if attention_blocks:
        best_name, best_mod, _ = attention_blocks[-1]
        reco.append(("Dernier bloc d'attention", best_name, best_mod))

    # Dernier bloc d'encodeur avant le neck
    if encoder_blocks:
        best_name, best_mod, _ = encoder_blocks[-1]
        reco.append(("Dernier bloc encodeur (avant neck)", best_name, best_mod))

    # Neck (ce que tu avais déjà)
    if neck_blocks:
        best_name, best_mod, _ = neck_blocks[-1]
        reco.append(("Neck — projection finale (actuel)", best_name, best_mod))

    for label, name, _ in reco:
        print(f"\n  ✓ {label}")
        print(f"    Nom : {name}")
        print(f"    → Ajout dans inferenceCellposeSam.py :")
        hook_key = name.replace(".", "_")
        print(f"      hook_key = '{hook_key}'")
        print(f"      net.get_submodule('{name}').register_forward_hook(get_features_hook(hook_key))")

    # ── Résumé paramètres ────────────────────────────────────────────────
    total_params = sum(p.numel() for p in net.parameters())
    print(f"\n{'─'*70}")
    print(f"  Total paramètres du réseau : {total_params:,}")
    print(f"  Profondeur (nb modules nommés) : {len(all_named)}")
    print("=" * 70)

    return net


def main():
    parser = argparse.ArgumentParser(description="Exploration architecture CellposeSAM")
    parser.add_argument("--model", type=str, default=None,
                        help="Chemin du modèle (laisser vide pour le modèle par défaut)")
    args = parser.parse_args()
    explore_model(args.model)


if __name__ == "__main__":
    main()
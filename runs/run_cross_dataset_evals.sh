#!/bin/bash

# ==========================================
# ÉVALUATION CROSS-DATASET
# Objectif : mesurer la généralisation des modèles
#   - Modèles CytoDArk testés sur PUMA
#   - Modèles PUMA testés sur CytoDArk
# ==========================================

CSV_PATH="../resultats_comparaison/results_cross_dataset.csv"
CYTO_TEST="../cytoDArk_split/20x/256x256/test"
PUMA_TEST="../puma_data/test"

source $(conda info --base)/etc/profile.d/conda.sh

# ==========================================
# 1. MODÈLES CYTODARK → TESTÉS SUR PUMA
# ==========================================
echo "==================================================="
echo " PARTIE 1 : Modèles CytoDArk → Test PUMA"
echo "==================================================="

conda activate pasteur_env
cd cellpose-sam

echo ">> Cellpose FT CytoDArk → PUMA"
python3 eval_cellposeSam.py --test_dir "$PUMA_TEST" \
    --model "models/cellposeSam_cytodark_complet_epoch_0005" \
    --modele_nom "Cellpose_FT_Cyto_sur_PUMA" \
    --famille "Cellpose-SAM" \
    --zoom "cross" \
    --dim "1024" \
    --csv "$CSV_PATH"

conda deactivate
cd ..

conda activate vista
cd vista2d

echo ">> VISTA FT CytoDArk → PUMA"
python3 eval_vista.py --test_dir "$PUMA_TEST" \
    --model "models/model.pt" \
    --modele_nom "VISTA_FT_Cyto_sur_PUMA" \
    --famille "VISTA-2D" \
    --zoom "cross" \
    --dim "1024" \
    --csv "$CSV_PATH"

conda deactivate
cd ..

# ==========================================
# 2. MODÈLES PUMA → TESTÉS SUR CYTODARK
# ==========================================
echo "==================================================="
echo " PARTIE 2 : Modèles PUMA → Test CytoDArk"
echo "==================================================="

conda activate pasteur_env
cd cellpose-sam

echo ">> Cellpose FT PUMA → CytoDArk"
python3 eval_cellposeSam.py --test_dir "$CYTO_TEST" \
    --model "models/cellposeSam_puma" \
    --modele_nom "Cellpose_FT_PUMA_sur_Cyto" \
    --famille "Cellpose-SAM" \
    --zoom "cross" \
    --dim "256" \
    --csv "$CSV_PATH"

conda deactivate
cd ..

conda activate vista
cd vista2d

echo ">> VISTA FT PUMA → CytoDArk"
python3 eval_vista.py --test_dir "$CYTO_TEST" \
    --model "models/puma/vista_puma.pt" \
    --modele_nom "VISTA_FT_PUMA_sur_Cyto" \
    --famille "VISTA-2D" \
    --zoom "cross" \
    --dim "256" \
    --csv "$CSV_PATH"

conda deactivate
cd ..

echo "==================================================="
echo " Évaluations cross-dataset terminées !"
echo " CSV : $CSV_PATH"
echo "==================================================="
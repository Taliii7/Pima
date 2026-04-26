#!/bin/bash

# ==========================================
# 1. PARAMÈTRES GLOBAUX
# ==========================================
CSV_PATH="../resultats_comparaison/results_puma.csv"
DATASET_ROOT="../puma_data"
TEST_DIR="$DATASET_ROOT/test"

# ==========================================
# 2. ACTIVATION DE L'ENVIRONNEMENT
# ==========================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pasteur_env

# On rentre dans le dossier Cellpose
cd cellpose-sam

# ==========================================
# 3. ÉVALUATIONS SUR PUMA
# ==========================================
echo "==================================================="
echo " Début de l'évaluation des modèles sur PUMA"
echo " Test dir : $TEST_DIR"
echo "==================================================="

# --- TEST 1 : Baseline (sans fine-tuning) ---
echo ">> Évaluation 1/2 : Baseline (Sans fine-tuning)"
python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
    --model "baseline" \
    --modele_nom "Cellpose_Baseline" \
    --famille "Cellpose-SAM" \
    --zoom "puma" \
    --dim "1024" \
    --csv "$CSV_PATH"

# --- TEST 2 : Modèle fine-tuné sur PUMA ---
echo ">> Évaluation 2/2 : Modèle Fine-tuné sur PUMA"
python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
    --model "models/cellposeSam_puma" \
    --modele_nom "Cellpose_FT_PUMA" \
    --famille "Cellpose-SAM" \
    --zoom "puma" \
    --dim "1024" \
    --csv "$CSV_PATH"

# ==========================================
# 4. NETTOYAGE FIN DE SCRIPT
# ==========================================
conda deactivate
cd ..
echo "==================================================="
echo " Évaluations PUMA terminées ! CSV : $CSV_PATH"
echo "==================================================="

#!/bin/bash

# ==========================================
# 1. PARAMÈTRES GLOBAUX
# ==========================================
CSV_PATH="../resultats_comparaison/results_puma.csv"
TEST_DIR="../puma_data_converted/test"

# ==========================================
# 2. ACTIVATION DE L'ENVIRONNEMENT
# ==========================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vista

cd vista2d

# ==========================================
# 3. ÉVALUATIONS VISTA-2D SUR PUMA
# ==========================================
echo "==================================================="
echo " Début de l'évaluation VISTA-2D sur PUMA"
echo " Test dir : $TEST_DIR"
echo "==================================================="

# --- TEST 1 : Baseline (zero-shot) ---
echo ">> Évaluation 1/2 : VISTA-2D Baseline (zero-shot)"
python3 eval_vista.py --test_dir "$TEST_DIR" \
    --model "models/model_baseline.pt" \
    --modele_nom "Vista_Baseline" \
    --famille "VISTA-2D" \
    --zoom "puma" \
    --dim "1024" \
    --csv "$CSV_PATH"

# --- TEST 2 : Fine-tuné sur PUMA ---
echo ">> Évaluation 2/2 : VISTA-2D Fine-tuné sur PUMA"
python3 eval_vista.py --test_dir "$TEST_DIR" \
    --model "models/puma/vista_puma.pt" \
    --modele_nom "Vista_FT_PUMA" \
    --famille "VISTA-2D" \
    --zoom "puma" \
    --dim "1024" \
    --csv "$CSV_PATH"

# ==========================================
# 4. NETTOYAGE FIN DE SCRIPT
# ==========================================
conda deactivate
cd ..
echo "==================================================="
echo " Évaluations VISTA-2D PUMA terminées ! CSV : $CSV_PATH"
echo "==================================================="

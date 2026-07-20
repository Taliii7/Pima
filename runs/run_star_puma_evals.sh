#!/bin/bash



# ==========================================
# 2. ACTIVATION DE L'ENVIRONNEMENT
# ==========================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate stardist_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# On rentre dans le dossier stardist
cd stardist

# ==========================================
# 3. ÉVALUATIONS SUR PUMA
# ==========================================
echo "==================================================="
echo " Début de l'évaluation des modèles sur PUMA"
echo " Test dir : puma_data_converted/test"
echo "==================================================="

# --- TEST 1 : Baseline (sans fine-tuning) ---
echo ">> Évaluation 1/2 : Baseline (Sans FT)"
python3 eval_StarDist.py \
    --test_dir "../puma_data_converted/test" \
    --model "2D_versatile_he" \
    --modele_nom "StarDist_Baseline" \
    --famille "StarDist" \
    --zoom "puma" \
    --dim "1024" \
    --csv "../resultats_comparaison/results_puma.csv"

# --- TEST 2 : Modèle fine-tuné sur PUMA ---
echo ">> Évaluation 2/2 : Modèle StarDist FT sur PUMA"
    python3 eval_StarDist.py \
    --test_dir "../puma_data_converted/test" \
    --model "stardist_puma" \
    --modele_nom "StarDist_FT_PUMA" \
    --famille "StarDist" \
    --zoom "puma" \
    --dim "1024" \
    --csv "../resultats_comparaison/results_puma.csv"

# ==========================================
# 4. NETTOYAGE FIN DE SCRIPT
# ==========================================
conda deactivate
cd ..
echo "==================================================="
echo " Évaluations PUMA terminées ! CSV : resultats_comparaison/results_puma.csv"
echo "==================================================="


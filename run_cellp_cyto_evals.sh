#!/bin/bash

# ==========================================
# 1. PARAMÈTRES GLOBAUX
# ==========================================
CSV_PATH="../resultats_comparaison/results_cytoDark.csv"
DATASET_ROOT="../cytoDArk_split"

# ==========================================
# 2. ACTIVATION DE L'ENVIRONNEMENT
# ==========================================
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pasteur_env

# On rentre dans le dossier Cellpose
cd cellpose-sam

# ==========================================
# 3. LISTE DES CONFIGURATIONS À TESTER
# ==========================================
ZOOMS=("20x" "40x" "20x_40x")
DIMS=("256" "512" "1024" "2048")

echo " Début de l'évaluation automatique des 4 modèles Cellpose..."

# ==========================================
# 4. LA GRANDE BOUCLE D'ÉVALUATION
# ==========================================
for zoom in "${ZOOMS[@]}"; do
    for dim in "${DIMS[@]}"; do
        
        # Chemin du sous-dossier de test
        TEST_DIR="$DATASET_ROOT/$zoom/${dim}x${dim}/test"
        
        # Si le dossier n'existe pas (ex: 20x en 2048), le script l'ignore
        if [ -d "$TEST_DIR" ]; then
            echo "==================================================="
            echo "DOSSIER EN COURS : $zoom / Dimension : $dim"
            echo "==================================================="
            
            # --- TEST 1 : La Baseline ---
            echo ">> Évaluation 1/4 : Baseline (Sans fine-tuning)"
            python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
                --model "baseline" --modele_nom "Cellpose_Baseline" \
                --famille "Cellpose-SAM" --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"
            
            # --- TEST 2 : Modèle entraîné sur 20x ---
            echo ">> Évaluation 2/4 : Modèle Fine-tuné (20x uniquement)"
            python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
                --model "models/cellposeSam_cytodark_20_epoch_0020" --modele_nom "Cellpose_FT_20x" \
                --famille "Cellpose-SAM" --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

            # --- TEST 3 : Modèle entraîné sur 40x ---
            echo ">> Évaluation 3/4 : Modèle Fine-tuné (40x uniquement)"
            python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
                --model "models/cellposeSam_cytodark_40_epoch_0005" --modele_nom "Cellpose_FT_40x" \
                --famille "Cellpose-SAM" --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

            # --- TEST 4 : Modèle Complet (20x + 40x) ---
            echo ">> Évaluation 4/4 : Modèle Fine-tuné Complet"
            python3 eval_cellposeSam.py --test_dir "$TEST_DIR" \
                --model "models/cellposeSam_cytodark_complet_epoch_0005" --modele_nom "Cellpose_FT_Complet" \
                --famille "Cellpose-SAM" --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"
        fi
    done
done

# ==========================================
# 5. NETTOYAGE FIN DE SCRIPT
# ==========================================
conda deactivate
cd ..
echo "Toutes les évaluations Cellpose sont terminées ! Le fichier CSV est rempli."

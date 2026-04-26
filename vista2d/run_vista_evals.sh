#!/bin/bash

CSV_PATH="../resultats_comparaison/metriques_vista.csv"
DATASET_ROOT="../cytoDArk_split"


source $(conda info --base)/etc/profile.d/conda.sh
conda activate pasteur_env

cd vista2d


ZOOMS=("20x" "40x" "20x_40x")
DIMS=("256" "512" "1024" "2048")

echo "Début de l'évaluation automatique de VISTA-2D..."

for zoom in "${ZOOMS[@]}"; do
    for dim in "${DIMS[@]}"; do
        
        TEST_DIR="$DATASET_ROOT/$zoom/${dim}x${dim}/test"
        
        if [ -d "$TEST_DIR" ]; then
            echo "==================================================="
            echo "DOSSIER EN COURS : $zoom / Dimension : $dim"
            echo "==================================================="
            
            # --- TEST 1 : VISTA GLOBAL ---
            echo ">> Évaluation 1/3 : VISTA-2D (Modèle Global)"
            python3 eval_vista.py --test_dir "$TEST_DIR" \
                --model "models/model.pt" \
                --modele_nom "Vista_baseline" --famille "VISTA-2D" \
                --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

            # --- TEST 2 : VISTA FT 22 epoches ---
            echo ">> Évaluation 2/3 : VISTA-2D (Entraîné sur 20x)"
            python3 eval_vista.py --test_dir "$TEST_DIR" \
                --model "models/model_ft_22e.pt" \
                --modele_nom "Vista_22e" --famille "VISTA-2D" \
                --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

            # --- TEST 3 : VISTA FT 50 epoches ---
            echo ">> Évaluation 3/3 : VISTA-2D (Entraîné sur 40x)"
            python3 eval_vista.py --test_dir "$TEST_DIR" \
                --model "models/model_final.pt" \
                --modele_nom "Vista_final" --famille "VISTA-2D" \
                --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"
        fi
    done
done


conda deactivate
cd ..
echo "L'évaluation VISTA-2D est terminée ! Les résultats sont dans metriques_vista.csv."
#!/bin/bash

CSV_PATH="../resultats_comparaison/results_cytoDark.csv"
DATASET_ROOT="../cytoDArk_split"


source $(conda info --base)/etc/profile.d/conda.sh
conda activate vista

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
            echo ">> Évaluation 1/3 : VISTA-2D (base line : zero shot))"
            python3 eval_vista.py --test_dir "$TEST_DIR" \
                --model "models/model_baseline.pt" \
                --modele_nom "Vista_baseline" --famille "VISTA-2D" \
                --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

            # --- TEST 2 : VISTA FT ---
            echo ">> Évaluation 2/3 : VISTA-2D (Entraîné sur X  epochs)"
            python3 eval_vista.py --test_dir "$TEST_DIR" \
                --model "models/vista_cyto.pt" \
                --modele_nom "Vista_FT_CYTO" --famille "VISTA-2D" \
                --zoom "$zoom" --dim "$dim" --csv "$CSV_PATH"

        fi
    done
done


conda deactivate
cd ..
echo "L'évaluation VISTA-2D est terminée ! Les résultats sont dans results_cytoDark.csv ."

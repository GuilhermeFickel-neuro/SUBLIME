#!/bin/bash

# List of configurations to run
allowed_configs=(
  "config106" "config147" "config171" "config156" "config164" 
  "config133" "config159" "config144" "config163" "config151" 
  "config160" "config155" "config168" "config118" "config149" 
  "config107" "config146" "config145" "config165" "config153"
)

# Get all subdirectories in model_experiments/
for dir in model_experiments/*/; do
  # Extract directory name without trailing slash
  dir_name=$(basename "$dir")
  
  # # Skip if not in allowed configurations list
  # if [[ ! " ${allowed_configs[@]} " =~ " ${dir_name} " ]]; then
  #   echo "Skipping configuration: $dir_name (not in allowed list)"
  #   continue
  # fi
  
  # Create corresponding output directory if it doesn't exist
  mkdir -p "sublime_eval/$dir_name"
  
  echo "Running evaluation for $dir_name..."
  
  # Run the command with the appropriate directories
  python evaluate_sublime_on_test_data.py \
    --neurolake-csv SAFRA_train_enriquecido_v3_20k.csv \
    --dataset-features-csv SAFRA_train_features_v2_updated_2025_20k.csv \
    --model-dir "$dir" \
    --target-column alvo \
    --output-dir "sublime_eval/$dir_name/" \
    --use-loaded-adj-for-extraction \
    --extract-embeddings-only \
    --n-trials 30 \
    --batch-size 32

  python evaluate_sublime_on_test_data.py \
    --neurolake-csv SAFRA_train_enriquecido_v3_20k.csv \
    --dataset-features-csv SAFRA_train_features_v2_updated_2025_20k.csv \
    --model-dir "$dir" \
    --target-column alvo \
    --output-dir "sublime_eval/$dir_name/" \
    --extract-embeddings-only \
    --generate-new-anchor-adj-for-eval \
    --n-trials 30 \
    --batch-size 32
  
  echo "Completed evaluation for $dir_name"
done

echo "All evaluations completed!"
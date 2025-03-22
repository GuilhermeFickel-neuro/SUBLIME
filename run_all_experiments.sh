#!/bin/bash

# Get all subdirectories in model_experiments/
for dir in model_experiments/*/; do
  # Extract directory name without trailing slash
  dir_name=$(basename "$dir")
  
  # Create corresponding output directory if it doesn't exist
  mkdir -p "sublime_eval/$dir_name"
  
  echo "Running evaluation for $dir_name..."
  
  # Run the command with the appropriate directories
  python evaluate_sublime_on_test_data.py \
    --neurolake-csv SAFRA_enriquecimento_100k_v8.csv \
    --dataset-features-csv SAFRA_features_100k_v9.csv \
    --model-dir "$dir" \
    --target-column alvo \
    --output-dir "sublime_eval/$dir_name/" \
    --n-trials 30 \
    --batch-size 32
  
  echo "Completed evaluation for $dir_name"
done

echo "All evaluations completed!"
# Extended SUBLIME: Towards Unsupervised Deep Graph Structure Learning

This is an extended implementation of the WWW-2022 paper "[Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/pdf/2201.06367.pdf)" (SUBLIME), with significant enhancements for real-world applications including ArcFace loss integration, multi-modal graph construction, and memory-efficient training.

![The proposed framework](pipeline.png)

## Architecture Overview

This framework extends the original SUBLIME with several key enhancements:

### Multi-Modal Graph Construction
The system constructs graphs by combining three complementary sources:

1. **Feature Similarity Graph**: KNN graph based on cosine similarity of preprocessed features
2. **Relationship Graph**: Explicit relationships from external data (e.g., family/business connections via CPF identifiers)  
3. **Geographical Graph**: Spatial proximity using latitude/longitude coordinates with distance-weighted k-NN

These graphs are combined using element-wise maximum operations to create a rich, multi-modal anchor graph.

### Enhanced Learning Components

- **ArcFace Loss Integration**: Implements both standard and memory-efficient sampled ArcFace loss for better embedding discrimination
- **Binary Classification Head**: Optional MLP head for supervised tasks on annotated subsets
- **Three-Phase Training**: 
  - Phase 1: Train embeddings only using anchor graph
  - Phase 2: Train graph learner only with frozen embeddings  
  - Phase 3: Joint training of both components
- **Structure Bootstrapping**: Dynamically updates anchor graph by blending with learned graph structure

### Memory Optimizations

- **Chunked Processing**: Processes large graphs in memory-efficient chunks
- **Sampled ArcFace**: Samples subset of classes to reduce memory footprint
- **Sparse Operations**: Efficient sparse tensor operations throughout

### Graph Learners
Supports multiple graph structure learning approaches:
- **FGP**: Feature-based graph propagation learner
- **MLP**: Multi-layer perceptron learner  
- **ATT**: Attention-based learner
- **GNN**: Graph neural network learner

## Installation

```bash
chmod +x install.sh
./install.sh
conda activate OpenGSL
```

## Training

### Basic Training Command

```bash
python train_person_data.py \
  -dataset neurolake_target_train_50k_v8_1k.csv \
  -annotated_dataset neurolake_target_train_50k_v8.csv \
  -annotation_column target_value \
  -relationship_dataset neurolake_relation_target_train_42k_v8.csv \
  --use_geo_graph 1 --latitude_col ifdmcpf004c00v02 --longitude_col ifdmcpf005c00v02 \
  -epochs 11500 -lr 0.001 -hidden_dim 256 -rep_dim 48 \
  -type_learner mlp -use_one_cycle 1 \
  -output_dir "saved_models/my_experiment"
```

### Key Training Parameters

**Data Configuration:**
- `-dataset`: Main training dataset (CSV/TSV format)
- `-annotated_dataset`: Optional dataset with binary labels for classification
- `-annotation_column`: Column name containing binary target values (0/1)
- `-relationship_dataset`: CSV with relationship pairs (CPF columns)
- `-drop_columns_file`: CSV listing feature columns to exclude

**Graph Construction:**
- `-k`: Number of neighbors for KNN graph (default: 30)
- `--use_geo_graph 1`: Enable geographical graph construction
- `--latitude_col`, `--longitude_col`: Column names for coordinates
- `--geo_k`: Geographical neighbors per node (default: 10)
- `--relationship_weight`: Weight for relationship edges (default: 1.0)

**Model Architecture:**
- `-hidden_dim`: Hidden layer dimensions (default: 512)
- `-rep_dim`: Final embedding dimension (default: 64) 
- `-proj_dim`: Projection head dimension (default: 64)
- `-nlayers`: Number of GCN layers (default: 2)
- `-type_learner`: Graph learner type (fgp|mlp|att|gnn)

**Training Configuration:**
- `-epochs`: Total training epochs
- `-lr`: Learning rate for main model
- `-lr_learner`: Learning rate for graph learner
- `--embedding_only_epochs`: Phase 1 duration (embedding-only training)
- `--graph_learner_only_epochs`: Phase 2 duration (learner-only training)

**Memory & Performance:**
- `-sparse 1`: Use sparse operations (recommended)
- `-use_sampled_arcface 1`: Enable memory-efficient ArcFace
- `-arcface_num_samples`: Number of sampled classes for ArcFace

### Advanced Features

**ArcFace Loss:**
```bash
-use_arcface 1 -arcface_scale 30.0 -arcface_margin 0.5 \
-use_sampled_arcface 1 -arcface_num_samples 4000
```

**Phase-based Training:**
```bash
--embedding_only_epochs 1000 --graph_learner_only_epochs 500 -epochs 5000
```

**One-Cycle Learning Rate:**
```bash
-use_one_cycle 1 -one_cycle_pct_start 0.3
```

## Evaluation

### Running Evaluation

The evaluation system extracts embeddings from trained models and evaluates them using multiple ML algorithms:

```bash
python evaluate_sublime_on_test_data.py \
  --model_dir saved_models/my_experiment \
  --test_data_path test_dataset.csv \
  --target_column target_value \
  --output_dir evaluation_results \
  --k_neighbors 50 \
  --optuna_trials 100
```

### Evaluation Methodology

The evaluation follows a comprehensive methodology:

1. **Model Loading**: Loads trained SUBLIME model, graph learner, and preprocessing pipeline
2. **Test Data Processing**: Applies same preprocessing as training data
3. **Graph Construction**: Builds evaluation graph using same multi-modal approach
4. **Embedding Extraction**: Generates embeddings for test samples
5. **Feature Engineering**: Creates multiple feature sets:
   - Raw embeddings
   - KNN-based features (distances, label statistics from training neighbors)
   - Classification probabilities (if available)
   - Combined feature sets

6. **Multi-Model Evaluation**: Tests multiple algorithms with Optuna hyperparameter optimization:
   - XGBoost, LightGBM, CatBoost (gradient boosting)
   - Random Forest, Extra Trees (ensemble methods)
   - Logistic Regression (linear baseline)

7. **Stacking Ensemble**: Creates meta-learner combining all base models

8. **Comprehensive Reporting**:
   - ROC curves and AUC scores
   - Feature importance analysis  
   - Best hyperparameters for each model
   - Detailed performance metrics

### Batch Evaluation

For multiple experiments, use the batch evaluation script:

```bash
bash run_all_experiments_multi_datasets.sh
```

This script:
- Processes multiple model directories automatically
- Handles different dataset configurations
- Generates comparative results across experiments
- Supports parallel processing for efficiency

### Evaluation Parameters

**Core Settings:**
- `--model_dir`: Path to trained model directory
- `--test_data_path`: Test dataset file
- `--target_column`: Binary target column name
- `--output_dir`: Results output directory

**Feature Engineering:**
- `--k_neighbors`: Number of neighbors for KNN features (default: 50)
- `--sample_size`: Sample size for large datasets (default: 10000)

**Optimization:**
- `--optuna_trials`: Hyperparameter tuning trials per model (default: 60)
- `--enable_stacking`: Enable ensemble stacking (default: True)

**Output Control:**
- `--save_roc_plots`: Generate ROC curve visualizations
- `--save_feature_importance`: Generate feature importance plots
- `--verbose`: Detailed logging output


## Citation

If you use this extended framework, please cite both the original SUBLIME paper and acknowledge the extensions:

```bibtex
@inproceedings{liu2022towards,
  title={Towards unsupervised deep graph structure learning},
  author={Liu, Yixin and Zheng, Yu and Zhang, Daokun and Chen, Hongxu and Peng, Hao and Pan, Shirui},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1392--1403},
  year={2022}
}
```

import argparse
import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
import os
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from main import Experiment
from utils import sparse_mx_to_torch_sparse_tensor, knn_fast, get_memory_usage
from main import create_parser # Import the shared parser creation function
import warnings
import gc # Import garbage collector
from sklearn.model_selection import train_test_split # Import for validation split

# Add device selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2, earth_radius_km=6371):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = earth_radius_km
    return c * r

def find_column_case_insensitive(df_columns, target_name_lower):
    """Finds the actual column name in a list of columns, ignoring case."""
    for col in df_columns:
        if col.lower() == target_name_lower:
            return col
    return None

def _check_processed_data(data, dataset_name=""):
    """Internal helper to check processed data for issues."""
    print(f"\n--- Post-processing Checks ({dataset_name}) ---")
    if not isinstance(data, np.ndarray):
        print(f"Error ({dataset_name}): Processed data is not a numpy array (type: {type(data)}).")
        return False, data # Return False indicating failure

    try:
        data = data.astype(np.float32) # Ensure float32
    except ValueError as e:
        print(f"Error ({dataset_name}): Could not convert processed data to float32: {e}")
        return False, data # Return False indicating failure

    if np.isnan(data).any() or np.isinf(data).any():
        print(f"Warning ({dataset_name}): NaN or Infinite values found!")
        nan_cols = np.where(np.isnan(data).any(axis=0))[0]
        inf_cols = np.where(np.isinf(data).any(axis=0))[0]
        print(f"  Indices of columns with NaN: {nan_cols}")
        print(f"  Indices of columns with Inf: {inf_cols}")
        # Option to impute: data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        # return False, data # Indicate potential issue even if imputed
    else:
        print(f"OK ({dataset_name}): No NaN or Infinite values found.")

    try:
        variances = np.var(data, axis=0)
        low_variance_threshold = 1e-6
        low_variance_indices = np.where(variances < low_variance_threshold)[0]
        if len(low_variance_indices) > 0:
            print(f"Warning ({dataset_name}): {len(low_variance_indices)} columns have variance < {low_variance_threshold}.")
            print(f"  Indices of low-variance columns: {low_variance_indices}")
        else:
            print(f"OK ({dataset_name}): All columns have variance >= {low_variance_threshold}.")
    except Exception as e:
        print(f"Error ({dataset_name}) during variance check: {e}")
        return False, data # Return False indicating failure

    return True, data # Return True indicating success

def preprocess_mixed_data(df_main, df_annotated=None, output_dir=None, target_column=None, args=None):
    """
    Preprocess main and optional annotated dataframes using a single transformer.
    Transformer is fitted ONLY on df_main if not loaded.

    Args:
        df_main: Main Pandas DataFrame.
        df_annotated: Optional annotated Pandas DataFrame.
        output_dir: Directory to save/load the transformation model (data_transformer.joblib).
        target_column: Name of the target column in df_annotated (if provided).
        args: Arguments from the script, used for drop_columns_file.

    Returns:
        Tuple: (
            processed_main (np.ndarray),
            processed_annotated (np.ndarray or None),
            preprocessor (ColumnTransformer)
        )
        Raises ValueError if preprocessing fails critical checks.
    """
    if not output_dir:
        raise ValueError("output_dir must be provided to preprocess_mixed_data.")
    # Create model directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    transformer_path = os.path.join(output_dir, 'data_transformer.joblib')
    preprocessor = None

    # --- 0. Drop columns based on drop_columns_file ---
    if args and hasattr(args, 'drop_columns_file') and args.drop_columns_file:
        if os.path.exists(args.drop_columns_file):
            try:
                df_drop_cols = pd.read_csv(args.drop_columns_file)
                if 'col' in df_drop_cols.columns:
                    cols_to_drop = df_drop_cols['col'].tolist()
                    
                    original_main_cols = df_main.columns.tolist()
                    df_main = df_main.drop(columns=cols_to_drop, errors='ignore')
                    dropped_main_cols = list(set(original_main_cols) - set(df_main.columns.tolist()))
                    if dropped_main_cols:
                        print(f"Dropped columns from main data: {dropped_main_cols}")

                    if df_annotated is not None:
                        original_annotated_cols = df_annotated.columns.tolist()
                        df_annotated = df_annotated.drop(columns=cols_to_drop, errors='ignore')
                        dropped_annotated_cols = list(set(original_annotated_cols) - set(df_annotated.columns.tolist()))
                        if dropped_annotated_cols:
                            print(f"Dropped columns from annotated data: {dropped_annotated_cols}")
                else:
                    print(f"Warning: Column 'col' not found in {args.drop_columns_file}. No columns dropped.")
            except Exception as e:
                print(f"Error reading or processing {args.drop_columns_file}: {e}. No columns dropped.")
        else:
            print(f"Warning: drop_columns_file '{args.drop_columns_file}' not found. No columns dropped.")
    # --- End Drop columns ---

    # Assume the annotated dataframe has the same relevant columns for transformation
    categorical_cols = df_main.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df_main.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Target column should not be part of numerical/categorical cols used for fitting
    if target_column:
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)

    processed_main = None
    processed_annotated = None

    try:
        # --- Load or Fit Transformer ---
        if joblib and os.path.exists(transformer_path):
            print(f"Loading transformer from {transformer_path}")
            preprocessor = joblib.load(transformer_path)

            # --- Transform Data using Loaded Transformer ---
            print("Transforming main data using loaded transformer...")
            processed_main = preprocessor.transform(df_main)
            print(f"Original main shape: {df_main.shape}")
            print(f"Processed main shape: {processed_main.shape}")
            ok, processed_main = _check_processed_data(processed_main, "Main Data (Loaded Transformer)")
            if not ok:
                raise ValueError("Preprocessing check failed for main data using loaded transformer.")

            if df_annotated is not None:
                print("Transforming annotated data using loaded transformer...")
                # Ensure annotated data only has columns the transformer expects
                # These are the columns used during fitting (numerical_cols + categorical_cols determined during fit)
                # We need to get these columns from the loaded preprocessor
                fit_cols = []
                for name, _, cols in preprocessor.transformers_:
                    if name != 'remainder': # Exclude remainder if it exists
                       fit_cols.extend(cols)

                # Select only the necessary columns from df_annotated
                df_annotated_for_transform = df_annotated[fit_cols]

                processed_annotated = preprocessor.transform(df_annotated_for_transform)
                print(f"Original annotated shape: {df_annotated.shape}")
                print(f"Processed annotated shape: {processed_annotated.shape}")
                ok, processed_annotated = _check_processed_data(processed_annotated, "Annotated Data (Loaded Transformer)")
                if not ok:
                    raise ValueError("Preprocessing check failed for annotated data using loaded transformer.")

                # Check feature dimensions match (should match if loaded correctly)
                if processed_main.shape[1] != processed_annotated.shape[1]:
                     raise ValueError(f"Feature dimension mismatch after processing with loaded transformer: "
                                      f"Main data has {processed_main.shape[1]} features, "
                                      f"Annotated data has {processed_annotated.shape[1]} features.")

        else:
            print("Transformer not found. Preparing data for fitting new transformer...")

            # --- Prepare Data for Fitting --- #
            # Columns to use for fitting are determined ONLY by df_main's structure
            main_cols = df_main.columns.tolist()
            cols_for_fitting = [col for col in main_cols if col != target_column]

            # Create the dataframe to fit on
            df_fit = df_main[cols_for_fitting].copy()
            if df_annotated is not None:
                # Select only columns present in df_main from df_annotated
                annotated_cols_to_use = [col for col in cols_for_fitting if col in df_annotated.columns]
                if len(annotated_cols_to_use) != len(cols_for_fitting):
                    print("Warning: Not all columns from df_main used for fitting were found in df_annotated. Using available subset.")
                    # Find missing columns for clarity
                    missing_cols = list(set(cols_for_fitting) - set(annotated_cols_to_use))
                    print(f"  Missing columns in annotated data: {missing_cols}")

                # Concatenate using only the common columns found
                df_annotated_subset = df_annotated[annotated_cols_to_use].copy()
                df_fit = pd.concat([df_fit[annotated_cols_to_use], df_annotated_subset], ignore_index=True)
                print(f"Concatenated main and annotated data (columns from main) for fitting. Fit data shape: {df_fit.shape}")
            else:
                print(f"Fitting transformer ONLY on main data. Fit data shape: {df_fit.shape}")

            # --- Identify Column Types from the Combined Fitting Data --- #
            numerical_cols_fit = df_fit.select_dtypes(include=np.number).columns.tolist()
            categorical_cols_fit = df_fit.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"Identified {len(numerical_cols_fit)} numerical and {len(categorical_cols_fit)} categorical columns from fitting data.")

            # --- Create and Fit Preprocessor --- #
            # Create preprocessing pipelines
            numerical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')), # Use median for numeric
                ('scaler', MinMaxScaler(feature_range=(-1, 1)))
            ])
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')), # Use most_frequent for categoric
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            # Combine pipelines using types identified from df_fit
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_cols_fit),
                    ('cat', categorical_pipeline, categorical_cols_fit)
                ],
                remainder='drop' # Drop columns not specified
            )
            # Fit on the combined data (or just main data if no annotated)
            print("Fitting new transformer...")
            preprocessor.fit(df_fit)
            print("Transformer fitted.")

            # Clean up large intermediate dataframe
            del df_fit
            gc.collect()

            # Save the transformer
            if joblib:
                print(f"Saving transformer to {transformer_path}")
                joblib.dump(preprocessor, transformer_path)
            else:
                print("joblib not installed. Cannot save transformer.")

            # --- Transform Data using Newly Fitted Transformer --- #
            print("Transforming main data using newly fitted transformer...")
            processed_main = preprocessor.transform(df_main)
            print(f"Original main shape: {df_main.shape}")
            print(f"Processed main shape: {processed_main.shape}")
            ok, processed_main = _check_processed_data(processed_main, "Main Data (New Transformer)")
            if not ok:
                raise ValueError("Preprocessing check failed for main data after fitting.")

            if df_annotated is not None:
                print("Transforming annotated data using newly fitted transformer...")
                # Select columns for transform based on what was fitted
                fit_cols_new = numerical_cols_fit + categorical_cols_fit
                df_annotated_for_transform = df_annotated[fit_cols_new]

                processed_annotated = preprocessor.transform(df_annotated_for_transform)
                print(f"Original annotated shape: {df_annotated.shape}")
                print(f"Processed annotated shape: {processed_annotated.shape}")
                ok, processed_annotated = _check_processed_data(processed_annotated, "Annotated Data (New Transformer)")
                if not ok:
                    raise ValueError("Preprocessing check failed for annotated data after fitting.")

                # Check feature dimensions match
                if processed_main.shape[1] != processed_annotated.shape[1]:
                     raise ValueError(f"Feature dimension mismatch after processing: "
                                      f"Main data has {processed_main.shape[1]} features, "
                                      f"Annotated data has {processed_annotated.shape[1]} features. "
                                      "Ensure both datasets have compatible columns and the transformer is applied correctly.")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise # Re-raise the exception after logging

    return processed_main, processed_annotated, preprocessor


def load_person_data(args):
    """
    Loads main and annotated person data, preprocesses them using a single
    transformer (fitted on main data if needed), combines them, calculates
    a combined initial KNN graph, and returns data for SUBLIME training.

    Args:
        args: Arguments containing dataset paths, k, sparse flag, etc.

    Returns:
        Tuple: (
            features: Combined preprocessed features (Tensor).
            nfeats: Number of features.
            combined_labels: Combined labels (Tensor, -1 for main, 0/1 for annotated).
            nclasses_or_clusters: Target number for evaluation (based on args.downstream_task).
            train_mask, val_mask, test_mask: Dummy masks (Tensor).
            initial_graph: Combined initial graph structure (Tensor, dense or sparse).
        )
    """
    # === 1. Load Main Data ===
    print(f"Loading main dataset from {args.dataset}")
    try:
        df_main = pd.read_csv(args.dataset, delimiter='\t')
        n_main = len(df_main)
    except FileNotFoundError:
        print(f"Error: Main dataset file not found at {args.dataset}")
        raise
    except Exception as e:
        print(f"Error loading main dataset: {e}")
        raise

    # === 2. Load Annotated Data (if specified) ===
    df_annotated = None
    extracted_labels_all = None
    n_annotated = 0
    target_column = args.annotation_column if hasattr(args, 'annotation_column') else None

    if args.annotated_dataset:
        print(f"Loading annotated dataset from {args.annotated_dataset}")
        try:
            df_annotated = pd.read_csv(args.annotated_dataset, delimiter='\t')
            n_annotated = len(df_annotated)

            # Check for and extract target column
            if not target_column:
                raise ValueError(f"Annotated dataset '{args.annotated_dataset}' loaded, but 'annotation_column' not specified.")
            if target_column not in df_annotated.columns:
                 raise ValueError(f"Annotation column '{target_column}' not found in annotated dataset '{args.annotated_dataset}'. "
                                  f"Available columns: {df_annotated.columns.tolist()}")

            print(f"Extracting labels from target column '{target_column}'...")
            try:
                # Get all label values (-1, 0, 1)
                label_values_all = df_annotated[target_column].values
                unique_labels = np.unique(label_values_all)
                print(f"  Found unique label values in target column: {unique_labels}")

                # Check if at least 0 and 1 are present for classification task splitting
                if not np.any(np.isin(label_values_all, [0, 1])):
                    print(f"Warning: No valid labels (0 or 1) found in target column '{target_column}'. Masks will be empty.")
                    # Store all labels including -1 for combined_labels tensor
                    extracted_labels_all = torch.tensor(label_values_all, dtype=torch.long)
                else:
                    # Store all labels including -1 for combined_labels tensor
                    extracted_labels_all = torch.tensor(label_values_all, dtype=torch.long)

                # Keep the original n_annotated (count before filtering)
                # n_annotated is already set correctly before this block

                # Drop the target column from the original df_annotated
                df_annotated = df_annotated.drop(columns=[target_column])

            except Exception as e:
                print(f"Error processing target column '{target_column}': {e}")
                raise

        except FileNotFoundError:
            print(f"Error: Annotated dataset file not found at {args.annotated_dataset}")
            raise
        except Exception as e:
            print(f"Error loading or processing annotated dataset: {e}")
            raise
    else:
        print("No annotated dataset specified.")

    # === 2.5 Load Relationship Data (if specified) ===
    df_relationships = None
    if hasattr(args, 'relationship_dataset') and args.relationship_dataset:
        print(f"Loading relationship dataset from {args.relationship_dataset}")
        try:
            df_relationships = pd.read_csv(args.relationship_dataset, sep='\t') # Assuming default delimiter is comma, adjust if needed
            # Minimal validation: Check if required columns exist
            required_rel_cols = ['CPF', 'CPF_VINCULO'] # Assuming these are the linking columns
            # Use find_column_case_insensitive
            actual_cpf_rel_col1 = find_column_case_insensitive(df_relationships.columns, required_rel_cols[0].lower())
            actual_cpf_rel_col2 = find_column_case_insensitive(df_relationships.columns, required_rel_cols[1].lower())
            if not actual_cpf_rel_col1 or not actual_cpf_rel_col2:
                raise ValueError(f"Relationship dataset missing required columns '{required_rel_cols}' (case-insensitive). Found columns: {df_relationships.columns.tolist()}")
            print(f"Found relationship columns: '{actual_cpf_rel_col1}', '{actual_cpf_rel_col2}'")
        except FileNotFoundError:
            print(f"Error: Relationship dataset file not found at {args.relationship_dataset}")
            raise
        except Exception as e:
            print(f"Error loading or validating relationship dataset: {e}")
            # Optionally decide whether to raise or just continue without relationships
            raise
    else:
        print("No relationship dataset specified.")

    # === 3. Preprocess Data ===
    # Pass both dataframes, preprocessor handles fitting/loading and transforming
    processed_main, processed_annotated, preprocessor = preprocess_mixed_data(
        df_main, df_annotated, output_dir=args.output_dir, target_column=target_column, args=args
    )

    # === 4. Combine Features ===
    if processed_annotated is not None:
        print("Combining processed main and annotated features...")
        combined_features_np = np.vstack((processed_main, processed_annotated))
    else:
        print("Using only main features.")
        combined_features_np = processed_main

    n_total_samples = combined_features_np.shape[0]
    nfeats = combined_features_np.shape[1]
    print(f"Combined features shape: ({n_total_samples}, {nfeats})")
    features = torch.FloatTensor(combined_features_np).to(device)

    # === 4.5 Create CPF to Index Mapping ===
    print("Creating CPF to index mapping...")
    # IMPORTANT ASSUMPTION: The 'CPF' column exists in the original df_main and df_annotated
    # and their concatenation order matches combined_features_np. Need case-insensitive find.
    cpf_col_target_name = 'CPF'
    actual_cpf_col_main = find_column_case_insensitive(df_main.columns, cpf_col_target_name.lower())
    if actual_cpf_col_main is None:
        raise ValueError(f"Main dataset missing required identifier column '{cpf_col_target_name}' (case-insensitive).")

    all_cpfs_list = [df_main[actual_cpf_col_main]]
    # Store coordinates along with CPFs
    all_lat_list = []
    all_lon_list = []

    actual_lat_col_main = find_column_case_insensitive(df_main.columns, args.latitude_col.lower())
    actual_lon_col_main = find_column_case_insensitive(df_main.columns, args.longitude_col.lower())

    if args.use_geo_graph:
        if actual_lat_col_main is None:
            raise ValueError(f"Main dataset missing latitude column '{args.latitude_col}' for geo graph.")
        if actual_lon_col_main is None:
            raise ValueError(f"Main dataset missing longitude column '{args.longitude_col}' for geo graph.")
        all_lat_list.append(df_main[actual_lat_col_main])
        all_lon_list.append(df_main[actual_lon_col_main])

    # Reload annotated df briefly JUST to get CPF column if needed, then discard
    # df_annotated_temp_for_cpf = None # No longer needed
    actual_cpf_col_annotated = None
    if args.annotated_dataset and df_annotated is not None: # Check if df_annotated exists
        actual_cpf_col_annotated = find_column_case_insensitive(df_annotated.columns, cpf_col_target_name.lower())
        if actual_cpf_col_annotated is None:
                raise ValueError(f"Annotated dataset missing required identifier column '{cpf_col_target_name}' (case-insensitive) after potential modifications.")
        all_cpfs_list.append(df_annotated[actual_cpf_col_annotated]) # Append from existing df

        if args.use_geo_graph:
            actual_lat_col_annotated = find_column_case_insensitive(df_annotated.columns, args.latitude_col.lower())
            actual_lon_col_annotated = find_column_case_insensitive(df_annotated.columns, args.longitude_col.lower())
            if actual_lat_col_annotated is None:
                raise ValueError(f"Annotated dataset missing latitude column '{args.latitude_col}' for geo graph.")
            if actual_lon_col_annotated is None:
                raise ValueError(f"Annotated dataset missing longitude column '{args.longitude_col}' for geo graph.")
            all_lat_list.append(df_annotated[actual_lat_col_annotated])
            all_lon_list.append(df_annotated[actual_lon_col_annotated])


    combined_cpfs = pd.concat(all_cpfs_list, ignore_index=True)
    if len(combined_cpfs) != n_total_samples:
         raise ValueError(f"Length of combined CPFs ({len(combined_cpfs)}) does not match total samples ({n_total_samples}).")

    cpf_to_index = {cpf: idx for idx, cpf in enumerate(combined_cpfs)}
    print(f"Created CPF to index mapping for {len(cpf_to_index)} unique CPFs / {n_total_samples} total samples.")

    # Combine coordinates if geo graph is used
    combined_lat = None
    combined_lon = None
    if args.use_geo_graph:
        combined_lat = pd.concat(all_lat_list, ignore_index=True).to_numpy()
        combined_lon = pd.concat(all_lon_list, ignore_index=True).to_numpy()
        if len(combined_lat) != n_total_samples or len(combined_lon) != n_total_samples:
            raise ValueError("Length of combined lat/lon does not match total samples.")
        print(f"Combined latitude and longitude for {n_total_samples} samples.")

    # Delete combined CPF series and original dataframes no longer needed
    del combined_cpfs, df_main # df_main is no longer needed after preprocessing and CPF extraction
    if 'df_annotated' in locals(): del df_annotated # Delete the potentially modified annotated df
    gc.collect()
    print(f"Original dataframes and combined CPFs deleted. Memory usage: {get_memory_usage():.2f} GB")

    # === 5. Create Train/Val Split (for annotated data) and Masks ===
    train_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device)
    val_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device)
    test_mask = torch.zeros(n_total_samples, dtype=torch.bool, device=device) # Keep test mask as all False for now

    if extracted_labels_all is not None and n_annotated > 0:
        # Identify indices within the annotated data that have valid labels (0 or 1)
        annotated_indices_relative = np.arange(n_annotated)
        valid_labels_mask_annotated = np.isin(extracted_labels_all.cpu().numpy(), [0, 1])
        valid_annotated_indices_relative = annotated_indices_relative[valid_labels_mask_annotated]

        if len(valid_annotated_indices_relative) > 0:
            # Get the actual labels (0 or 1) for stratification
            labels_for_split = extracted_labels_all[valid_labels_mask_annotated].cpu().numpy()

            print(f"Splitting {len(valid_annotated_indices_relative)} annotated samples with valid labels (0 or 1)...")
            # Split only the valid relative indices
            train_valid_indices_relative, val_valid_indices_relative = train_test_split(
                valid_annotated_indices_relative,
                test_size=0.20,
                random_state=42,
                stratify=labels_for_split
            )

            print(f"  Train split size: {len(train_valid_indices_relative)}, Validation split size: {len(val_valid_indices_relative)}")

            # Calculate the global indices in the combined dataset for these splits
            # Annotated data starts after main data (index n_main)
            global_train_indices = train_valid_indices_relative + n_main
            global_val_indices = val_valid_indices_relative + n_main

            # Populate the masks based on the global indices of valid samples
            train_mask[global_train_indices] = True
            val_mask[global_val_indices] = True

            print(f"Populated train_mask ({train_mask.sum().item()} True) and val_mask ({val_mask.sum().item()} True) using only labels 0/1.")
        else:
            print("No annotated samples with valid labels (0 or 1) found to create train/val split.")
    else:
        print("No annotated data provided or extracted_labels_all is None. train_mask and val_mask remain False.")

    # === 5b. Combine Labels (Full labels needed for indexing in main.py) ===
    if extracted_labels_all is not None:
        print("Creating combined labels tensor (-1 for main, 0/1 for annotated)...")
        main_placeholders = torch.full((n_main,), -1, dtype=torch.long)
        combined_labels = torch.cat((main_placeholders, extracted_labels_all)).to(device)
        # Verify shape
        if len(combined_labels) != n_total_samples:
             raise ValueError(f"Combined labels length ({len(combined_labels)}) does not match total samples ({n_total_samples}).")
        print(f"Combined labels shape: {combined_labels.shape}")
    else:
        # If no annotated data, create labels tensor of -1s or handle as needed
        # For consistency, we'll create a tensor of -1s. Loss function needs to handle this.
        print("No annotated data provided, creating labels tensor with -1 placeholders.")
        combined_labels = torch.full((n_total_samples,), -1, dtype=torch.long).to(device)

    # === 6. Determine nclasses/n_clusters for evaluation ===
    if args.downstream_task == 'clustering':
        nclasses_or_clusters = args.n_clusters if hasattr(args, 'n_clusters') else n_total_samples # Default to nodes if not specified
        print(f"Downstream task is clustering. Using n_clusters = {nclasses_or_clusters} for evaluation.")
    elif extracted_labels_all is not None:
        # If classification task and we have binary labels, nclasses for evaluation is 2
        nclasses_or_clusters = 2
        print(f"Downstream task is classification. Using n_classes = {nclasses_or_clusters} for evaluation.")
    else:
        # Fallback if classification task but no labels somehow (shouldn't happen with checks)
        # Or if another task is added later.
        nclasses_or_clusters = n_total_samples # Default to number of nodes
        print(f"Warning: Downstream task '{args.downstream_task}' but labels are missing or not binary. Using n_nodes = {nclasses_or_clusters} for evaluation.")


    # === 7. Determine Features for Initial KNN Graph ===
    # (Optional: Logic to exclude columns for KNN based on args.drop_columns_file)
    # For simplicity now, using *all* combined features for KNN.
    # If column dropping is needed, apply it to combined_features_np *before* KNN.
    features_for_knn = combined_features_np # Using combined features
    print(f"Using combined features (shape: {features_for_knn.shape}) for initial KNN graph construction.")
    # Add variance check for KNN features
    ok, _ = _check_processed_data(features_for_knn, "KNN Features")
    if not ok:
        raise ValueError("Preprocessing check failed for features used in KNN.")


    # === 8. Calculate Combined Initial Graph Structure ===
    initial_graph = None # Initialize final graph variable

    # --- 8a. KNN Graph (Feature Similarity) ---
    print(f"\nConstructing KNN graph (k={args.k}, cosine, threshold={args.knn_threshold_type}) on processed features...")
    adj_knn_sparse = None
    try:
        if features_for_knn.shape[1] == 0:
            raise ValueError("Cannot compute KNN graph with zero features.")

        # Convert numpy features to tensor first
        features_knn_tensor = torch.tensor(features_for_knn, dtype=torch.float32).to(device)
        # Request values from knn_fast
        knn_rows, knn_cols, knn_vals = knn_fast(
            features_knn_tensor,
            k=args.k,
            use_gpu=(device.type == 'cuda'),
            knn_threshold_type=args.knn_threshold_type,
            knn_std_dev_factor=args.knn_std_dev_factor,
            return_values=True # <-- Request values
        )
        # Create sparse scipy matrix from the COO tensors (move to CPU)
        n = features_knn_tensor.shape[0]
        # Use knn_vals for the data
        adj_knn_sparse = sp.csr_matrix((knn_vals.cpu().numpy(), (knn_rows.cpu().numpy(), knn_cols.cpu().numpy())),
                                      shape=(n, n))
        # --- Remove binary conversion --- 
        # adj_knn_sparse = adj_knn_sparse.astype(bool).astype(np.float32)
        
        # Symmetrize using maximum to preserve weights
        adj_knn_sparse = adj_knn_sparse.maximum(adj_knn_sparse.T)
        # --- Remove binary conversion after symmetrization --- 
        # adj_knn_sparse = adj_knn_sparse.astype(bool).astype(np.float32)

        print(f"  KNN graph computed using knn_fast. Shape: {adj_knn_sparse.shape}, Non-zero entries: {adj_knn_sparse.nnz}")
    except Exception as e:
        print(f"Error during KNN graph construction with knn_fast: {e}")
        adj_knn_sparse = sp.eye(n_total_samples, dtype=np.float32, format='csr') # Fallback
        print(f"  Using identity matrix for KNN graph. Shape: {adj_knn_sparse.shape}, nnz: {adj_knn_sparse.nnz}")

    # Delete tensor used only for KNN computation
    if 'features_knn_tensor' in locals():
        del features_knn_tensor
        gc.collect()
        print(f"  KNN features tensor deleted. Memory: {get_memory_usage():.2f} GB")

    # Delete features_for_knn as it's now embedded in adj_knn_sparse
    del features_for_knn

    # --- 8b. Relationship Graph (Explicit Links) ---
    adj_rel_sparse = None
    if df_relationships is not None:
        print("Constructing relationship graph...")
        rows, cols, data = [], [], []
        valid_edges = 0
        skipped_edges = 0
        skipped_self_loops = 0
        missing_cpf1 = 0
        missing_cpf2 = 0
        missing_cpf1_values = set() # Store missing CPF1 values
        missing_cpf2_values = set() # Store missing CPF2 values

        # Use actual column names found earlier
        actual_cpf_rel_col1_used = actual_cpf_rel_col1
        actual_cpf_rel_col2_used = actual_cpf_rel_col2

        for _, row in df_relationships.iterrows():
            cpf1 = row[actual_cpf_rel_col1_used]
            cpf2 = row[actual_cpf_rel_col2_used]

            idx1 = cpf_to_index.get(cpf1)
            idx2 = cpf_to_index.get(cpf2)

            if idx1 is not None and idx2 is not None:
                if idx1 != idx2:
                    rows.extend([idx1, idx2]) # Add edge in both directions for symmetry
                    cols.extend([idx2, idx1])
                    # Use the weight from args, default 1.0
                    rel_weight = args.relationship_weight if hasattr(args, 'relationship_weight') else 1.0
                    data.extend([rel_weight, rel_weight])
                    valid_edges += 1
                else:
                    skipped_self_loops += 1
            else:
                skipped_edges += 1
                if idx1 is None:
                    missing_cpf1 += 1
                    missing_cpf1_values.add(cpf1) # Add missing CPF1
                if idx2 is None:
                    missing_cpf2 += 1
                    missing_cpf2_values.add(cpf2) # Add missing CPF2

        if valid_edges > 0:
            adj_rel_sparse = sp.csr_matrix((data, (rows, cols)), shape=(n_total_samples, n_total_samples), dtype=np.float32)
            adj_rel_sparse.sum_duplicates() # Remove duplicates
            print(f"  Relationship graph computed. Shape: {adj_rel_sparse.shape}, Non-zero entries: {adj_rel_sparse.nnz}")
            print(f"  (Found {valid_edges} valid relationship pairs. Skipped {skipped_edges} edges due to missing CPFs "
                  f"[{missing_cpf1} missing {actual_cpf_rel_col1_used}, {missing_cpf2} missing {actual_cpf_rel_col2_used}] and {skipped_self_loops} self-loops)")
            # Print the missing CPFs if any were found
            if missing_cpf1_values:
                print(f"    Missing CPFs from '{actual_cpf_rel_col1_used}' column: {sorted(list(missing_cpf1_values))}")
            if missing_cpf2_values:
                print(f"    Missing CPFs from '{actual_cpf_rel_col2_used}' column: {sorted(list(missing_cpf2_values))}")
        else:
            print(f"  No valid relationship edges created. Total skipped edges: {skipped_edges}, self-loops: {skipped_self_loops}.")
            # Print the missing CPFs even if no valid edges were created
            if missing_cpf1_values:
                print(f"    Missing CPFs from '{actual_cpf_rel_col1_used}' column: {sorted(list(missing_cpf1_values))}")
            if missing_cpf2_values:
                print(f"    Missing CPFs from '{actual_cpf_rel_col2_used}' column: {sorted(list(missing_cpf2_values))}")
            adj_rel_sparse = sp.csr_matrix((n_total_samples, n_total_samples), dtype=np.float32) # Empty sparse matrix

        # Delete relationship dataframe after use
        del df_relationships
        gc.collect()
        print(f"Relationship dataframe deleted. Memory usage: {get_memory_usage():.2f} GB")

    # --- 8c. Geographical Proximity Graph ---
    adj_geo_sparse = None
    if args.use_geo_graph:
        # K-nearest neighbors geographical graph with normalization
        print(f"Constructing geographical k-NN graph (k={args.geo_k}, max_weight={args.geo_max_weight}, min_weight={args.geo_min_weight})...")
        if combined_lat is None or combined_lon is None:
            print("Warning: Latitude or Longitude data not available. Skipping geographical graph.")
        else:
            # Filter out samples with invalid coordinates
            valid_coord_mask = ~(pd.isna(combined_lat) | pd.isna(combined_lon))
            valid_indices = np.where(valid_coord_mask)[0]
            n_valid = len(valid_indices)
            
            if n_valid < 2:
                print(f"Warning: Not enough valid geographical coordinates ({n_valid} < 2). Skipping geographical graph.")
                adj_geo_sparse = sp.csr_matrix((n_total_samples, n_total_samples), dtype=np.float32)
            else:
                print(f"  Found {n_valid} samples with valid coordinates out of {n_total_samples} total samples.")
                
                # Extract valid coordinates
                valid_lat = combined_lat[valid_indices]
                valid_lon = combined_lon[valid_indices]
                
                # Efficient batch distance computation
                # Pre-compute all pairwise distances for valid samples
                print("  Computing pairwise geographical distances...")
                
                # For memory efficiency with large datasets, process in chunks
                chunk_size = min(1000, n_valid)  # Process up to 1000 samples at a time
                geo_rows, geo_cols, geo_data = [], [], []
                
                for i_start in range(0, n_valid, chunk_size):
                    i_end = min(i_start + chunk_size, n_valid)
                    chunk_indices = range(i_start, i_end)
                    
                    # Compute distances from this chunk to all valid samples
                    distances_chunk = np.zeros((len(chunk_indices), n_valid))
                    
                    for i_local, i_global in enumerate(chunk_indices):
                        # Vectorized distance computation
                        distances_chunk[i_local, :] = haversine(
                            valid_lat[i_global], valid_lon[i_global],
                            valid_lat, valid_lon
                        )
                    
                    # Process each node in the chunk
                    for i_local, i_global in enumerate(chunk_indices):
                        # Get distances for this node
                        distances = distances_chunk[i_local, :]
                        
                        # Set distance to self as infinity to exclude it
                        distances[i_global] = np.inf
                        
                        # Find k nearest neighbors
                        k_actual = min(args.geo_k, n_valid - 1)  # Can't have more neighbors than available
                        nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
                        nearest_distances = distances[nearest_indices]
                        
                        # Sort the k neighbors by distance
                        sorted_idx = np.argsort(nearest_distances)
                        nearest_indices = nearest_indices[sorted_idx]
                        nearest_distances = nearest_distances[sorted_idx]
                        
                        # Normalize weights: closest gets geo_max_weight, furthest gets geo_min_weight
                        if len(nearest_indices) > 0:
                            min_dist = nearest_distances[0]
                            max_dist = nearest_distances[-1]
                            
                            # Map back to original indices
                            original_i = valid_indices[i_global]
                            
                            for j_local, dist in zip(nearest_indices, nearest_distances):
                                original_j = valid_indices[j_local]
                                
                                # Calculate normalized weight
                                if max_dist > min_dist and len(nearest_indices) > 1:
                                    # Linear interpolation from max_weight to min_weight
                                    weight_ratio = (dist - min_dist) / (max_dist - min_dist)
                                    weight = args.geo_max_weight - weight_ratio * (args.geo_max_weight - args.geo_min_weight)
                                else:
                                    # All neighbors at same distance or only one neighbor
                                    weight = args.geo_max_weight
                                
                                # Add symmetric edges
                                geo_rows.extend([original_i, original_j])
                                geo_cols.extend([original_j, original_i])
                                geo_data.extend([weight, weight])
                    
                    if i_end % 5000 == 0 or i_end == n_valid:
                        print(f"    Processed {i_end}/{n_valid} nodes...")
                
                # Create sparse matrix
                if len(geo_rows) > 0:
                    adj_geo_sparse = sp.csr_matrix((geo_data, (geo_rows, geo_cols)),
                                                   shape=(n_total_samples, n_total_samples), dtype=np.float32)
                    adj_geo_sparse.sum_duplicates()  # Remove duplicate edges
                    
                    # Calculate statistics
                    degrees = np.array((adj_geo_sparse > 0).sum(axis=1)).flatten()
                    valid_degrees = degrees[valid_indices]
                    
                    print(f"  Geographical k-NN graph computed. Shape: {adj_geo_sparse.shape}, Non-zero entries: {adj_geo_sparse.nnz}")
                    print(f"  Average degree (for nodes with coordinates): {np.mean(valid_degrees):.2f}")
                    print(f"  Min/Max degree: {np.min(valid_degrees)}/{np.max(valid_degrees)}")
                    print(f"  Edge weight range: [{np.min(geo_data):.4f}, {np.max(geo_data):.4f}]")
                else:
                    print(f"  No geographical edges created.")
                    adj_geo_sparse = sp.csr_matrix((n_total_samples, n_total_samples), dtype=np.float32)
        
        # Clean up combined coordinates
        del combined_lat, combined_lon
        gc.collect()

    # --- 8d. Combine Graphs ---
    print("Combining KNN, Relationship, and Geographical graphs...")
    
    # Start with KNN graph
    current_combined_graph = adj_knn_sparse
    
    # Combine with Relationship graph if it exists and has edges
    if adj_rel_sparse is not None and adj_rel_sparse.nnz > 0:
        current_combined_graph = current_combined_graph.maximum(adj_rel_sparse)
        print(f"  Combined with Relationship graph. Non-zero entries: {current_combined_graph.nnz}")
    else:
        print("  No valid Relationship graph to combine or it was not specified.")
        
    # Combine with Geographical graph if it exists and has edges
    if adj_geo_sparse is not None and adj_geo_sparse.nnz > 0:
        current_combined_graph = current_combined_graph.maximum(adj_geo_sparse)
        print(f"  Combined with Geographical graph. Non-zero entries: {current_combined_graph.nnz}")
    else:
        print("  No valid Geographical graph to combine or it was not specified.")

    initial_graph_sparse = current_combined_graph
    print(f"  Total non-zero entries after all combinations (before self-loops): {initial_graph_sparse.nnz}")


    # Add self-loops with weight 1.0 using maximum
    print("Adding self-loops (weight 1.0)...")
    initial_graph_sparse = initial_graph_sparse.maximum(sp.eye(initial_graph_sparse.shape[0], dtype=np.float32, format='csr'))
    # --- Remove final binary conversion --- 
    # initial_graph_sparse = initial_graph_sparse.astype(bool).astype(np.float32)
    print(f"  Combined graph + self-loops non-zero entries: {initial_graph_sparse.nnz}")

    # --- 8d. Convert Final Sparse Graph to Tensor ---
    # This final `initial_graph_sparse` is what will be passed to the training loop
    print("Converting final graph structure to required format...")
    try:
        if args.sparse:
            # Convert the final scipy sparse matrix to torch sparse tensor
            initial_graph = sparse_mx_to_torch_sparse_tensor(initial_graph_sparse).to(device)
            print(f"Initial combined graph prepared as sparse torch tensor on {device}.")
        else:
            # Convert final scipy sparse matrix to dense torch tensor
            initial_graph = torch.FloatTensor(initial_graph_sparse.todense()).to(device)
            print(f"Initial combined graph prepared as dense torch tensor on {device}.")

    except Exception as e:
         print(f"Error during final sparse/dense conversion: {e}")
         print("Falling back to identity matrix.")
         # Create fallback directly as torch tensor on the correct device
         if args.sparse:
              eye_indices = torch.arange(n_total_samples, device=device).unsqueeze(0).repeat(2, 1)
              eye_values = torch.ones(n_total_samples, device=device)
              initial_graph = torch.sparse_coo_tensor(eye_indices, eye_values, (n_total_samples, n_total_samples))
         else:
              initial_graph = torch.eye(n_total_samples, dtype=torch.float32, device=device)
         print(f"Using identity matrix ({ 'sparse' if args.sparse else 'dense'}) on {device}.")

    del adj_knn_sparse # No longer needed after combination
    if adj_rel_sparse is not None:
        del adj_rel_sparse # No longer needed after combination
    if adj_geo_sparse is not None: # Delete geo graph if it was created
        del adj_geo_sparse
    del initial_graph_sparse # No longer needed after conversion to tensor
    gc.collect()
    print(f"Intermediate sparse matrices deleted. Memory: {get_memory_usage():.2f} GB")
    
    print(f"Prepared combined dataset. Features shape: {features.shape}. Initial graph type: {'Sparse' if args.sparse else 'Dense'}. Labels shape: {combined_labels.shape}.")

    # Return combined data
    return features, nfeats, combined_labels, nclasses_or_clusters, train_mask, val_mask, test_mask, initial_graph

def main():
    # Get the base parser from main.py
    parent_parser = create_parser()

    # Create a new parser that inherits from the parent, but don't add help to avoid conflicts
    # Use argument_default=argparse.SUPPRESS to allow defaults from parent to take precedence
    # unless explicitly overridden here or by command line.
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False, argument_default=argparse.SUPPRESS)

    # Set defaults specific to this script *if they need to override the parent*
    parser.set_defaults(
        dataset='person_data.csv', # Keep or change as needed
        # annotated_dataset=None, # Let parent handle this default or set via command line
        # annotation_column=None, # Let parent handle this default or set via command line
        ntrials=1,
        sparse=0,
        gsl_mode="structure_inference", # Changed default: inference makes more sense if we always compute KNN
        eval_freq=50,               # Example: adjusted eval freq
        downstream_task='clustering', # Default to clustering for this script
        epochs=50,                 # Example: Reduced epochs for person data
        save_model=1,
        output_dir='sublime_models/',
        checkpoint_dir='sublime_checkpoints/', # Give specific checkpoint dir
        checkpoint_freq=25,          # Example: checkpoint freq
        verbose=1,
        k=10,                        # Example: adjusted k
        # n_clusters=5                 # Let parent handle default or set via command line
        # type_learner defaults to 'fgp' from parent
        # lr, hidden_dim etc. default from parent
        # Defaults for geo graph
        use_geo_graph=0,
        latitude_col='LATITUDE',
        longitude_col='LONGITUDE',
        # Geo graph k-NN defaults
        geo_k=10,
        geo_max_weight=1.0,
        geo_min_weight=0.1
    )

    # Add arguments specific to this script OR override parent defaults forcefully
    parser.add_argument('-n_clusters', type=int, help='Number of clusters for clustering task') # Keep help text


    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --- Configuration Validation ---
    if args.annotated_dataset and not args.annotation_column:
        parser.error("If --annotated_dataset is provided, --annotation_column must also be specified.")
    if not args.annotated_dataset and args.annotation_column:
        print("Warning: --annotation_column specified but --annotated_dataset is missing. The annotation column will be ignored.")
        args.annotation_column = None # Ensure it's None if no annotated dataset
    
    # --- Geo Graph Validation ---
    if args.use_geo_graph:
        if not hasattr(args, 'latitude_col') or not args.latitude_col:
            parser.error("--use_geo_graph requires --latitude_col to be specified.")
        if not hasattr(args, 'longitude_col') or not args.longitude_col:
            parser.error("--use_geo_graph requires --longitude_col to be specified.")
        
        # K-NN mode validation
        if not hasattr(args, 'geo_k') or args.geo_k <= 0:
            parser.error("--use_geo_graph requires --geo_k to be a positive integer.")
        if not hasattr(args, 'geo_max_weight') or args.geo_max_weight <= 0:
            parser.error("--use_geo_graph requires --geo_max_weight to be positive.")
        if not hasattr(args, 'geo_min_weight') or args.geo_min_weight < 0:
            parser.error("--use_geo_graph requires --geo_min_weight to be non-negative.")
        if args.geo_min_weight >= args.geo_max_weight:
            parser.error("--geo_min_weight must be less than --geo_max_weight.")

    # --- End Validation ---


    # Print final config being used
    print("--- Running Training with Configuration ---")
    for arg, value in sorted(vars(args).items()):
         print(f"{arg}: {value}")
    print("-----------------------------------------")


    # Create experiment and train model
    experiment = Experiment(device)
    # Pass the specific load_person_data function
    experiment.train(args, load_data_fn=load_person_data)

if __name__ == "__main__":
    main() 

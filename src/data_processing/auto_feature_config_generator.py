# -*- coding: utf-8 -*-
"""
Auto Feature Config Generator for EasyRec.

This script infers schema from a sample Pandas DataFrame (simulating data that
would typically be read from a Parquet file on GCS), classifies its columns
into feature types (e.g., categorical, numerical, ID, label), and then
generates protobuf text format snippets for EasyRec's `input_fields` and
`feature_config.features`.

The primary goal is to automate the initial setup of feature configurations
for EasyRec, reducing manual effort and potential errors.

Key functionalities:
1.  Generation of a sample DataFrame with diverse data types.
2.  Classification of features based on column names and data characteristics.
3.  Generation of `input_fields` protobuf configuration.
4.  Generation of `feature_config.features` protobuf configuration.

This script does NOT perform actual GCS access; it uses a local DataFrame
to simulate the data input process.
"""

import pandas as pd
import numpy as np

def get_sample_dataframe():
    """Simulates reading a sample Parquet file from GCS and returns a Pandas DataFrame.

    The DataFrame includes a variety of column types (int, float, string)
    and specific columns like user_id, item_id, click, and conversion to
    mimic a real-world dataset.

    Returns:
        pandas.DataFrame: A sample DataFrame.
    """
    data = {
        'user_id': [f'user_{i}' for i in range(1000)],
        'item_id': [f'item_{i % 100}' for i in range(1000)],  # 100 unique items
        'feature_cat_1': np.random.choice(['A', 'B', 'C', 'D', 'E'], size=1000),
        'feature_num_1': np.random.rand(1000),
        'feature_int_treat_as_cat': np.random.randint(0, 10, size=1000), # Low cardinality int
        'feature_int_treat_as_num': np.random.randint(0, 10000, size=1000), # High cardinality int
        'click': np.random.randint(0, 2, size=1000),
        'conversion': np.random.randint(0, 2, size=1000),
        'feature_cat_2_mixed_type': [str(i) if i % 2 == 0 else np.nan for i in range(1000)], # Mixed type with NaNs
        'feature_float_with_nan': np.random.choice([1.0, 2.0, np.nan, 4.0, 5.0], size=1000) # Float with NaNs
    }
    df = pd.DataFrame(data)
    return df

def classify_features(df):
    """
    Classifies features in a DataFrame based on their names and data types.

    Heuristics:
    -   'user_id', 'item_id', 'click', 'conversion' are identified by name.
    -   `float` dtypes are 'numerical'.
    -   `object` (string) dtypes are 'categorical'.
    -   `int` dtypes are 'categorical' if nunique is low (nunique/total < 0.05 or nunique < 50),
        otherwise 'numerical'.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        dict: A dictionary mapping column names to their inferred feature type.
              Possible types: 'user_id', 'item_id', 'click_label',
                              'conversion_label', 'categorical', 'numerical', 'unknown'.
    """
    classified = {}
    sample_size = len(df)

    # Thresholds for classifying integer columns
    # If number of unique values / sample_size < unique_threshold_ratio OR
    # number of unique values < unique_threshold_abs, it's considered categorical.
    unique_threshold_ratio = 0.05
    unique_threshold_abs = 50

    for col_name in df.columns:
        dtype = df[col_name].dtype
        nunique = df[col_name].nunique(dropna=True) # dropna=True is important for mixed types with NaNs

        if col_name == 'user_id':
            classified[col_name] = 'user_id'
        elif col_name == 'item_id':
            classified[col_name] = 'item_id'
        elif col_name == 'click':
            classified[col_name] = 'click_label'
        elif col_name == 'conversion':
            classified[col_name] = 'conversion_label'
        elif dtype == 'float' or dtype == np.float64 or dtype == np.float32:
            classified[col_name] = 'numerical'
        elif dtype == 'object':
            # Assuming 'object' dtype columns are categorical strings.
            # More sophisticated checks could be added here (e.g., try to convert to numeric).
            classified[col_name] = 'categorical'
        elif dtype == 'int' or dtype == np.int64 or dtype == np.int32:
            # For integer types, decide if it's categorical or numerical based on cardinality
            if (nunique / sample_size < unique_threshold_ratio) or \
               (nunique < unique_threshold_abs):
                classified[col_name] = 'categorical'
            else:
                classified[col_name] = 'numerical'
        else:
            # Fallback for unhandled dtypes
            print(f"Warning: Column '{col_name}' has unhandled dtype '{dtype}'. Classified as 'unknown'.")
            classified[col_name] = 'unknown'

    return classified

def generate_input_fields_proto(classified_features):
    """
    Generates the EasyRec input_fields protobuf text format string.

    Maps inferred feature types to EasyRec input types:
    - 'user_id', 'item_id', 'categorical' -> STRING
    - 'numerical' -> DOUBLE
    - 'click_label', 'conversion_label' -> INT32 (with default_val: "0")

    Args:
        classified_features (dict): A dictionary mapping column names to their
                                    inferred feature type.

    Returns:
        str: A string representing the input_fields protobuf configuration.
    """
    proto_str_parts = []
    for name, f_type in classified_features.items():
        if f_type == 'unknown': # Skip unknown types
            print(f"Info: Skipping column '{name}' (type: {f_type}) in input_fields generation.")
            continue

        field_str = f"input_fields {{\n  input_name: \"{name}\"\n"
        if f_type in ['user_id', 'item_id', 'categorical']:
            # User IDs, Item IDs, and other categorical features are treated as STRING inputs.
            # Their specific roles (e.g., as user/item identifiers) are further defined
            # in the feature_config and model architecture.
            field_str += "  input_type: STRING\n"
        elif f_type == 'numerical':
            field_str += "  input_type: DOUBLE\n" # Using DOUBLE for precision, FLOAT is also an option.
        elif f_type in ['click_label', 'conversion_label']:
            field_str += "  input_type: INT32\n"
            # It's common practice to provide a default value for label fields.
            field_str += "  default_val: \"0\"\n"
        else:
            # This case should ideally not be reached if 'unknown' is handled
            print(f"Warning: Unhandled feature type '{f_type}' for column '{name}' in input_fields generation. Skipping.")
            continue
        field_str += "}"
        proto_str_parts.append(field_str)
    return "\n".join(proto_str_parts)

def generate_feature_config_proto(classified_features):
    """
    Generates the EasyRec feature_config.features protobuf text format string.

    Maps inferred feature types to EasyRec feature types:
    - 'user_id', 'item_id', 'categorical' -> IdFeature (with embedding_dim and hash_bucket_size)
    - 'numerical' -> RawFeature
    - Labels ('click_label', 'conversion_label') and 'unknown' types are ignored.

    Args:
        classified_features (dict): A dictionary mapping column names to their
                                    inferred feature type.

    Returns:
        str: A string representing the feature_config.features protobuf configuration.
    """
    proto_str_parts = []
    # Default parameters for feature configurations
    default_embedding_dim = 16
    default_hash_bucket_size = 100000  # For general categorical features
    user_id_hash_bucket_size = 1000000 # Larger bucket for user IDs
    item_id_hash_bucket_size = 500000  # Larger bucket for item IDs

    for name, f_type in classified_features.items():
        if f_type in ['click_label', 'conversion_label', 'unknown']:
            # Labels and unknown types are not included in feature_config.features
            continue

        feature_entry = f"features {{\n  input_names: \"{name}\"\n" # `input_names` is plural but usually takes one
        if f_type == 'user_id':
            feature_entry += "  feature_type: IdFeature\n"
            feature_entry += f"  embedding_dim: {default_embedding_dim}\n"
            feature_entry += f"  hash_bucket_size: {user_id_hash_bucket_size}\n"
        elif f_type == 'item_id':
            feature_entry += "  feature_type: IdFeature\n"
            feature_entry += f"  embedding_dim: {default_embedding_dim}\n"
            feature_entry += f"  hash_bucket_size: {item_id_hash_bucket_size}\n"
        elif f_type == 'categorical':
            feature_entry += "  feature_type: IdFeature\n"
            feature_entry += f"  embedding_dim: {default_embedding_dim}\n"
            feature_entry += f"  hash_bucket_size: {default_hash_bucket_size}\n"
        elif f_type == 'numerical':
            feature_entry += "  feature_type: RawFeature\n"
            # For RawFeature, no embedding_dim or hash_bucket_size is needed.
            # Options like boundaries for discretization could be added here in the future.
        else:
            # This case should ideally not be reached if 'unknown' and labels are handled
            print(f"Warning: Unhandled feature type '{f_type}' for column '{name}' in feature_config generation. Skipping.")
            continue
        feature_entry += "}"
        proto_str_parts.append(feature_entry)
    return "\n".join(proto_str_parts)

if __name__ == '__main__':
    # This main block demonstrates the script's functionality.
    # 1. Generate a sample DataFrame.
    # 2. Classify its features.
    # 3. Generate and print input_fields protobuf.
    # 4. Generate and print feature_config.features protobuf.

    print("Starting Auto Feature Config Generation Process...")

    # Step 1: Get Sample DataFrame
    sample_df = get_sample_dataframe()
    print("\nSample DataFrame (first 5 rows):")
    print(sample_df.head())
    # print("\nDataFrame Info (can be verbose):")
    # sample_df.info()

    # Step 2: Classify Features
    print("\nClassifying features...")
    feature_classification = classify_features(sample_df)
    print("Classification complete. Results:")
    for feature, f_type in feature_classification.items():
        print(f"  Column: {feature:<30} | Inferred Type: {f_type}")

    # Step 3: Generate input_fields protobuf
    print("\nGenerating input_fields protobuf snippet...")
    input_fields_proto = generate_input_fields_proto(feature_classification)
    print("-------------------- input_fields --------------------")
    print(input_fields_proto)
    print("------------------------------------------------------")

    # Step 4: Generate feature_config.features protobuf
    print("\nGenerating feature_config.features protobuf snippet...")
    feature_config_proto = generate_feature_config_proto(feature_classification)
    print("---------------- feature_config.features -------------")
    print(feature_config_proto)
    print("------------------------------------------------------")

    print("\nAuto Feature Config Generation Process Finished.")


# --- Functions for GCS Sample Loading (to be used by other scripts) ---

def load_sample_from_gcs_parquet(gcs_parquet_path_or_dir, num_files_to_sample=1, num_rows_per_file_sample=1000):
    """
    (Mock) Simulates loading a sample DataFrame from Parquet files in a GCS directory.

    In a real scenario, this would:
    1. List files in `gcs_parquet_path_or_dir` (if it's a directory).
    2. Select `num_files_to_sample` files.
    3. For each selected file, read `num_rows_per_file_sample` (e.g., using pyarrow's ParquetFile.read_row_group or similar).
    4. Concatenate the samples into a single Pandas DataFrame.
    5. Requires `gcsfs` and `pandas`/`pyarrow` and appropriate GCS authentication.

    Args:
        gcs_parquet_path_or_dir (str): GCS path to a single Parquet file or a directory
                                       containing Parquet files.
        num_files_to_sample (int): Number of Parquet files to sample from if a directory is given.
        num_rows_per_file_sample (int): Number of rows to sample from each selected file.

    Returns:
        pandas.DataFrame: A sample DataFrame. For this mock version, it returns
                          the same DataFrame as get_sample_dataframe().
    """
    print(f"[Mock GCS Read] Called load_sample_from_gcs_parquet:")
    print(f"  Path: {gcs_parquet_path_or_dir}")
    print(f"  Num files to sample: {num_files_to_sample}")
    print(f"  Num rows per file sample: {num_rows_per_file_sample}")
    print("  Note: This is a mock function. It will return a predefined sample DataFrame.")

    # ---  Example of how a real implementation might start (commented out) ---
    # import pandas as pd
    # import gcsfs # For accessing GCS
    # from google.cloud import storage # For listing files if it's a directory
    #
    # gcs = gcsfs.GCSFileSystem() # Assumes ADC or gcloud auth is set up
    #
    # all_sampled_dfs = []
    #
    # if gcs_parquet_path_or_dir.endswith(".parquet"):
    #     file_paths_to_process = [gcs_parquet_path_or_dir]
    # else: # Assume it's a directory
    #     print(f"  (Real GCS list would happen here for directory: {gcs_parquet_path_or_dir})")
    #     # bucket_name, prefix = gcs_parquet_path_or_dir.replace("gs://", "").split("/", 1)
    #     # storage_client = storage.Client()
    #     # blobs = list(storage_client.list_blobs(bucket_name, prefix=prefix.rstrip('/') + '/'))
    #     # parquet_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(".parquet")]
    #     # file_paths_to_process = np.random.choice(parquet_files, size=min(num_files_to_sample, len(parquet_files)), replace=False)
    #     file_paths_to_process = ["gs://mock_bucket/mock_path/train_data_part1.parquet", "gs://mock_bucket/mock_path/train_data_part2.parquet"] # Mocked list
    #     file_paths_to_process = np.random.choice(file_paths_to_process, size=min(num_files_to_sample, len(file_paths_to_process)), replace=False)


    # for gcs_file_path in file_paths_to_process:
    #     try:
    #         print(f"  (Real GCS read would happen for file: {gcs_file_path} for {num_rows_per_file_sample} rows)")
    #         # This is complex; Parquet doesn't directly support reading N rows without full scan or row group knowledge.
    #         # Using `pyarrow.parquet.ParquetFile` and reading specific row groups or iterating through batches
    #         # would be more efficient than full `pd.read_parquet` if only a sample is needed.
    #         # For simplicity of a mock, we are not implementing this.
    #         # Example (conceptual, might be inefficient for large files if reading full and then head):
    #         # with gcs.open(gcs_file_path, 'rb') as f:
    #         #     df_sample = pd.read_parquet(f).head(num_rows_per_file_sample)
    #         # all_sampled_dfs.append(df_sample)
    #     except Exception as e:
    #         print(f"  (Mock) Error reading or sampling file {gcs_file_path}: {e}")
    #
    # if not all_sampled_dfs:
    #      print("  (Mock) No data sampled. Returning a default internal sample.")
    #      return get_sample_dataframe() # Fallback to the existing sample
    # else:
    #      return pd.concat(all_sampled_dfs, ignore_index=True)
    # --- End of commented out real implementation sketch ---

    # For this mock version, just return the standard sample DataFrame.
    # This allows testing the pipeline flow without actual GCS interaction.
    return get_sample_dataframe()

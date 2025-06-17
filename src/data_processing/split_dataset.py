# -*- coding: utf-8 -*-
"""
Processes a single day's snapshot of Parquet files from GCS,
performs a row-level random split (train/validation/test),
and saves the splits back to GCS.
"""

import os
import argparse
import pandas as pd
# import pyarrow # Not strictly needed for pd.read_parquet/to_parquet if engine isn't specified to pyarrow, but good to list
from sklearn.model_selection import train_test_split

# Mock GCS libraries for now - uncomment and implement when GCS access is live
# from google.cloud import storage
# import gcsfs

# --- Mock GCS Functions ---
# TODO: Replace these with actual GCS interaction code when ready.
# For actual GCS interaction, you would typically use:
# from google.cloud import storage
# import gcsfs
#
# # Initialize GCS filesystem if needed for pandas
# # gcs = gcsfs.GCSFileSystem(project='your-gcp-project') # if pandas needs explicit setup
#
# # For listing blobs:
# # storage_client = storage.Client()
# # bucket = storage_client.bucket(bucket_name)
# # blobs = bucket.list_blobs(prefix=prefix_in_bucket)

def list_parquet_files_in_gcs_directory(gcs_day_path):
    """
    (Mock) Simulates listing Parquet files in a GCS directory.

    Args:
        gcs_day_path (str): The GCS path to the directory for a specific day.
                            Example: "gs://bucket/path/to/data/YYYYMMDD/"

    Returns:
        list: A list of mock GCS file paths (strings).

    In a real scenario, this would use `google.cloud.storage.Client` to list blobs
    matching a prefix and filter for '.parquet' files.
    Example:
        # from google.cloud import storage
        # storage_client = storage.Client()
        # bucket_name, prefix = gcs_day_path.replace("gs://", "").split("/", 1)
        # bucket = storage_client.bucket(bucket_name)
        # blobs = list(bucket.list_blobs(prefix=prefix))
        # parquet_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if blob.name.endswith(".parquet")]
        # return parquet_files
    """
    print(f"[Mock GCS] Listing Parquet files in: {gcs_day_path}")
    if not gcs_day_path.startswith("gs://"):
        # This check is for the mock; real GCS paths are essential for actual operations.
        print(f"Warning: Expected GCS path (gs://...) but got: {gcs_day_path}")

    # Assume GCS path is like gs://bucket/path/data/YYYYMMDD/
    # Create more realistic mock file names based on the presumed 'YYYYMMDD' part.
    dir_name = gcs_day_path.strip('/').split('/')[-1] # Should be 'YYYYMMDD'

    # Simulate finding a couple of Parquet files, as if they were part files.
    return [
        f"{gcs_day_path.rstrip('/')}/{dir_name}_data_part-00000-of-00002.parquet",
        f"{gcs_day_path.rstrip('/')}/{dir_name}_data_part-00001-of-00002.parquet"
    ]

def read_parquet_from_gcs(gcs_file_path):
    """
    (Mock) Simulates reading a single Parquet file from GCS into a Pandas DataFrame.

    Args:
        gcs_file_path (str): The GCS path to the Parquet file.

    Returns:
        pandas.DataFrame: A sample DataFrame.

    In a real scenario, this would use `pd.read_parquet(gcs_file_path)`.
    If `gcsfs` is installed and configured, pandas can handle "gs://" paths directly.
    Example:
        # import pandas as pd
        # return pd.read_parquet(gcs_file_path, engine='pyarrow') # or 'fastparquet'
    """
    print(f"[Mock GCS] Reading Parquet file: {gcs_file_path}")
    # Generate a sample DataFrame. The schema should be consistent for concatenation.
    # Let's make the IDs somewhat unique based on the file name for more robust testing.
    num_records = 10
    id_start_offset = 0
    if "part-00001" in gcs_file_path:
        id_start_offset = num_records # Start IDs for the second file where the first left off

    data = {
        'user_id': [f'user_{i+id_start_offset}' for i in range(num_records)],
        'item_id': [f'item_{(i+id_start_offset) % 5}' for i in range(num_records)], # 5 unique items per "file"
        'feature_cat': [chr(65 + (i % 3)) for i in range(num_records)], # A, B, C
        'feature_num': [i * 0.1 + id_start_offset * 0.1 for i in range(num_records)],
        'event_timestamp': pd.to_datetime('2023-01-01 00:00:00') + pd.to_timedelta(range(num_records), unit='h') + pd.Timedelta(days=id_start_offset/10),
        'label': [(i + id_start_offset) % 2 for i in range(num_records)]
    }
    return pd.DataFrame(data)

def write_df_to_parquet_gcs(df, gcs_output_path):
    """
    (Mock) Simulates writing a Pandas DataFrame to a Parquet file in GCS.

    Args:
        df (pandas.DataFrame): The DataFrame to write.
        gcs_output_path (str): The GCS path where the Parquet file should be saved.

    In a real scenario, this would use `df.to_parquet(gcs_output_path)`.
    The GCS path needs to include the filename (e.g., ".../part-00000.parquet").
    Make sure the parent directory structure (e.g., .../train/) exists or is created.
    Example:
        # Ensure the GCS directory exists (optional, to_parquet might create it or fail if deep)
        # gcs_directory = os.path.dirname(gcs_output_path)
        # if not gcsfs.GCSFileSystem().exists(gcs_directory):
        #    gcsfs.GCSFileSystem().mkdirs(gcs_directory, exist_ok=True)
        #
        # df.to_parquet(gcs_output_path, index=False, engine='pyarrow')
    """
    print(f"[Mock GCS] Writing DataFrame with {len(df)} rows to Parquet: {gcs_output_path}")
    # In a real implementation, ensure the GCS path includes the actual filename,
    # and directories are handled (e.g., gs://bucket/prefix/YYYYMMDD/train/filename.parquet).
    # df.to_parquet(gcs_output_path, index=False, engine='pyarrow')
    pass # Just printing for the mock

# --- Argument Parsing ---
# (parse_arguments function is defined above the mock functions)

# --- Main Logic ---
def main():
    """
    Main function to orchestrate the data splitting process.
    It handles argument parsing, listing GCS files (mocked), reading data (mocked),
    splitting the data, and writing the splits back to GCS (mocked).
    """
    args = parse_arguments()
    print("Starting data splitting process with the following arguments:")
    # Displaying arguments for verification
    print(f"  Input GCS Day Path: {args.input_gcs_day_path}")
    print(f"  Output GCS Path Prefix: {args.output_gcs_path_prefix}")
    print(f"  Train Fraction: {args.train_fraction}")
    print(f"  Validation Fraction: {args.validation_fraction}")
    test_fraction = 1.0 - args.train_fraction - args.validation_fraction
    print(f"  Calculated Test Fraction: {test_fraction:.2f}") # Display calculated test fraction
    print(f"  Random Seed: {args.random_seed}")

    # --- Argument Validation ---
    # Displaying arguments for verification
    print(f"  Input GCS Day Path: {args.input_gcs_day_path}")
    print(f"  Output GCS Path Prefix: {args.output_gcs_path_prefix}")
    print(f"  Train Fraction: {args.train_fraction}")
    print(f"  Validation Fraction: {args.validation_fraction}")
    test_fraction = 1.0 - args.train_fraction - args.validation_fraction
    print(f"  Calculated Test Fraction: {test_fraction:.2f}") # Display calculated test fraction
    print(f"  Random Seed: {args.random_seed}")

    # --- Argument Validation ---
    if not (0 < args.train_fraction < 1): # train_fraction must be > 0 and < 1
        raise ValueError("train_fraction must be strictly between 0 and 1.")
    if not (0 <= args.validation_fraction < 1): # validation_fraction can be 0
        raise ValueError("validation_fraction must be between 0 (inclusive) and 1 (exclusive).")
    if args.train_fraction + args.validation_fraction > 1.0: # Sum cannot exceed 1.0
        raise ValueError("The sum of train_fraction and validation_fraction must not exceed 1.0.")
    if args.train_fraction + args.validation_fraction == 1.0:
        print("Info: train_fraction + validation_fraction is 1.0, so the test set will be empty.")

    # --- Step 1: List Parquet files from the input GCS directory (mocked) ---
    print(f"\nStep 1: Listing Parquet files from '{args.input_gcs_day_path}' (mocked)...")
    try:
        gcs_file_paths = list_parquet_files_in_gcs_directory(args.input_gcs_day_path)
        if not gcs_file_paths:
            print("  No files found at the specified GCS path. Exiting.")
            return
        print(f"  Found {len(gcs_file_paths)} files: {gcs_file_paths}")
    except Exception as e:
        print(f"  Error listing files from GCS (mock): {e}")
        # In a real application, you might want to implement retries or more robust error handling.
        return # Exit if file listing fails

    # --- Step 2: Read and concatenate Parquet files into a single DataFrame (mocked) ---
    print("\nStep 2: Reading and concatenating Parquet files (mocked)...")
    all_data_frames = []
    for file_path in gcs_file_paths:
        try:
            df = read_parquet_from_gcs(file_path) # Mocked read
            all_data_frames.append(df)
            print(f"  Successfully read (mock) file: {file_path}, shape: {df.shape}")
        except Exception as e:
            print(f"  Error reading (mock) Parquet file {file_path}: {e}. Skipping this file.")
            # Error handling: decide whether to skip the file or halt the process.
            # For this script, we'll skip the problematic file.
            continue

    if not all_data_frames:
        print("  No DataFrames were successfully read. Exiting.")
        return

    daily_df = pd.concat(all_data_frames, ignore_index=True)
    print(f"  Concatenated DataFrame shape: {daily_df.shape}")
    if daily_df.empty:
        print("  The concatenated DataFrame is empty. Exiting.")
        return

    # --- Step 3: Perform train-validation-test split ---
    print("\nStep 3: Performing train-validation-test split...")

    # First, split into training and temporary (validation + test)
    # Ensure shuffling for random split if not already default
    train_df, temp_df = train_test_split(
        daily_df,
        train_size=args.train_fraction,
        random_state=args.random_seed,
        shuffle=True
    )

    # Calculate the proportion of validation set needed from the temp_df
    # This is crucial if a test set is also desired.
    remaining_fraction_after_train = 1.0 - args.train_fraction

    if remaining_fraction_after_train == 0: # Should be caught by (train_fraction + val_fraction > 1.0) check if val_fraction > 0
        val_df = pd.DataFrame() # No space left for validation or test
        test_df = pd.DataFrame()
        if args.validation_fraction > 0: # This case implies train_fraction was 1.0
             print("  Warning: train_fraction is 1.0. Validation and test sets will be empty.")
    elif args.validation_fraction == 0: # No validation set is explicitly requested
        val_df = pd.DataFrame() # Empty validation DataFrame
        test_df = temp_df       # All remaining data goes to the test set
    elif args.validation_fraction < remaining_fraction_after_train:
        # We need both validation and test sets from temp_df
        val_relative_fraction = args.validation_fraction / remaining_fraction_after_train
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_relative_fraction,
            random_state=args.random_seed, # Use the same seed for reproducibility
            shuffle=True
        )
    else: # This means validation_fraction == remaining_fraction_after_train
          # (i.e., train_fraction + validation_fraction == 1.0)
        val_df = temp_df      # All remaining data goes to validation
        test_df = pd.DataFrame() # Empty test DataFrame

    print(f"  Train set size: {len(train_df)} rows")
    print(f"  Validation set size: {len(val_df)} rows")
    print(f"  Test set size: {len(test_df)} rows")

    # --- Step 4: Construct output GCS paths and write DataFrames (mocked) ---
    print("\nStep 4: Writing split DataFrames to GCS (mocked)...")

    # Extract date string (e.g., YYYYMMDD) from the input GCS path.
    # Assumes input path ends with '.../YYYYMMDD/'.
    try:
        # os.path.basename works even with trailing slashes
        date_str = os.path.basename(args.input_gcs_day_path.rstrip('/'))
        # Basic validation for YYYYMMDD format
        if not (date_str.isdigit() and len(date_str) == 8):
            print(f"  Warning: Extracted date string '{date_str}' from '{args.input_gcs_day_path}' does not look like YYYYMMDD. Using it as is.")
    except Exception as e:
        print(f"  Warning: Could not reliably extract date from input path '{args.input_gcs_day_path}': {e}. Using 'unknown_date'.")
        date_str = "unknown_date"

    output_base_for_day = f"{args.output_gcs_path_prefix.rstrip('/')}/{date_str}"

    # Define full GCS paths for the output Parquet files.
    # For simplicity, each split is written as a single Parquet file.
    # In production, consider sharding large DataFrames into multiple files (e.g., using Dask or Spark).
    train_output_path = f"{output_base_for_day}/train/train_data.parquet" # Simplified filename
    val_output_path = f"{output_base_for_day}/validation/validation_data.parquet"
    test_output_path = f"{output_base_for_day}/test/test_data.parquet"

    # (Mock) Write DataFrames to their respective GCS locations
    if not train_df.empty:
        write_df_to_parquet_gcs(train_df, train_output_path)
    else:
        print("  Train DataFrame is empty. Skipping GCS write.")

    if not val_df.empty:
        write_df_to_parquet_gcs(val_df, val_output_path)
    else:
        print("  Validation DataFrame is empty. Skipping GCS write.")

    if not test_df.empty:
        write_df_to_parquet_gcs(test_df, test_output_path)
    else:
        print("  Test DataFrame is empty. Skipping GCS write.")

    print("\nData splitting process (mock) complete.")


def parse_arguments():
    """Parses command-line arguments for the data splitting script."""
    parser = argparse.ArgumentParser(description="Splits daily Parquet data from GCS.")
    parser.add_argument(
        "--input_gcs_day_path",
        type=str,
        required=True,
        help="GCS path to the specific day's directory containing multiple Parquet files (e.g., gs://bucket/path/to/data/YYYYMMDD/).",
    )
    parser.add_argument(
        "--output_gcs_path_prefix",
        type=str,
        required=True,
        help="GCS path prefix where split data for that day will be saved (e.g., gs://bucket/path/to/processed_data/).",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Proportion for the training set (default: 0.8).",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="Proportion for the validation set (default: 0.1). Test fraction will be 1 - train - validation.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of the split (default: 42).",
    )
    return parser.parse_args()

if __name__ == '__main__':
    # This block now directly calls main(), which handles its own argument parsing and validation.
    main()
    print("\nScript execution finished.")

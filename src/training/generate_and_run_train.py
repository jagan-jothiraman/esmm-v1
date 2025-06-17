# -*- coding: utf-8 -*-
"""
Generates an EasyRec pipeline configuration file from a template
by populating it with auto-generated feature configurations and
user-provided paths/parameters. It then prints a command to run
EasyRec training.
"""

import argparse
import os
import subprocess
import sys

# Adjust Python path to import from sibling directory 'data_processing'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_processing import auto_feature_config_generator
except ImportError:
    print("Error: Could not import 'auto_feature_config_generator' from 'data_processing'.")
    # Fallback to dummy for basic script parsing and testing other parts
    class DummyAutoFeatureConfigGenerator:
        def get_sample_dataframe(self):
            print("Warning: Using dummy 'get_sample_dataframe'.")
            return pd.DataFrame()
        def classify_features(self, df):
            print("Warning: Using dummy 'classify_features'.")
            return {}
        def generate_input_fields_proto(self, classified_features):
            print("Warning: Using dummy 'generate_input_fields_proto'.")
            return "# Dummy Input Fields"
        def generate_feature_config_proto(self, classified_features):
            print("Warning: Using dummy 'generate_feature_config_proto'.")
            return "# Dummy Feature Config"
        def load_sample_from_gcs_parquet(self, gcs_parquet_path_or_dir, num_files_to_sample=1, num_rows_per_file_sample=1000):
            print(f"Warning: Using dummy 'load_sample_from_gcs_parquet' for path {gcs_parquet_path_or_dir}.")
            return self.get_sample_dataframe()
    auto_feature_config_generator = DummyAutoFeatureConfigGenerator()
    import pandas as pd # Ensure pandas is imported for the dummy

import numpy as np # numpy might be used by auto_feature_config_generator's real version

def parse_arguments():
    """Parses command-line arguments for the training pipeline generation script."""
    parser = argparse.ArgumentParser(
        description="Generates an EasyRec pipeline configuration, optionally executes training."
    )
    parser.add_argument(
        "--gcs_train_data_for_schema_inference", type=str, default=None,
        help="Optional: GCS path to training data directory for schema inference."
    )
    parser.add_argument(
        "--template_config_path", type=str, required=True,
        help="Path to the EasyRec pipeline template config file."
    )
    parser.add_argument(
        "--output_config_path", type=str, required=True,
        help="Path to save the generated EasyRec pipeline config file."
    )
    parser.add_argument(
        "--gcs_processed_data_path_train", type=str, required=True,
        help="GCS path for training data."
    )
    parser.add_argument(
        "--gcs_processed_data_path_eval", type=str, required=True,
        help="GCS path for evaluation data."
    )
    parser.add_argument(
        "--gcs_model_dir_path", type=str, required=True,
        help="GCS path for model output."
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch_size.")
    parser.add_argument("--num_epochs", type=int, default=None, help="Override num_epochs.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning_rate.")
    parser.add_argument("--save_checkpoints_steps", type=int, default=None, help="Override save_checkpoints_steps.")
    parser.add_argument(
        "--execute_training", action="store_true",
        help="If set, executes EasyRec training after generating config."
    )
    return parser.parse_args()

def load_and_generate_feature_configs(gcs_train_data_path_for_schema):
    """Loads sample data, infers features, and generates proto snippets."""
    print(f"Step 1: Loading sample data and generating feature configurations...")
    if gcs_train_data_path_for_schema:
        print(f"  Attempting to (mock) load sample from GCS path: {gcs_train_data_path_for_schema}")
        df_sample = auto_feature_config_generator.load_sample_from_gcs_parquet(
            gcs_parquet_path_or_dir=gcs_train_data_path_for_schema
        )
    else:
        print("  No '--gcs_train_data_for_schema_inference' provided. Using default internal sample generator.")
        df_sample = auto_feature_config_generator.get_sample_dataframe()

    if df_sample is None or df_sample.empty:
        print("Error: Failed to get a valid sample DataFrame. Exiting.")
        # In case of dummy, it returns empty df, so classify_features should handle it.
        if isinstance(auto_feature_config_generator, DummyAutoFeatureConfigGenerator):
             classified_features = {} # Proceed with empty if dummy
        else:
            sys.exit(1)

    if not df_sample.empty:
        print(f"  Inferring feature types from sample DataFrame (shape: {df_sample.shape})...")
        classified_features = auto_feature_config_generator.classify_features(df_sample)
    else: # if df_sample is empty (e.g. from Dummy)
        classified_features = {}

    print(f"  Classified features: {classified_features}")
    input_fields_str = auto_feature_config_generator.generate_input_fields_proto(classified_features)
    feature_config_str = auto_feature_config_generator.generate_feature_config_proto(classified_features)
    input_fields_str_indented = "\n".join([f"  {line}" for line in input_fields_str.splitlines()])
    feature_config_str_indented = "\n".join([f"  {line}" for line in feature_config_str.splitlines()])
    print("  Successfully generated and indented input_fields and feature_config snippets.")
    return classified_features, input_fields_str_indented, feature_config_str_indented

def group_features(classified_features_dict):
    """Automatically groups features based on naming conventions."""
    print("\nStep 2: Grouping features automatically...")
    feature_groups = {'user_features': [], 'item_features': [], 'other_features': []}
    for name, f_type in classified_features_dict.items():
        if f_type == 'user_id' or name == 'user_id':
            if 'user_id' not in feature_groups['user_features']: feature_groups['user_features'].append('user_id')
        elif f_type == 'item_id' or name == 'item_id':
            if 'item_id' not in feature_groups['item_features']: feature_groups['item_features'].append('item_id')
        elif name.startswith('user_'):
            feature_groups['user_features'].append(name)
        elif name.startswith('item_'):
            feature_groups['item_features'].append(name)
        elif f_type not in ['click_label', 'conversion_label']: # Neither ID, label, nor user/item prefixed
            print(f"  Info: Assigning unmatched feature '{name}' (type: {f_type}) to 'user_features' by default for ESMM towers.")
            feature_groups['user_features'].append(name) # Default assignment, review for specific models

    # Ensure IDs are present if their group has other features
    if feature_groups['user_features'] and 'user_id' not in feature_groups['user_features'] and 'user_id' in classified_features_dict:
        feature_groups['user_features'].insert(0, 'user_id')
    if feature_groups['item_features'] and 'item_id' not in feature_groups['item_features'] and 'item_id' in classified_features_dict:
        feature_groups['item_features'].insert(0, 'item_id')

    print(f"  User features: {feature_groups['user_features']}")
    print(f"  Item features: {feature_groups['item_features']}")
    return feature_groups

def populate_and_save_config(template_path, output_path, input_fields_pb, feature_config_pb,
                             grouped_feats, gcs_paths_map, cli_params):
    """Reads template, replaces placeholders, and saves the populated config."""
    print(f"\nStep 3: Populating configuration from template '{template_path}'...")
    try:
        with open(template_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Template file not found at {template_path}. Exiting.")
        return None

    content = content.replace("gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/train/", gcs_paths_map['train'])
    content = content.replace("gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/validation/", gcs_paths_map['eval'])
    content = content.replace("gs://<YOUR_BUCKET>/<YOUR_MODEL_OUTPUT_PATH>/<YYYYMMDD>/", gcs_paths_map['model_dir'])
    content = content.replace("#<INPUT_FIELDS_PLACEHOLDER>", input_fields_pb)
    content = content.replace("#<FEATURE_CONFIG_PLACEHOLDER>", feature_config_pb)

    user_feats_str = ", ".join([f'"{f}"' for f in grouped_feats.get('user_features', [])])
    item_feats_str = ", ".join([f'"{f}"' for f in grouped_feats.get('item_features', [])])
    content = content.replace("#<USER_FEATURES_PLACEHOLDER>", user_feats_str)
    content = content.replace("#<ITEM_FEATURES_PLACEHOLDER>", item_feats_str)
    print("  Replaced GCS paths, feature snippets, and feature groups.")

    if cli_params.batch_size is not None:
        content = content.replace("batch_size: 4096", f"batch_size: {cli_params.batch_size}")
    if cli_params.num_epochs is not None:
        content = content.replace("num_epochs: 1", f"num_epochs: {cli_params.num_epochs}")
    if cli_params.learning_rate is not None:
        content = content.replace("learning_rate: 0.0005", f"learning_rate: {cli_params.learning_rate}")
    if cli_params.save_checkpoints_steps is not None:
        content = content.replace("save_checkpoints_steps: 1000", f"save_checkpoints_steps: {cli_params.save_checkpoints_steps}")
    print("  Applied parameter overrides if any.")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)
        print(f"  Successfully saved populated config to '{output_path}'")
        return output_path
    except IOError as e:
        print(f"Error saving config to '{output_path}': {e}. Exiting.")
        return None

def main():
    """Orchestrates config generation and optional training execution."""
    args = parse_arguments()

    classified_feats, input_fields_pb_str, feature_config_pb_str = \
        load_and_generate_feature_configs(args.gcs_train_data_for_schema_inference)

    if classified_feats is None: sys.exit(1) # Error already printed in helper

    grouped_features_map = group_features(classified_feats)

    gcs_paths = {
        "train": args.gcs_processed_data_path_train,
        "eval": args.gcs_processed_data_path_eval,
        "model_dir": args.gcs_model_dir_path,
    }

    generated_config_file = populate_and_save_config(
        args.template_config_path, args.output_config_path,
        input_fields_pb_str, feature_config_pb_str,
        grouped_features_map, gcs_paths, args
    )

    if not generated_config_file:
        sys.exit(1) # Error already printed

    # Step 4: Optionally execute EasyRec training
    abs_output_config_path = os.path.abspath(generated_config_file)
    training_cmd_parts = ["python", "-m", "easy_rec.python.train_eval", f"--pipeline_config_path={abs_output_config_path}"]
    training_cmd_str = " ".join(training_cmd_parts)
    print(f"\nStep 4: EasyRec Training Execution")
    print(f"  Command: {training_cmd_str}")

    if args.execute_training:
        print(f"  --execute_training flag is set. Running EasyRec training...")
        try:
            result = subprocess.run(training_cmd_parts, check=True, text=True, capture_output=True)
            print("  --- EasyRec Training STDOUT ---")
            print(result.stdout)
            print("  --- End of EasyRec Training STDOUT ---")
            print("  EasyRec training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"  Error during EasyRec training (return code {e.returncode}):")
            print("  --- STDOUT ---"); print(e.stdout)
            print("  --- STDERR ---"); print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("  Error: 'python' or EasyRec module not found. Ensure environment is set up.")
            sys.exit(1)
    else:
        print("  --execute_training flag not set. Skipping actual training execution.")

if __name__ == '__main__':
    main()

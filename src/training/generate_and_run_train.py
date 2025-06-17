# -*- coding: utf-8 -*-
"""
Generates an EasyRec pipeline configuration file from a template
by populating it with auto-generated feature configurations and
user-provided paths/parameters. It then prints a command to run
EasyRec training.
"""

import argparse
import os
import subprocess # For future use if we actually trigger the training
import sys

# Adjust Python path to import from sibling directory 'data_processing'
# This is a common way to handle local module imports.
# In a more structured project, these might be installable packages.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from data_processing import auto_feature_config_generator
except ImportError:
    print("Error: Could not import 'auto_feature_config_generator' from 'data_processing'.")
    print("Ensure 'src/data_processing/auto_feature_config_generator.py' exists and 'src' is in the Python path if necessary.")
    # As a fallback for an IDE/environment where this path adjustment isn't enough,
    # or if the file is missing, we can define dummy functions to allow the script
    # to be parsed, but it won't be fully functional.
    class DummyAutoFeatureConfigGenerator:
        def get_sample_dataframe(self):
            print("Warning: Using dummy 'get_sample_dataframe'. Feature inference will not work.")
            return None # Or a minimal pd.DataFrame if needed for type checking
        def classify_features(self, df):
            print("Warning: Using dummy 'classify_features'.")
            return {}
        def generate_input_fields_proto(self, classified_features):
            print("Warning: Using dummy 'generate_input_fields_proto'.")
            return ""
        def generate_feature_config_proto(self, classified_features):
            print("Warning: Using dummy 'generate_feature_config_proto'.")
            return ""
    # This dummy class is used as a fallback if the actual generator cannot be imported.
    # It allows the script to run without feature generation for testing other parts.
    class DummyAutoFeatureConfigGenerator:
        def get_sample_dataframe(self):
            print("Warning: Using dummy 'get_sample_dataframe'. Feature inference will be skipped.")
            # Return a very minimal DataFrame that might allow classify_features to not fail,
            # or ensure classify_features handles None gracefully.
            # For a truly dummied out version, this might return None and expect downstream to handle it.
            return pd.DataFrame() # Return empty DataFrame
        def classify_features(self, df):
            print("Warning: Using dummy 'classify_features'. Returning empty classification.")
            return {}
        def generate_input_fields_proto(self, classified_features):
            print("Warning: Using dummy 'generate_input_fields_proto'. Returning empty string.")
            return "# Dummy Input Fields"
        def generate_feature_config_proto(self, classified_features):
            print("Warning: Using dummy 'generate_feature_config_proto'. Returning empty string.")
            return "# Dummy Feature Config"
    auto_feature_config_generator = DummyAutoFeatureConfigGenerator()
    # Also import pandas for the dummy DataFrame if it's not already imported at the top level
    # and numpy for the mock GCS file listing simulation in auto_feature_config_generator
    import pandas as pd
    import numpy as np


def parse_arguments():
    """Parses command-line arguments for the training pipeline generation script."""
    parser = argparse.ArgumentParser(
        description="Generates an EasyRec pipeline configuration, optionally executes training, and manages related tasks."
    )

    # --- Paths ---
    parser.add_argument(
        "--gcs_train_data_for_schema_inference", type=str, default=None,
        help=("Optional: GCS path to the training data directory (e.g., gs://bucket/processed/YYYYMMDD/train/). "
              "A sample from this path will be used for schema inference. "
              "If None, auto_feature_config_generator's internal default sample is used.")
    )
    parser.add_argument(
        "--template_config_path", type=str, required=True,
        help="Path to the EasyRec pipeline template config file (e.g., 'configs/esmm_pipeline_template.config')."
    )
    parser.add_argument(
        "--output_config_path", type=str, required=True,
        help="Path to save the generated (populated) EasyRec pipeline config file (e.g., 'configs/esmm_generated_pipeline.config')."
    )
    parser.add_argument(
        "--gcs_processed_data_path_train", type=str, required=True,
        help="GCS path for the training data directory (e.g., 'gs://<YOUR_BUCKET>/<PROCESSED_DATA_PATH>/<YYYYMMDD>/train/')."
    )
    parser.add_argument(
        "--gcs_processed_data_path_eval", type=str, required=True,
        help="GCS path for the evaluation data directory (e.g., 'gs://<YOUR_BUCKET>/<PROCESSED_DATA_PATH>/<YYYYMMDD>/validation/')."
    )
    parser.add_argument(
        "--gcs_model_dir_path", type=str, required=True,
        help="GCS path for saving the trained model (e.g., 'gs://<YOUR_BUCKET>/<MODEL_OUTPUT_PATH>/<YYYYMMDD>/').",
    )

    # --- Config Overrides ---
    # These arguments allow overriding default values present in the template.
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override 'batch_size' in the data_config section of the template."
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None,
        help="Override 'num_epochs' in the data_config section of the template."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="Override 'learning_rate' in the train_config.optimizer_config section."
    )
    parser.add_argument(
        "--save_checkpoints_steps", type=int, default=None,
        help="Override 'save_checkpoints_steps' in the train_config section."
    )

    )
    # Add more feature group arguments if needed (e.g., --context_features)
    parser.add_argument(
        "--execute_training",
        action="store_true", # Makes it a boolean flag, True if present, False otherwise
        help="If set, executes the EasyRec training command after generating the config. Default: False."
    )

    return parser.parse_args()

def load_and_generate_feature_configs(gcs_train_data_path_for_schema):
    """
    Loads sample data (using the new GCS mock or default generator),
    infers feature types, and generates EasyRec protobuf snippets.

    Args:
        gcs_train_data_path_for_schema (str, optional): GCS path to training data,
            from which a sample will be (mock) loaded for schema inference.
            If None, uses the default sample from `auto_feature_config_generator`.

    Returns:
        tuple: (classified_features, input_fields_str_indented, feature_config_str_indented)
               The dictionary of classified features, and strings containing the
               generated protobuf snippets, indented for insertion.
               Returns (None, None, None) if feature generation fails critically.
    """
    print(f"Step 1: Loading sample data and generating feature configurations...")
    if gcs_train_data_path_for_schema:
        print(f"  Attempting to (mock) load sample from GCS path: {gcs_train_data_path_for_schema}")
        # Use the new function from auto_feature_config_generator
        df_sample = auto_feature_config_generator.load_sample_from_gcs_parquet(
            gcs_parquet_path_or_dir=gcs_train_data_path_for_schema,
            num_files_to_sample=1, # Example: sample from 1 file
            num_rows_per_file_sample=500 # Example: sample 500 rows
        )
    else:
        print("  No '--gcs_train_data_for_schema_inference' provided. Using default internal sample generator.")
        df_sample = auto_feature_config_generator.get_sample_dataframe()

    if df_sample is None:
        if not isinstance(auto_feature_config_generator, DummyAutoFeatureConfigGenerator):
            print("Error: Failed to get a sample DataFrame for feature inference. Exiting.")
            sys.exit(1)
        classified_features = {}
    elif df_sample.empty and isinstance(auto_feature_config_generator, DummyAutoFeatureConfigGenerator):
        print("  Dummy generator returned an empty DataFrame. Proceeding with empty feature classification.")
        classified_features = {}
    else:
        try:
            print(f"  Inferring feature types from sample DataFrame (shape: {df_sample.shape})...")
            classified_features = auto_feature_config_generator.classify_features(df_sample)
            print(f"  Classified features: {classified_features}")
        except Exception as e:
            print(f"Error during feature classification: {e}. Exiting.")
            sys.exit(1)

    try:
        input_fields_str = auto_feature_config_generator.generate_input_fields_proto(classified_features)
        feature_config_str = auto_feature_config_generator.generate_feature_config_proto(classified_features)
    except Exception as e:
        print(f"Error during protobuf snippet generation: {e}. Exiting.")
        sys.exit(1)

    input_fields_str_indented = "\n".join([f"  {line}" for line in input_fields_str.splitlines()])
    feature_config_str_indented = "\n".join([f"  {line}" for line in feature_config_str.splitlines()])

    print("  Successfully generated and indented input_fields and feature_config snippets.")
    return classified_features, input_fields_str_indented, feature_config_str_indented

def group_features(classified_features_dict):
    """
    Automatically groups features based on naming conventions.

    Args:
        classified_features_dict (dict): Dictionary from feature name to type.

    Returns:
        dict: A dictionary like {'user_features': [...], 'item_features': [...]}.
    """
    print("\nStep 2: Grouping features automatically...")
    feature_groups = {
        'user_features': [],
        'item_features': [],
        'other_features': [] # For features not matching user/item prefixes
    }
    # These are features that should not be part of user/item specific towers directly
    # (e.g. labels, or numerical features that might be globally used or part of context)
    # This exclusion list might need to be more dynamic or configurable.
    excluded_from_grouping = ['click_label', 'conversion_label', 'user_id', 'item_id']


    for feature_name, feature_type in classified_features_dict.items():
        if feature_type in ['user_id', 'item_id', 'click_label', 'conversion_label']:
            # IDs and labels are handled specially, not typically in these explicit feature_groups for towers
            # but user_id and item_id ARE needed for the feature_groups that define tower inputs.
            if feature_name == 'user_id': # Ensure user_id is in user_features
                 if 'user_id' not in feature_groups['user_features']:
                    feature_groups['user_features'].append(feature_name)
            elif feature_name == 'item_id': # Ensure item_id is in item_features
                 if 'item_id' not in feature_groups['item_features']:
                    feature_groups['item_features'].append(feature_name)
            continue # Skip further processing for these specific names

        if feature_name.startswith('user_'):
            feature_groups['user_features'].append(feature_name)
        elif feature_name.startswith('item_'):
            feature_groups['item_features'].append(feature_name)
        else:
            # For features that don't match 'user_' or 'item_' prefixes,
            # and are not special IDs/labels.
            # Example: 'feature_num_1', 'feature_cat_1' from the sample generator.
            # These might go into a 'context' group or be assigned based on other rules.
            # For ESMM, we primarily need user and item features for their respective towers.
            # Let's assign remaining categorical and numerical features to user tower by default for now,
            # as context features are not explicitly handled by this basic ESMM config.
            # This is a simplification and might need refinement based on actual feature semantics.
            if feature_type == 'categorical' or feature_type == 'numerical':
                 print(f"  Info: Assigning unmatched feature '{feature_name}' (type: {feature_type}) to 'user_features' by default.")
                 feature_groups['user_features'].append(feature_name)
            else:
                 feature_groups['other_features'].append(feature_name)


    # Ensure user_id and item_id are present if any other user/item features were added,
    # as they are critical for the towers.
    if feature_groups['user_features'] and 'user_id' not in feature_groups['user_features']:
        if 'user_id' in classified_features_dict:
             feature_groups['user_features'].insert(0, 'user_id') # Add user_id if available
    if feature_groups['item_features'] and 'item_id' not in feature_groups['item_features']:
        if 'item_id' in classified_features_dict:
            feature_groups['item_features'].insert(0, 'item_id') # Add item_id if available

    print(f"  User features: {feature_groups['user_features']}")
    print(f"  Item features: {feature_groups['item_features']}")
    if feature_groups['other_features']:
        print(f"  Other features (not grouped into user/item towers): {feature_groups['other_features']}")
    return feature_groups


def main():
    """
    Main function to:
    1. Parse command-line arguments.
    2. Load sample data and generate feature configuration snippets.
    3. Automatically group features.
    4. Read the pipeline template file.
    5. Replace placeholders in the template.
    6. Save the populated configuration file.
    7. Optionally execute EasyRec training.
    """
    args = parse_arguments()

    # Step 1: Load sample data and generate feature configuration strings
    classified_features, input_fields_str, feature_config_str = \
        load_and_generate_feature_configs(args.gcs_train_data_for_schema_inference)

    if classified_features is None: # Critical failure in previous step
        sys.exit(1)

    # Step 2: Automatically group features
    # This replaces the manual --user_features and --item_features arguments
    grouped_features = group_features(classified_features)


    # Step 3: Read the template configuration file
    print(f"\nStep 3: Reading template configuration from '{args.template_config_path}'...")
    try:
        with open(args.template_config_path, 'r') as f:
            template_content = f.read()
        print(f"  Successfully read template file: {args.template_config_path}")
    except FileNotFoundError:
        print(f"Error: Template configuration file not found at '{args.template_config_path}'. Please check the path. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading template configuration file '{args.template_config_path}': {e}. Exiting.")
        sys.exit(1)

    # Step 4: Replace placeholders in the template content
    print("\nStep 4: Replacing placeholders in the template...")
    populated_content = template_content

    # A. Replace GCS path placeholders
    populated_content = populated_content.replace(
        "gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/train/",
        args.gcs_processed_data_path_train
    )
    populated_content = populated_content.replace(
        "gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/validation/",
        args.gcs_processed_data_path_eval
    )
    populated_content = populated_content.replace(
        "gs://<YOUR_BUCKET>/<YOUR_MODEL_OUTPUT_PATH>/<YYYYMMDD>/",
        args.gcs_model_dir_path
    )
    print(f"  Replaced GCS path placeholders.")

    # B. Replace feature and input field configuration placeholders
    populated_content = populated_content.replace("#<INPUT_FIELDS_PLACEHOLDER>", input_fields_str)
    populated_content = populated_content.replace("#<FEATURE_CONFIG_PLACEHOLDER>", feature_config_str)
    print(f"  Replaced #<INPUT_FIELDS_PLACEHOLDER> and #<FEATURE_CONFIG_PLACEHOLDER>.")

    # C. Replace feature group placeholders using automatically grouped features
    user_features_str = ", ".join([f'"{name}"' for name in grouped_features.get('user_features', [])])
    item_features_str = ", ".join([f'"{name}"' for name in grouped_features.get('item_features', [])])

    populated_content = populated_content.replace("#<USER_FEATURES_PLACEHOLDER>", user_features_str)
    populated_content = populated_content.replace("#<ITEM_FEATURES_PLACEHOLDER>", item_features_str)
    print(f"  Replaced feature group placeholders with automatically grouped features: User features=[{user_features_str}], Item features=[{item_features_str}].")

    # D. Replace other configurable parameters if provided via arguments
    # (This logic remains similar to the previous version)
    if args.batch_size is not None:
        populated_content = populated_content.replace("batch_size: 4096", f"batch_size: {args.batch_size}")
        print(f"  Overrode 'batch_size' to {args.batch_size}.")
    if args.num_epochs is not None:
        populated_content = populated_content.replace("num_epochs: 1", f"num_epochs: {args.num_epochs}")
        print(f"  Overrode 'num_epochs' to {args.num_epochs}.")
    if args.learning_rate is not None:
        default_lr_str = "learning_rate: 0.0005"
        target_lr_str = f"learning_rate: {args.learning_rate}"
        if default_lr_str in populated_content:
            populated_content = populated_content.replace(default_lr_str, target_lr_str)
            print(f"  Overrode 'learning_rate' to {args.learning_rate}.")
        else:
            print(f"  Warning: Could not find default learning rate string ('{default_lr_str}') to replace.")

    if args.save_checkpoints_steps is not None:
        populated_content = populated_content.replace("save_checkpoints_steps: 1000", f"save_checkpoints_steps: {args.save_checkpoints_steps}")
        print(f"  Overrode 'save_checkpoints_steps' to {args.save_checkpoints_steps}.")

    # Step 5: Save the populated configuration to the output file
    print(f"\nStep 5: Saving populated configuration to '{args.output_config_path}'...")
    try:
        output_dir = os.path.dirname(args.output_config_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_config_path, 'w') as f:
            f.write(populated_content)
        print(f"  Successfully saved generated config to '{args.output_config_path}'")
    except IOError as e:
        print(f"Error saving populated configuration to '{args.output_config_path}': {e}. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while saving the configuration: {e}. Exiting.")
        sys.exit(1)

    # Step 6: Optionally execute EasyRec training
    abs_output_config_path = os.path.abspath(args.output_config_path)
    training_command_parts = [
        "python", "-m", "easy_rec.python.train_eval",
        f"--pipeline_config_path={abs_output_config_path}"
    ]
    training_command_str = " ".join(training_command_parts)

    print("\nStep 6: EasyRec Training Execution")
    print("====================================================================")
    print("To run EasyRec training with the generated configuration:")
    print(f"  {training_command_str}")
    print("====================================================================")

    if args.execute_training:
        print(f"\n--execute_training flag is set. Attempting to run EasyRec training...")
        print(f"Executing command: {training_command_str}")
        try:
            # Using list of command parts for subprocess.run is generally safer.
            process_result = subprocess.run(training_command_parts, check=True, text=True, capture_output=True)
            print("\n--- EasyRec Training STDOUT ---")
            print(process_result.stdout)
            print("--- End of EasyRec Training STDOUT ---")
            print("\nEasyRec training process completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"\nError during EasyRec training execution (return code {e.returncode}):")
            print("--- EasyRec Training STDOUT ---")
            print(e.stdout)
            print("--- End of EasyRec Training STDOUT ---")
            print("--- EasyRec Training STDERR ---")
            print(e.stderr)
            print("--- End of EasyRec Training STDERR ---")
            print("EasyRec training failed.")
            sys.exit(1) # Exit with error if training fails
        except FileNotFoundError:
            print("\nError: 'python' command not found, or 'easy_rec' module not in PYTHONPATH.")
            print("Please ensure your Python environment is set up correctly for EasyRec.")
            sys.exit(1)
    else:
        print("\n--execute_training flag is not set. Skipping actual training execution.")
        print("Please run the command manually if training is desired.")

if __name__ == '__main__':
    main()

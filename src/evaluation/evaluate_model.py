# -*- coding: utf-8 -*-
"""
Calculates Group AUC (GAUC) for click and conversion tasks from a prediction file.
This script is intended to be used for offline evaluation of model predictions.
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score # Used as a reference or for individual group AUCs

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Calculate Group AUC from prediction files.")
    parser.add_argument(
        "--prediction_file_path", type=str, required=True,
        help="Path to the Parquet or CSV file containing predictions."
    )
    parser.add_argument(
        "--group_by_column", type=str, default="user_id",
        help="Name of the column to group by for GAUC calculation (e.g., 'user_id', 'item_id')."
    )
    parser.add_argument(
        "--click_label_column", type=str, default="click",
        help="Name of the true click label column."
    )
    parser.add_argument(
        "--click_score_column", type=str, default="click_prediction_score",
        help="Name of the predicted click score column."
    )
    parser.add_argument(
        "--conversion_label_column", type=str, default="conversion",
        help="Name of the true conversion label column."
    )
    parser.add_argument(
        "--conversion_score_column", type=str, default="conversion_prediction_score",
        help="Name of the predicted conversion score column."
    )
    parser.add_argument(
        "--file_type", type=str, default="parquet", choices=["parquet", "csv"],
        help="Type of the prediction file ('parquet' or 'csv')."
    )
    parser.add_argument(
        "--create_mock_file_if_not_exists", action="store_true",
        help="If set, creates a mock prediction file if the specified one doesn't exist."
    )
    return parser.parse_args()

def create_mock_prediction_file(file_path, num_rows=10000, num_groups=100, group_col_name="user_id",
                                click_label_col="click", click_score_col="click_prediction_score",
                                conv_label_col="conversion", conv_score_col="conversion_prediction_score",
                                file_type="parquet"):
    """
    Creates a dummy Parquet or CSV file with mock prediction data.
    """
    print(f"Creating mock prediction file at: {file_path}")
    group_ids = [f"group_{i}" for i in range(num_groups)]
    data = {
        group_col_name: np.random.choice(group_ids, size=num_rows),
        click_label_col: np.random.randint(0, 2, size=num_rows),
        click_score_col: np.random.rand(num_rows),
        conv_label_col: np.random.randint(0, 2, size=num_rows),
        conv_score_col: np.random.rand(num_rows),
    }

    # Ensure some clicks actually lead to conversions for more realistic CVR data
    # For rows where click is 0, conversion must be 0.
    data[conv_label_col][data[click_label_col] == 0] = 0
    # For CVR scores, they should ideally be conditional on click as well,
    # pCVR = P(conversion=1 | click=1). Scores when click=0 are often ignored or handled differently.
    # For mock data, we can just generate scores, but keep this in mind for real data.
    # A simple way: make conversion scores lower if click is 0.
    data[conv_score_col][data[click_label_col] == 0] = np.random.rand(np.sum(data[click_label_col] == 0)) * 0.1


    df = pd.DataFrame(data)

    try:
        if file_type == "parquet":
            df.to_parquet(file_path, index=False)
        elif file_type == "csv":
            df.to_csv(file_path, index=False)
        print(f"Mock {file_type} file created successfully with {num_rows} rows and {num_groups} groups.")
    except Exception as e:
        print(f"Error creating mock file: {e}")
        raise

def calculate_group_auc(df, score_col, label_col, group_col):
    """
    Calculates Group AUC (GAUC), a metric that averages AUC scores across different groups.
    This version weights each group's AUC by the number of impressions (samples) in that group.

    The formula for AUC within a single group (e.g., for a single user_id) is:
      AUC_group = (sum_rank_pos - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    where:
      - `sum_rank_pos` is the sum of ranks of positive instances within the group.
      - `pos_count` is the number of positive instances (label=1) in the group.
      - `neg_count` is the number of negative instances (label=0) in the group.
    Ranks are 1-based and assigned based on the prediction score (higher score = higher rank).
    Ties in scores are handled by assigning the average rank (`method='average'`) or
    ranking by order of appearance (`method='first'`). Using 'first' ensures unique ranks
    which is often preferred for this AUC formula.

    Groups where `pos_count` is 0 or `neg_count` is 0 (i.e., all labels are the same)
    have an undefined AUC by this formula (division by zero). These groups are typically
    skipped in the GAUC calculation. The final GAUC is the weighted average of AUCs
    from valid groups.

    Args:
        df (pd.DataFrame): DataFrame containing prediction scores, true labels, and group identifiers.
        score_col (str): Name of the column containing prediction scores.
        label_col (str): Name of the column containing true labels (0 or 1).
        group_col (str): Name of the column used for grouping (e.g., 'user_id').

    Returns:
        float: The calculated GAUC score, weighted by group size.
               Returns np.nan if GAUC cannot be computed (e.g., no valid groups).
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Warning: Input DataFrame is empty or not a Pandas DataFrame. Cannot calculate GAUC.")
        return np.nan

    required_cols = [score_col, label_col, group_col]
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns: {missing_cols}. GAUC calculation failed.")
        return np.nan

    print(f"Calculating GAUC for score_col='{score_col}', label_col='{label_col}', grouped_by='{group_col}'...")

    # Assign ranks within each group based on score. Higher score = higher rank.
    # `method='first'` ensures unique ranks for ties, which is important for the formula.
    # `ascending=True` means lower scores get lower ranks (1, 2, ...).
    # The formula assumes sum of ranks of *positive* examples.
    df_copy = df.copy() # Avoid modifying the original DataFrame
    df_copy['rank_in_group'] = df_copy.groupby(group_col)[score_col].rank(method='first', ascending=True)

    group_auc_values = []
    group_weights_values = [] # Using impression count (group size) as weight

    num_total_groups = df_copy[group_col].nunique()
    num_processed_groups = 0
    num_skipped_groups_no_pos = 0
    num_skipped_groups_no_neg = 0

    for group_id, group_data in df_copy.groupby(group_col):
        num_processed_groups += 1
        pos_count = group_data[label_col].sum()
        neg_count = len(group_data) - pos_count

        if pos_count == 0:
            num_skipped_groups_no_pos += 1
            # print(f"  Skipping group '{group_id}': No positive samples (pos_count=0).")
            continue # AUC is undefined for this group

        if neg_count == 0:
            num_skipped_groups_no_neg += 1
            # print(f"  Skipping group '{group_id}': No negative samples (neg_count=0).")
            continue # AUC is undefined for this group

        # Sum of ranks for positive instances within the current group
        sum_rank_pos = group_data.loc[group_data[label_col] == 1, 'rank_in_group'].sum()

        # Calculate AUC for the current group using the formula
        group_auc = (sum_rank_pos - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)

        group_auc_values.append(group_auc)
        group_weights_values.append(len(group_data)) # Weight by group size (number of impressions)

    if num_skipped_groups_no_pos > 0:
        print(f"  Info: Skipped {num_skipped_groups_no_pos} groups because they had no positive samples.")
    if num_skipped_groups_no_neg > 0:
        print(f"  Info: Skipped {num_skipped_groups_no_neg} groups because they had no negative samples.")

    if not group_auc_values:
        print("Warning: No valid group AUCs could be calculated (e.g., all groups had only one class or were empty). GAUC is np.nan.")
        return np.nan

    # Calculate the weighted average of the valid group AUCs
    # np.average returns np.nan if weights sum to 0, which shouldn't happen if group_auc_values is not empty.
    final_gauc = np.average(group_auc_values, weights=group_weights_values if sum(group_weights_values) > 0 else None)

    num_valid_groups_for_auc = len(group_auc_values)
    print(f"  Calculated AUC for {num_valid_groups_for_auc} out of {num_total_groups} total groups.")
    print(f"  Final GAUC (weighted by group size): {final_gauc:.6f}")
    return final_gauc


def main():
    """Main function to load data and calculate Group AUCs."""
    args = parse_arguments()

    # --- Ensure prediction file exists or create mock ---
    prediction_file_exists = os.path.exists(args.prediction_file_path)
    if not prediction_file_exists and args.create_mock_file_if_not_exists:
        print(f"Prediction file '{args.prediction_file_path}' not found.")
        create_mock_prediction_file(
            args.prediction_file_path,
            group_col_name=args.group_by_column,
            click_label_col=args.click_label_column,
            click_score_col=args.click_score_column,
            conv_label_col=args.conversion_label_column,
            conv_score_col=args.conversion_score_column,
            file_type=args.file_type
        )
    elif not prediction_file_exists:
        print(f"Error: Prediction file '{args.prediction_file_path}' not found. Use --create_mock_file_if_not_exists to generate one. Exiting.")
        return

    # --- Load prediction data ---
    print(f"\nLoading prediction data from: {args.prediction_file_path} (type: {args.file_type})")
    try:
        if args.file_type == "parquet":
            df_predictions = pd.read_parquet(args.prediction_file_path)
        elif args.file_type == "csv":
            df_predictions = pd.read_csv(args.prediction_file_path)
        else: # Should not happen due to argparse choices
            print(f"Error: Unsupported file type '{args.file_type}'. Exiting.")
            return
        print(f"  Successfully loaded prediction data. Shape: {df_predictions.shape}")
    except Exception as e:
        print(f"Error loading prediction file '{args.prediction_file_path}': {e}. Exiting.")
        return

    if df_predictions.empty:
        print("Error: Loaded prediction DataFrame is empty. Exiting.")
        return

    # --- Calculate and print Group AUC for Click task ---
    print("\n--- Click Task Evaluation ---")
    gauc_click = calculate_group_auc(
        df_predictions.copy(), # Pass a copy to avoid modifying original df with 'rank_in_group'
        score_col=args.click_score_column,
        label_col=args.click_label_column,
        group_col=args.group_by_column
    )
    if pd.notna(gauc_click):
        print(f"Group AUC (GAUC) for Click Task, grouped by '{args.group_by_column}': {gauc_click:.6f}")
    else:
        print(f"Could not calculate GAUC for Click Task (grouped by '{args.group_by_column}').")

    # --- Calculate and print Group AUC for Conversion task ---
    # For CVR, predictions are typically pCVR = P(conversion=1 | click=1).
    # So, evaluation should ideally be on the subset of data where click=1.
    print("\n--- Conversion Task Evaluation (on clicked samples) ---")
    df_clicked_samples = df_predictions[df_predictions[args.click_label_column] == 1].copy()

    if df_clicked_samples.empty:
        print("No samples with click=1 found. Cannot calculate GAUC for conversion task on clicked samples.")
    else:
        print(f"  Evaluating CVR on {len(df_clicked_samples)} samples where '{args.click_label_column}' is 1.")
        gauc_conversion = calculate_group_auc(
            df_clicked_samples, # Use only clicked samples for CVR GAUC
            score_col=args.conversion_score_column,
            label_col=args.conversion_label_column,
            group_col=args.group_by_column
        )
        if pd.notna(gauc_conversion):
            print(f"Group AUC (GAUC) for Conversion Task (on clicked samples), grouped by '{args.group_by_column}': {gauc_conversion:.6f}")
        else:
            print(f"Could not calculate GAUC for Conversion Task (on clicked samples, grouped by '{args.group_by_column}').")

    print("\nEvaluation script finished.")
    print("Note: This script provides GAUC. EasyRec's evaluation component would provide a broader set of metrics (AUC, Recall, etc.) based on the pipeline config.")
    print("Actual model predictions would be generated by EasyRec's prediction or evaluation steps.")


if __name__ == '__main__':
    # For basic testing from command line
    # Example:
    # python src/evaluation/evaluate_model.py --prediction_file_path=mock_preds.parquet --create_mock_file_if_not_exists
    import os # For checking file existence in main guard
    main()

# ESMM Recommender Pipeline with EasyRec

## Overview

This project implements an End-to-End (E2E) recommender system pipeline for training an Explicit Semantic Matching Model (ESMM) using the EasyRec framework. ESMM is a multi-task learning model commonly used for Click-Through Rate (CTR) and Conversion Rate (CVR) prediction in recommendation systems.

The pipeline includes scripts for:
- Data processing and splitting (simulated from GCS).
- Automatic feature configuration generation based on data schema.
- Training pipeline configuration generation from a template.
- (Optional) Execution of EasyRec training.
- Offline model evaluation with Group AUC (GAUC).

## Project Structure

```
esmm_recommender/
├── configs/
│   └── esmm_pipeline_template.config  # EasyRec pipeline configuration template
├── notebooks/
│   └── 01_data_processing_and_config_generation_test.ipynb # Notebook for testing/demo
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── auto_feature_config_generator.py # Infers schema, generates feature configs
│   │   └── split_dataset.py                 # Splits daily data (simulated GCS)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluate_model.py                # Calculates Group AUC
│   ├── training/
│   │   ├── __init__.py
│   │   └── generate_and_run_train.py        # Generates final config and runs training
│   └── __init__.py
├── requirements.txt                         # Python package dependencies
├── setup.py                                 # For packaging and installing the project
├── MANIFEST.in                              # Specifies files for sdist
└── README.md                                # This file
```

## Prerequisites

- **Python:** Version 3.7 - 3.10 (check EasyRec compatibility for specific versions).
- **Google Cloud SDK:** Required if interacting with actual GCS buckets. Ensure `gcloud` is configured.
- **EasyRec:** This pipeline relies on EasyRec for model training. Please follow the [EasyRec Installation Guide](https://easyrec.readthedocs.io/en/latest/install.html) (link is illustrative, replace with actual if available). EasyRec has its own dependencies, including specific versions of TensorFlow.
- **Java:** A Java Development Kit (JDK) might be required by some underlying libraries used by EasyRec or its dependencies (e.g., for Hadoop/Spark components if interacting with certain data formats or distributed file systems, though not directly used by this project's scripts for GCS).
- **Virtual Environment:** Recommended to avoid conflicts with system-wide packages.

## Setup / Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd esmm_recommender
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `google-cloud-storage` and `gcsfs` are listed in `requirements.txt` but commented out as current scripts use mocks for GCS operations. Uncomment them if you plan to implement real GCS interactions.*

4.  **Install the project in editable mode:**
    This allows you to use the scripts as command-line tools and makes the `src` modules importable.
    ```bash
    pip install -e .
    ```

5.  **Install EasyRec and TensorFlow:**
    Follow the official EasyRec documentation for installation. This step is crucial if you intend to execute the training (`--execute_training` flag).

## Configuration

The main configuration for the EasyRec training pipeline is managed through `configs/esmm_pipeline_template.config`. This template contains placeholders that are populated by the `generate_and_run_train.py` script.

Key placeholders to be aware of (these are filled by script arguments):
-   `train_input_path: "gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/train/"`
-   `eval_input_path: "gs://<YOUR_BUCKET>/<YOUR_PROCESSED_DATA_PATH>/<YYYYMMDD>/validation/"`
-   `model_dir: "gs://<YOUR_BUCKET>/<YOUR_MODEL_OUTPUT_PATH>/<YYYYMMDD>/"`
-   `#<INPUT_FIELDS_PLACEHOLDER>`: Populated by `auto_feature_config_generator.py`.
-   `#<FEATURE_CONFIG_PLACEHOLDER>`: Populated by `auto_feature_config_generator.py`.
-   `#<USER_FEATURES_PLACEHOLDER>`: Populated by automated grouping in `generate_and_run_train.py`.
-   `#<ITEM_FEATURES_PLACEHOLDER>`: Populated by automated grouping in `generate_and_run_train.py`.

Default parameters like `batch_size`, `num_epochs`, `learning_rate`, etc., within the template can also be overridden by command-line arguments to `generate_and_run_train.py`.

## Pipeline Workflow (How to Run)

The pipeline is designed to be run sequentially using the console scripts made available by `setup.py`.
*(Note: Current scripts use mock GCS operations. For real GCS interaction, these scripts would need to be updated.)*

### Step 1: Data Splitting

This step simulates processing a day's snapshot of Parquet files from GCS, performs a row-level random split (train/validation/test), and saves the splits back to GCS.

**Script:** `esmm_split_data` (points to `src.data_processing.split_dataset:main`)

**Example Command:**
```bash
esmm_split_data \
    --input_gcs_day_path "gs://your-bucket/path/to/raw_data/YYYYMMDD/" \
    --output_gcs_path_prefix "gs://your-bucket/path/to/processed_data/" \
    --train_fraction 0.8 \
    --validation_fraction 0.1 \
    --random_seed 42
```
Replace `YYYYMMDD` with the specific date partition.

### Step 2: Generate Config & Train Model

This step generates the final EasyRec pipeline configuration by populating the template. It uses a sample from the (mocked) GCS training data to infer schema and auto-generate feature configurations. It can also optionally trigger the EasyRec training process.

**Script:** `esmm_generate_train_config` (points to `src.training.generate_and_run_train:main`)

**Example Command (Generate Config Only):**
```bash
esmm_generate_train_config \
    --gcs_train_data_for_schema_inference "gs://your-bucket/path/to/processed_data/YYYYMMDD/train/" \
    --template_config_path "configs/esmm_pipeline_template.config" \
    --output_config_path "configs/generated_pipeline_YYYYMMDD.config" \
    --gcs_processed_data_path_train "gs://your-bucket/path/to/processed_data/YYYYMMDD/train/" \
    --gcs_processed_data_path_eval "gs://your-bucket/path/to/processed_data/YYYYMMDD/validation/" \
    --gcs_model_dir_path "gs://your-bucket/path/to/models/YYYYMMDD/" \
    --num_epochs 2 \
    --learning_rate 0.0003
```

**Example Command (Generate Config and Execute Training):**
```bash
esmm_generate_train_config \
    # ... (same arguments as above) ... \
    --execute_training
```
*(Requires EasyRec and TensorFlow to be installed and configured in your environment.)*

### Step 3: Evaluate Model

After training, EasyRec will typically output predictions if configured to do so, or you can run a separate prediction job. This script calculates Group AUC (GAUC) from such a prediction file.

**Script:** `esmm_evaluate_model` (points to `src.evaluation.evaluate_model:main`)

**Example Command:**
```bash
esmm_evaluate_model \
    --prediction_file_path "path/to/your/model_predictions.parquet" \
    --group_by_column "user_id" \
    --click_label_column "click" \
    --click_score_column "click_prediction_score" \
    --conversion_label_column "conversion" \
    --conversion_score_column "conversion_prediction_score" \
    --file_type "parquet"
```
*Note: The `prediction_file_path` should point to a file containing true labels and predicted scores, typically generated by an EasyRec prediction or evaluation job.*
*The script can also generate a mock prediction file using `--create_mock_file_if_not_exists` for testing its GAUC logic.*

## Key Scripts Overview

-   **`src/data_processing/split_dataset.py` (`esmm_split_data`):**
    Simulates splitting a day's worth of data from GCS into train, validation, and test sets. (Currently uses mock GCS functions).
-   **`src/data_processing/auto_feature_config_generator.py`:**
    Contains logic to infer feature types from a sample DataFrame and generate EasyRec `input_fields` and `feature_config` protobuf snippets. It also includes a (mock) function to simulate loading a sample from GCS Parquet files. The entry point `esmm_generate_features` is mainly for illustrative prints of these snippets.
-   **`src/training/generate_and_run_train.py` (`esmm_generate_train_config`):**
    The core script for preparing and optionally launching an EasyRec training job. It:
    1.  (Mocks) Loads a sample from GCS training data for schema inference.
    2.  Uses `auto_feature_config_generator` to create feature protobufs.
    3.  Automates basic feature grouping (user, item features) based on naming conventions.
    4.  Populates `configs/esmm_pipeline_template.config` with these generated configs, GCS paths, and other parameters.
    5.  Saves the final, populated pipeline configuration.
    6.  If `--execute_training` is specified, it attempts to run EasyRec training using `subprocess`.
-   **`src/evaluation/evaluate_model.py` (`esmm_evaluate_model`):**
    Calculates Group AUC (GAUC) for click and conversion tasks from a prediction output file (which would be generated by EasyRec). Includes a utility to create a mock prediction file.

## Notebooks

-   **`notebooks/01_data_processing_and_config_generation_test.ipynb`:**
    Provides a step-by-step guide and test environment for the data simulation, feature configuration generation, and pipeline config population logic. Useful for debugging the configuration generation part of the pipeline.

## Future Work / TODOs (Optional)

-   **Implement Real GCS Calls:** Replace mock GCS functions in `split_dataset.py` and `auto_feature_config_generator.py` with actual `google-cloud-storage` and `gcsfs` operations.
-   **Advanced Feature Grouping:** Enhance the automated feature grouping in `generate_and_run_train.py` to be more robust or configurable (e.g., using a config file for group assignments or more sophisticated inference).
-   **Hyperparameter Optimization (HPO):** Integrate HPO scripts or tools.
-   **Dockerization:** Create Dockerfiles for easier environment setup and deployment.
-   **Inference Pipeline:** Detail or implement scripts for running batch or online inference with the trained model.
-   **Error Handling and Logging:** Improve error handling and logging across all scripts.
-   **Testing:** Add unit and integration tests.

---
*This README provides a guide to the project. Ensure paths, bucket names, and specific configurations are updated to match your environment.*
```

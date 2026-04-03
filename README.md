# Sexismo en Twitter Notebook

This repository contains a Jupyter notebook for training a text classification model on the EXIST2021 dataset. The goal is to detect sexist vs. non-sexist tweets in Spanish using a Hugging Face Transformer model.

## Main Notebook

The main working notebook is `Sexismo_en_twitter_training_model.ipynb`. It performs the full training pipeline:

1. Load environment variables from a local `.env` file if present.
2. Read the Hugging Face token from `HF_TOKEN`.
3. Detect whether a GPU is available and select the runtime device.
4. Load the EXIST2021 training and labeled test data.
5. Keep only Spanish rows.
6. Remove unused dataset columns and create numeric labels.
7. Preprocess tweet text with a cleaning function.
8. Build the training and validation datasets.
9. Load a pretrained transformer tokenizer, configuration, and sequence classification model.
10. Train and evaluate the model.
11. Save the final tokenizer and model into `model_output/`.

## Data Files

The notebook uses the following dataset files:

- `EXIST2021_dataset/training/EXIST2021_training.tsv`: main training set.
- `EXIST2021_dataset/test/EXIST2021_test_labeled.tsv`: labeled test split used in the notebook.
- `EXIST2021_dataset/test/EXIST2021_test.tsv`: unlabeled test file included in the repository.

The notebook filters the data to Spanish examples only by checking the `language` column.

## Dependencies

The required Python packages are listed in `requirements.txt`.

The notebook also expects a Hugging Face access token to be available as the `HF_TOKEN` environment variable. If a `.env` file is used, it should contain something like:

```env
HF_TOKEN=your_token_here
```

## Notebook Workflow

### 1. Environment setup
The first cells load environment variables and verify that `HF_TOKEN` is available. The notebook will stop early if the token is missing.

### 2. Runtime selection
PyTorch is used to choose between GPU and CPU execution.

### 3. Data loading
The notebook reads the TSV files with pandas, filters to Spanish rows, drops metadata columns, and creates a binary `labels` column:

- `sexist` -> `1`
- `non-sexist` -> `0`

### 4. Text preprocessing
A custom `preprocess()` function normalizes tweet text.

### 5. Model preparation
The notebook loads a pretrained transformer model and tokenizer using Hugging Face utilities. It then prepares tokenized inputs, attention masks, and PyTorch dataloaders.

### 6. Training and evaluation
The model is trained for the configured number of epochs and evaluated on the validation split. Metrics and intermediate outputs are shown in notebook cells.

### 7. Saving artifacts
After training, the final model and tokenizer are written to `model_output/`.

## Outputs

After a successful run, `model_output/` should contain the saved model and tokenizer files used for later inference or reuse.

## Notes

- The notebook is intended to be run from the repository root so relative paths resolve correctly.

# Penguin Species Classification

This project predicts penguin species from four physical measurements:

- bill length;
- bill depth;
- flipper length;
- body mass.

`island` and `sex` are retained as metadata for error analysis but are not used as model inputs.

## Workflow

1. Load the Seaborn Penguins dataset.
2. Validate its schema and domain constraints with Pandera.
3. Store the validated source snapshot in Parquet through PyArrow.
4. Exclude the two observations for which all four model features are missing.
5. Create a deterministic stratified 70/15/15 train/validation/test split.
6. Compare a most-frequent-class baseline, scaled Logistic Regression, and Random Forest.
7. Select the best candidate by validation macro F1.
8. Refit the selected model on train plus validation data and evaluate it once on the test split.
9. Inspect the confusion matrix, individual errors, subgroup performance, and permutation importance.

## Data roles

- Features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Target: `species`
- Metadata: `island`, `sex`
- Stable record identifier: `id`

## Project structure

```text
penguins-ml-project/
├── data/
│   ├── raw/
│   │   └── Penguins.parquet
│   └── processed/
│       ├── penguins_model.parquet
│       └── split_assignments.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling_complete.ipynb
├── src/penguins_project/
│   ├── data.py
│   ├── schema.py
│   └── train.py
├── tests/
│   └── test_data.py
├── pyproject.toml
└── uv.lock
```

## Reproduce the project

Install the locked environment:

```powershell
uv sync
```

Generate the processed dataset and split assignments:

```powershell
uv run python -m penguins_project.train
```

Run JupyterLab:

```powershell
uv run jupyter lab
```

Open `notebooks/02_modeling_complete.ipynb` and select **Kernel → Restart Kernel and Run All Cells**.

Run the data validation tests:

```powershell
uv run pytest -q
```

## Evaluation

The notebook reports:

- validation and test accuracy;
- validation and test macro F1;
- per-class precision, recall, and F1;
- a test confusion matrix;
- individual misclassified observations;
- descriptive accuracy by island and sex;
- permutation feature importance.

The test split is not used for model selection. It is evaluated only after the best candidate has been selected using the validation split.

## Limitations

- The dataset is small, so metric estimates can vary across splits.
- The results may not generalize beyond the observed Palmer Archipelago population.
- Two observations with no usable physical measurements are excluded.
- Subgroup metrics are descriptive because some test subgroups contain few observations.
- Feature importance is associational rather than causal.

## AI disclosure

AI assistance was used for code review, debugging, explanations, and notebook structuring. The author reviewed the methodology, executed the analysis, inspected the outputs, and is responsible for the final conclusions.

# Penguin Species Classification

## Research question

Can penguin species be predicted from four physical measurements alone, without using island or sex as model inputs? What additional structure is revealed when the same measurements are clustered without species labels?

This is a compact, reproducible machine-learning investigation based on the Palmer Penguins dataset. It focuses on sound evaluation, leakage prevention, error analysis, and carefully limited conclusions rather than maximizing a single score.

## Data

The data are the Palmer Penguins dataset distributed with Seaborn. The observations were originally collected at Palmer Station in Antarctica by Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research program.

- [Seaborn dataset file](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)
- [Original Palmer Penguins project](https://allisonhorst.github.io/palmerpenguins/)

Data roles:

- Features: `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`
- Target: `species`
- Metadata used only for error analysis: `island`, `sex`
- Stable identifier: `id`

The schema and domain constraints are validated with Pandera. Two of the 344 observations contain no values for any of the four model features, so they are excluded. The remaining 342 observations are stored in Parquet and assigned to a deterministic stratified 70/15/15 train/validation/test split.

## Method

Three models are compared:

1. `DummyClassifier` with the most-frequent strategy as the baseline.
2. Scaled Logistic Regression as a simple linear candidate.
3. Random Forest as a nonlinear candidate.

Candidate models are compared using five-fold stratified cross-validation on the training partition. Accuracy and macro F1 are reported, with macro F1 used for model selection because Chinstrap is less common than Adelie and Gentoo. Scaling is kept inside a scikit-learn `Pipeline`, so it is fitted separately in each fold and cannot leak information from the held-out fold.

The fixed validation partition provides a secondary confirmation. The selected model is then refitted on train plus validation data and evaluated once on the untouched test partition.

## Results

Logistic Regression was selected.

- Five-fold CV macro F1: **0.985 ± 0.012**
- Validation accuracy: **0.980**
- Validation macro F1: **0.977**
- Test accuracy: **0.981**
- Test macro F1: **0.975**
- Test errors: **1 of 52 observations**

The model clearly improves on the most-frequent baseline. In this dataset, the four physical measurements contain enough information to distinguish the three species accurately without using island or sex as predictors. The small test set means that the numerical estimate should still be interpreted cautiously.

## Error analysis and interpretation

The notebook includes:

- the per-class precision, recall, and F1 report;
- a test confusion matrix;
- the complete misclassified observation;
- descriptive accuracy by island and sex;
- permutation feature importance.

Island and sex are retained only as metadata. They are not model inputs, so subgroup inspection does not change the classifier.

## Unsupervised analysis

K-means with three clusters is fitted to scaled physical measurements without access to species labels. PCA is used only to visualize the geometry in two dimensions. After fitting, cluster assignments are compared with species using a contingency table, Adjusted Rand Index, and Normalized Mutual Information.

The clusters recover most, but not all, of the species structure: Adjusted Rand Index is **0.900** and Normalized Mutual Information is **0.866**. This supports the conclusion that morphology contains strong natural grouping information while supervised labels are still useful for resolving overlap between species.

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
uv sync --frozen
```

Generate the validated processed dataset and deterministic split assignments:

```powershell
uv run python -m penguins_project.train
```

Run the validation tests:

```powershell
uv run pytest -q
```

Start JupyterLab:

```powershell
uv run jupyter lab
```

Open `notebooks/02_modeling_complete.ipynb`, choose **Kernel → Restart Kernel and Run All Cells**, verify that no cell fails, and save the notebook so that all final tables and figures remain embedded.

## Reproducibility controls

- Random seed: `32`
- Deterministic stratified split stored in `split_assignments.parquet`
- Leakage-sensitive scaling inside a `Pipeline`
- Five-fold `StratifiedKFold` with shuffling and a fixed seed
- Test set evaluated only after model selection
- Dependency resolution recorded in `uv.lock`
- Input validation implemented with Pandera

## Limitations

- The dataset is small, so metric estimates can vary across splits.
- The results may not generalize beyond the observed Palmer Archipelago population.
- Two observations with no usable physical measurements are excluded.
- Some test subgroups contain few observations, so subgroup metrics are descriptive.
- Permutation importance is associational rather than causal.
- K-means assumes roughly spherical clusters and requires the number of clusters in advance.

## Artificial-intelligence use disclosure

ChatGPT was used as a learning and development assistant for explaining Pandera and Parquet, reviewing the deterministic split for leakage, debugging selected code fragments, suggesting a reproducible notebook structure, and helping draft or revise selected implementation and documentation sections. The methodological choices retained in the final project were checked against the course specification and verified by running the workflow, tests, metrics, and plots.

Representative prompts included:

- “Explain what each Penguins column represents and which Pandera checks are justified.”
- “Review this stratified train/validation/test split and identify possible leakage.”
- “Explain why macro F1 should accompany accuracy for these class frequencies.”
- “Show how to compare the candidate models with stratified cross-validation while keeping preprocessing inside a pipeline.”
- “Explain how K-means and PCA can be used as a separate unsupervised analysis without using species labels during fitting.”


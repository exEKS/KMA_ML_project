import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from penguins_project.data import validate_penguins

METADATA: tuple[str, ...] = (
    "island",
    "sex",
)

MODEL_FEATURES: tuple[str, ...] = (
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
)

TARGET_COLUMN = "species"
ID_COLUMN = "id"
SPLIT_COLUMN = 'split'
SEED = 32
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DF_PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "Penguins.parquet"
MODEL_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "penguins_model.parquet"
ASSIGNMENTS_PATH = PROJECT_ROOT / "data" / "processed" / "split_assignments.parquet"

def create_assignments(
        df: pd.DataFrame,
        seed: int = SEED,
) -> pd.DataFrame:
    required_columns = {ID_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"missing columns: {missing_columns}")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError("id was duplicated")

    ordered_df = (
        df[[ID_COLUMN, TARGET_COLUMN]]
        .sort_values(by=ID_COLUMN)
        .reset_index(drop=True)
    )

    train, test_and_validate = train_test_split(
        ordered_df,
        test_size=0.3,
        random_state=seed,
        stratify=ordered_df[TARGET_COLUMN],
    )

    validate, test = train_test_split(
        test_and_validate,
        test_size=0.5,
        random_state=seed,
        stratify=test_and_validate[TARGET_COLUMN],
    )

    assignments = pd.concat(
        [
            train[[ID_COLUMN]].assign(split="train"),
            test[[ID_COLUMN]].assign(split="test"),
            validate[[ID_COLUMN]].assign(split="validate"),
        ],
        ignore_index=True,
    )

    return assignments.sort_values(ID_COLUMN).reset_index(drop=True)


def save_assignments(
        df: pd.DataFrame,
        path: Path,
        seed: int=SEED,
) -> pd.DataFrame:

    assignments = create_assignments(df, seed)

    path.parent.mkdir(parents=True,exist_ok=True)
    assignments.to_parquet(path=path, index=False)

    return assignments

def create_numeric_features_table(
        df: pd.DataFrame,
        features: tuple[str, ...] = MODEL_FEATURES,
) -> pd.DataFrame:
    df = df[list(features)].copy()
    return df

def nan_finder_in_feature_table(
        df: pd.DataFrame,
) -> pd.DataFrame:
    table = create_numeric_features_table(df)
    bool_mask = table.isna().any(axis=1)
    return df[bool_mask]

def main() -> None:
    frame = pd.read_parquet(DF_PARQUET_PATH)
    frame = validate_penguins(frame).copy()

    frame.loc[:, ID_COLUMN] = frame.index

    features_table = create_numeric_features_table(df=frame)
    rows_with_missing_features = nan_finder_in_feature_table(frame)

    number_of_nan = features_table.isna().sum(axis=1)

    nan_rows_devide_by_all_rows = (
        len(rows_with_missing_features) / len(frame)
    )

    print(nan_rows_devide_by_all_rows * 100)
    print(number_of_nan[number_of_nan > 0])

    bool_mask_for_all_columns = (
        features_table.isna().all(axis=1)
    )

    empty_rows = frame.loc[bool_mask_for_all_columns]
    empty_row = empty_rows[ID_COLUMN].tolist()

    print(empty_rows)
    print(empty_row)

    model_df = frame.loc[~bool_mask_for_all_columns].copy()
    MODEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_df.to_parquet(path=MODEL_DATA_PATH, engine="pyarrow", index=False)

    assignments = save_assignments(df=model_df, path=ASSIGNMENTS_PATH)

    data_with_split = model_df.merge(
        assignments,
        on=ID_COLUMN,
        validate='one_to_one',
    )

    print("model_df shape:", model_df.shape)
    print(
        "remaining missing values:",
        model_df[list(MODEL_FEATURES)].isna().sum().sum(),
    )

    print(assignments[SPLIT_COLUMN].value_counts())

    split_species_counts = pd.crosstab(
        data_with_split[SPLIT_COLUMN],
        data_with_split[TARGET_COLUMN],
    )

    split_species_counts_normalized = pd.crosstab(
        data_with_split[SPLIT_COLUMN],
        data_with_split[TARGET_COLUMN],
        normalize='index'
    )

    print(
        "\n\n=========================assignments======================",
        3*'\n', f"\ndata_with_split:\t{data_with_split.shape}",
        3*'\n', f"duplicate:\t{data_with_split[ID_COLUMN].duplicated().any()}",
        3*'\n', f"isna().sum() :\n\t{data_with_split[SPLIT_COLUMN].isna().sum()}",
        3*'\n', f"values count: {data_with_split[SPLIT_COLUMN].value_counts()}",
        3*'\n', split_species_counts,
        3*'\n', split_species_counts_normalized,
    )
if __name__ == "__main__":
    main()
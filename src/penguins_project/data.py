import pandas as pd
import pandera as pa

from penguins_project.schema import PENGUINS_SCHEMA

@pa.check_input(PENGUINS_SCHEMA, lazy=True)
def validate_penguins(df: pd.DataFrame) -> pd.DataFrame:
    return df
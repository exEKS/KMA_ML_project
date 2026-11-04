import pandas as pd
import pytest
import pandera.pandas as pa
from penguins_project.data import validate_penguins


def test_valid_penguins_dataframe_passes_validation():
    df = pd.DataFrame(
        {
            "species": ["Adelie", "Gentoo", "Chinstrap"],
            "island": ["Torgersen", "Biscoe", "Dream"],
            "bill_length_mm": [39.1, 46.1, None],
            "bill_depth_mm": [18.7, 13.2, None],
            "flipper_length_mm": [181.0, 211.0, None],
            "body_mass_g": [3750.0, 4500.0, None],
            "sex": ["Male", "Female", None],
        }
    )

    validated_df = validate_penguins(df)

    assert validated_df.shape == (3, 7)
    assert list(validated_df.columns) == list(df.columns)

def test_unknown_species_fails_validation():
    df = pd.DataFrame({
        "species": ["Adelie", "Gentoo", "Dragon"],
        "island": ["Torgersen", "Biscoe", "Dream"],
        "bill_length_mm": [39.1, 46.1, None],
        "bill_depth_mm": [18.7, 13.2, None],
        "flipper_length_mm": [181.0, 211.0, None],
        "body_mass_g": [3750.0, 4500.0, None],
        "sex": ["Male", "Female", None],
    })
    with pytest.raises(pa.errors.SchemaErrors) as error:
        validate_penguins(df)
    assert "Dragon" in str(error.value)
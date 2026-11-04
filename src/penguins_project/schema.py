import pandera.pandas as pa


columns = {
    "species": pa.Column(
        pa.String,
        nullable=False,
        checks=pa.Check.isin(
            [
                "Adelie",
                "Chinstrap",
                "Gentoo",
            ]
        ),
    ),
    "island": pa.Column(
        pa.String,
        nullable=False,
        checks=pa.Check.isin(
            [
                "Biscoe",
                "Dream",
                "Torgersen",
            ]
        ),
    ),
    "bill_length_mm": pa.Column(
        pa.Float64,
        nullable=True,
        checks=pa.Check.gt(0),
    ),
    "bill_depth_mm": pa.Column(
        pa.Float64,
        nullable=True,
        checks=pa.Check.gt(0),
    ),
    "flipper_length_mm": pa.Column(
        pa.Float64,
        nullable=True,
        checks=pa.Check.gt(0),
    ),
    "body_mass_g": pa.Column(
        pa.Float64,
        nullable=True,
        checks=pa.Check.gt(0),
    ),
    "sex": pa.Column(
        pa.String,
        nullable=True,
        checks=pa.Check.isin(
            [
                "Male",
                "Female",
            ]
        ),
    ),
}


PENGUINS_SCHEMA = pa.DataFrameSchema(
    columns=columns,
    strict=True,
    ordered=False,
    coerce=False,
    unique_column_names=True,
    name="PenguinsSchema",
)
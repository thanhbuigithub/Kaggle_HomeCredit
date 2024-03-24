import polars as pl


# Set data type
def set_table_dtypes(df):
    for col in df.columns:
        if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
            df = df.with_columns(pl.col(col).cast(pl.Int64))
        elif col in ["date_decision"]:
            df = df.with_columns(pl.col(col).cast(pl.Date))
        elif col[-1] in ("P", "A"):
            df = df.with_columns(pl.col(col).cast(pl.Float64))
        elif col[-1] in ("M",):
            df = df.with_columns(pl.col(col).cast(pl.String))
        elif col[-1] in ("D",):
            df = df.with_columns(pl.col(col).cast(pl.Date))
    return df


# Handle dates
def handle_dates(df):
    for col in df.columns:
        # Extract month and weekday from date columns
        if col[-1] in ("D",):
            # Subtract the "date_decision" column from the column specified by the variable `col`
            df = df.with_columns(pl.col(col) - pl.col("date_decision"))
            # Convert the date column to total number of days since the first date
            df = df.with_columns(pl.col(col).dt.total_days())
    # Drop the "date_decision" and "MONTH" columns
    df = df.drop("date_decision", "MONTH")

    return df


# Filter columns
# Filter 1: Remove columns with more than 95% missing values
# Filter 2: Remove columns with only one unique value or more than 200 unique values
def filter_cols(df):
    for col in df.columns:
        if col not in ["target", "case_id", "WEEK_NUM"]:
            isnull = df[col].is_null().mean()

            if isnull > 0.95:
                df = df.drop(col)

    for col in df.columns:
        if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
            freq = df[col].n_unique()

            if (freq == 1) | (freq > 200):
                df = df.drop(col)

    return df

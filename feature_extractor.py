import polars as pl
from glob import glob
import pipeline as pipeline
import aggregator as aggregator


# Read a single file and perform preprocessing
def read_file(path, depth=None):
    # Use Polars to read the parquet file
    df = pl.read_parquet(path)
    # Set the data types of the columns
    df = df.pipe(pipeline.set_table_dtypes)
    # If depth is 1 or 2, group by case_id and aggregate the columns
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(aggregator.get_exprs(df))
    return df


# Read multiple files and perform preprocessing
def read_files(regex_path, depth=None):
    chunks = []
    # Use glob to find all files that match the regex pattern
    for path in glob(str(regex_path)):
        # Read the parquet file using Polars and set the data types
        chunks.append(pl.read_parquet(path).pipe(pipeline.set_table_dtypes))
    # Concatenate the data frames vertically
    df = pl.concat(chunks, how="vertical_relaxed")
    # If depth is 1 or 2, group by case_id and aggregate the columns
    if depth in [1, 2]:
        df = df.group_by("case_id").agg(aggregator.get_exprs(df))
    return df


# Feature engineering function to join the data frames and create new features
def feature_eng(df_base, depth_0, depth_1, depth_2):
    # Extract the month and weekday from the date_decision column
    df_base = (
        df_base
        .with_columns(
            month_decision=pl.col("date_decision").dt.month(),
            weekday_decision=pl.col("date_decision").dt.weekday(),
        )
    )
    # Join the data frames on the case_id column
    for i, df in enumerate(depth_0 + depth_1 + depth_2):
        df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
    # Handle missing values
    df_base = df_base.pipe(pipeline.handle_dates)
    return df_base


# Convert Polars DataFrame to Pandas DataFrame and handle categorical columns
def to_pandas(df_data, cat_cols=None):
    df_data = df_data.to_pandas()
    if cat_cols is None:
        cat_cols = list(df_data.select_dtypes("object").columns)
    df_data[cat_cols] = df_data[cat_cols].astype("category")
    return df_data, cat_cols

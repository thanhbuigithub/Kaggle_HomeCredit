import polars as pl


# Generate max aggregation expression for numerical columns
def num_expr(df):
    cols = [col for col in df.columns if col[-1] in ("P", "A")]
    expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
    return expr_max


# Generate max aggregation expression for date columns
def date_expr(df):
    cols = [col for col in df.columns if col[-1] in ("D",)]
    expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
    return expr_max


# Generate max aggregation expression for string columns
def str_expr(df):
    cols = [col for col in df.columns if col[-1] in ("M",)]
    expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
    return expr_max


# Generate max aggregation expression for other columns
def other_expr(df):
    cols = [col for col in df.columns if col[-1] in ("T", "L")]
    expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
    return expr_max


# Generate max aggregation expression for count columns
def count_expr(df):
    cols = [col for col in df.columns if "num_group" in col]
    expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
    return expr_max


# Generate all aggregation expressions
def get_exprs(df):
    exprs = num_expr(df) + \
            date_expr(df) + \
            str_expr(df) + \
            other_expr(df) + \
            count_expr(df)
    return exprs

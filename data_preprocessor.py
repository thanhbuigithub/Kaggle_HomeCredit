import polars as pl
from feature_extractor import read_file, read_files, feature_eng, to_pandas
import pipeline as pipeline


class DataPreprocessor:
    def __init__(self, train_dir, test_dir, saved_dir, re_features_extracted=True):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.saved_dir = saved_dir
        self.re_features_extracted = re_features_extracted
        self.saved_df_train = saved_dir / "df_train.parquet"
        self.saved_df_test = saved_dir / "df_test.parquet"

        self.df_train = None
        self.df_test = None

    def create_train(self):
        if self.re_features_extracted or not self.saved_df_train.exists():
            print("Create features training data ...")
            # Read the training data
            data_store = {
                "df_base": read_file(self.train_dir / "train_base.parquet"),
                "depth_0": [
                    read_file(self.train_dir / "train_static_cb_0.parquet"),
                    read_files(self.train_dir / "train_static_0_*.parquet"),
                ],
                "depth_1": [
                    read_files(self.train_dir / "train_applprev_1_*.parquet", 1),
                    read_file(self.train_dir / "train_tax_registry_a_1.parquet", 1),
                    read_file(self.train_dir / "train_tax_registry_b_1.parquet", 1),
                    read_file(self.train_dir / "train_tax_registry_c_1.parquet", 1),
                    read_file(self.train_dir / "train_credit_bureau_b_1.parquet", 1),
                    read_file(self.train_dir / "train_other_1.parquet", 1),
                    read_file(self.train_dir / "train_person_1.parquet", 1),
                    read_file(self.train_dir / "train_deposit_1.parquet", 1),
                    read_file(self.train_dir / "train_debitcard_1.parquet", 1),
                ],
                "depth_2": [
                    read_file(self.train_dir / "train_credit_bureau_b_2.parquet", 2),
                ]
            }
            # Feature engineering on the training data
            df_train = feature_eng(**data_store)
            # Save the training data
            df_train.write_parquet(self.saved_df_train)
        else:
            print("Reading the saved features training data ...")
            df_train = pl.read_parquet(self.saved_df_train)

        print("Train data shape:\t", df_train.shape)
        return df_train
    
    def create_test(self):
        if self.re_features_extracted or not self.saved_df_test.exists():
            print("Create features test data ...")
            # Read the test data
            data_store = {
                "df_base": read_file(self.test_dir / "test_base.parquet"),
                "depth_0": [
                    read_file(self.test_dir / "test_static_cb_0.parquet"),
                    read_files(self.test_dir / "test_static_0_*.parquet"),
                ],
                "depth_1": [
                    read_files(self.test_dir / "test_applprev_1_*.parquet", 1),
                    read_file(self.test_dir / "test_tax_registry_a_1.parquet", 1),
                    read_file(self.test_dir / "test_tax_registry_b_1.parquet", 1),
                    read_file(self.test_dir / "test_tax_registry_c_1.parquet", 1),
                    read_file(self.test_dir / "test_credit_bureau_b_1.parquet", 1),
                    read_file(self.test_dir / "test_other_1.parquet", 1),
                    read_file(self.test_dir / "test_person_1.parquet", 1),
                    read_file(self.test_dir / "test_deposit_1.parquet", 1),
                    read_file(self.test_dir / "test_debitcard_1.parquet", 1),
                ],
                "depth_2": [
                    read_file(self.test_dir / "test_credit_bureau_b_2.parquet", 2),
                ]
            }

            # Feature engineering on the test data
            df_test = feature_eng(**data_store)
            # Save the test data
            df_test.write_parquet(self.saved_df_test)
        else:
            print("Reading the saved features test data ...")
            df_test = pl.read_parquet(self.saved_df_test)

        print("Test data shape:\t", df_test.shape)
        return df_test

    def preprocess(self):
        # Create training and test dataframes
        print("--- Create dataframes:")
        df_train = self.create_train()
        df_test = self.create_test()
        print("")

        print("--- Filter columns:")
        # Filter columns
        df_train = df_train.pipe(pipeline.filter_cols)
        df_test = df_test.select([col for col in df_train.columns if col != "target"])
        print("Train data shape:\t", df_train.shape)
        print("Test data shape:\t", df_test.shape)
        print("")

        # convert to pandas
        df_train, cat_cols = to_pandas(df_train)
        df_test, cat_cols = to_pandas(df_test, cat_cols)

        X_train = df_train.drop(columns=["target", "case_id", "WEEK_NUM"])
        y_train = df_train["target"]
        weeks = df_train["WEEK_NUM"]

        X_test = df_test.drop(columns=["WEEK_NUM"])
        X_test = X_test.set_index("case_id")

        return X_train, y_train, X_test, weeks

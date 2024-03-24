import pandas as pd


# Make submission
def make_submission(sample_submission_path, predictions, output_path):
    print("--- Making submission ...")
    df_subm = pd.read_csv(sample_submission_path)
    df_subm = df_subm.set_index("case_id")

    df_subm["score"] = predictions
    print("Check null: ", df_subm["score"].isnull().any())
    df_subm.to_csv(output_path)
    print("Submission created. Check the file: ", output_path)
    return df_subm

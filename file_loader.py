
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Get the file extension and convert to lowercase
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Dictionary mapping extensions to pandas read functions
    readers = {
        ".csv": pd.read_csv,
        ".txt": pd.read_csv,  # Assuming txt is comma/tab separated
        ".tsv": lambda f: pd.read_csv(f, sep="\t"),
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".pkl": pd.read_pickle,
        ".feather": pd.read_feather,
        ".h5": pd.read_hdf,
        ".hdf5": pd.read_hdf,
        ".html": lambda f: pd.read_html(f)[0],  # read_html returns a list of dfs
    }

    if file_extension in readers:
        print(f"Reading file with extension: {file_extension}")
        return readers[file_extension](file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


# --- Example Usage ---
# file_path = "data.csv"
# df = read_uploaded_file(file_path)
# print(df.head())

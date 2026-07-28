import pandas as pd


def perform_eda(df: pd.DataFrame):
    """Performs basic Exploratory Data Analysis (EDA) on a pandas DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataframe to analyze.
    """
    print("=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("=" * 60)
    print("2. FIRST 5 ROWS")
    print("=" * 60)
    display(df.head())
    print("\n")

    print("=" * 60)
    print("3. COLUMN DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    print("\n")

    print("=" * 60)
    print("4. MISSING VALUES ANALYSIS")
    print("=" * 60)
    missing_data = pd.DataFrame(
        {
            "Missing Count": df.isnull().sum(),
            "Missing Percentage (%)": (
                df.isnull().sum() / len(df) * 100
            ).round(2),
        }
    )
    # Filter to show only columns with missing values, sorted highest first
    missing_data = missing_data[missing_data["Missing Count"] > 0].sort_values(
        by="Missing Count", ascending=False
    )

    if missing_data.empty:
        print("Great news! There are no missing values in this dataset.\n")
    else:
        print(missing_data)
        print("\n")

    print("=" * 60)
    print("5. DUPLICATE ROWS")
    print("=" * 60)
    duplicates = df.duplicated().sum()
    print(
        f"Number of duplicate rows: {duplicates} ({(duplicates / len(df) * 100):.2f}% of total rows)\n"
    )

    print("=" * 60)
    print("6. NUMERICAL SUMMARY (STATISTICS)")
    print("=" * 60)
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        display(num_df.describe().T)
    else:
        print("No numerical columns found in the dataset.")
    print("\n")

    print("=" * 60)
    print("7. CATEGORICAL SUMMARY")
    print("=" * 60)
    cat_df = df.select_dtypes(include=["object", "category"])
    if not cat_df.empty:
        display(cat_df.describe().T)
    else:
        print("No categorical columns found in the dataset.")
    print("\n")

    print("=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)

def analyze_dataset(df):

    print("\n--- DATASET INFORMATION ---")
    print("Shape:", df.shape)

    print("\n--- COLUMN DATA TYPES ---")
    print(df.dtypes)

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(df.duplicated().sum())

    print("\n--- NUMERICAL SUMMARY ---")
    print(df.describe())

    print("\n--- CATEGORICAL COLUMN DISTRIBUTIONS ---")

    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:
        print(f"\n{column}")
        print(df[column].value_counts())

import pandas as pd

def merge_excel_by_cas_with_columns(file1_path, file2_path, output_path):
    """
    Merge two Excel files based on the 'compound' (CAS number) column,
    appending columns from the second file starting from the sixth column onward.

    :param file1_path: Path to the first Excel file
    :param file2_path: Path to the second Excel file
    :param output_path: Path to save the merged Excel file
    """
    # Read both Excel files
    df1 = pd.read_excel(file1_path)
    df2 = pd.read_excel(file2_path)

    # Merge the two DataFrames on 'compound' (left join keeps all rows from df1)
    merged_df = pd.merge(df1, df2, on='compound', how='left')

    # Select columns from the 6th column onward (index 5 and beyond)
    columns_to_add = merged_df.columns[5:]  # Keep all columns starting from column 6

    # Keep the first five columns from df1
    df1 = df1[['compound'] + df1.columns[1:5].tolist()]

    # Concatenate df1 with the selected new columns from merged_df
    df1 = pd.concat([df1, merged_df[columns_to_add]], axis=1)

    # Save the merged DataFrame to a new Excel file
    df1.to_excel(output_path, index=False)
    print(f"Merged data successfully saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    file1 = "# Path to the first Excel file"
    file2 = "# Path to the second Excel file"
    output_file = "# Path to save the merged file"
    merge_excel_by_cas_with_columns(file1, file2, output_file)

import pandas as pd

def filter_log_kd_by_3sigma(file_path, output_path, log_kd_column_index=2, id_column_index=0):
    """
    Filter outliers in the Log Kd column of an Excel file based on the μ ± 3σ rule,
    and save the cleaned dataset to a new Excel file.

    Prints the number of rows removed, total dimensions before and after filtering,
    and details of removed records.

    :param file_path: Path to the input Excel file
    :param output_path: Path to save the filtered Excel file
    :param log_kd_column_index: Index of the Log Kd column (default = 2, i.e., the 3rd column)
    :param id_column_index: Index of the identifier/name column (default = 0, i.e., the 1st column)
    """
    # Read Excel file
    df = pd.read_excel(file_path)

    # Extract Log Kd column
    log_kd_col = df.iloc[:, log_kd_column_index]

    # Calculate mean and standard deviation
    mean = log_kd_col.mean()
    std = log_kd_col.std()

    # Compute μ ± 3σ bounds
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    # Store original shape
    original_shape = df.shape

    # Identify outliers
    removed_df = df[(log_kd_col < lower_bound) | (log_kd_col > upper_bound)]

    # Retain values within μ ± 3σ
    filtered_df = df[(log_kd_col >= lower_bound) & (log_kd_col <= upper_bound)]

    # Save filtered data
    filtered_df.to_excel(output_path, index=False)

    # Print summary
    print(f"✅ Filtered data saved to: {output_path}")
    print(f"Original data shape: {original_shape}")
    print(f"Filtered data shape: {filtered_df.shape}")
    print(f"Rows removed: {removed_df.shape[0]}")
    print("Columns removed: 0 (none processed)")

    # Display details of removed records
    if not removed_df.empty:
        print("\n🧾 Removed rows (ID and corresponding Log Kd values):")
        for idx, row in removed_df.iterrows():
            name = row.iloc[id_column_index]
            log_kd = row.iloc[log_kd_column_index]
            print(f"- {name}: {log_kd}")
    else:
        print("No outliers were removed.")

# Example usage
if __name__ == "__main__":
    input_file = "# Input file path"
    output_file = "# Output file path"
    filter_log_kd_by_3sigma(input_file, output_file)

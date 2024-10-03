import pandas as pd
import re

def fill_blanks_from_excel(file_path, sheet_name=None, cell_range=None, dtypes=None):
    # Parse the cell range (e.g., "B21:D45")
    if cell_range:
        # Split the range into start and end points
        start_range, end_range = cell_range.split(':')

        # Use regex to split into column letters and row numbers
        start_match = re.match(r"([A-Za-z]+)(\d+)", start_range)
        end_match = re.match(r"([A-Za-z]+)(\d+)", end_range)

        if not start_match or not end_match:
            raise ValueError("Invalid cell range format")

        start_col, start_row = start_match.groups()
        end_col, end_row = end_match.groups()
        start_row = int(start_row)
        end_row = int(end_row)

        # Determine the columns to use based on letters
        usecols = f"{start_col}:{end_col}"
        skiprows = start_row - 1  # Adjust to get the correct row for headers (subtract 1 for zero-indexing and 1 for headers)
        nrows = end_row - start_row + 1  # Number of rows to read

        # Load the specific region into a DataFrame
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            usecols=usecols,
            skiprows=skiprows,
            nrows=nrows,
            header=0,
        )

        # Fill blanks with the closest earlier row value in the same column
        df_filled = df.ffill(axis=0)

        # Apply specific data types to columns if provided
        if dtypes:
            df_filled = df_filled.astype(dtypes)

        df_filled.columns = df_filled.columns.str.replace(r'\.\d+', '', regex=True)

    return df_filled
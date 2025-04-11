import os
import re


def extract_stock_names(directory="."):
    """
    Scans the specified directory for files matching the pattern:
    <stock_name>_data_aggregated.csv
    Extracts and returns a list of stock names.

    Parameters:
        directory (str): Path to the directory to scan. Defaults to current directory.

    Returns:
        list: A list of extracted stock names.
    """
    # List all files in the directory
    files = os.listdir(directory)
    stock_names = []

    # Compile regex to match files like "adobe_data_aggregated.csv"
    pattern = re.compile(r"^(.*?)_data_aggregated\.csv$", re.IGNORECASE)

    for file in files:
        match = pattern.match(file)
        if match:
            # Extract the stock name (everything before '_data_aggregated.csv')
            stock_name = match.group(1)
            stock_names.append(stock_name)

    return stock_names


if __name__ == "__main__":
    # You can change this to any directory you need to scan
    directory_to_scan = "."

    stocks = extract_stock_names(directory_to_scan)
    print("Extracted stock names:")
    for stock in stocks:
        print(f"python sentiment_1.py {stock}_data_aggregated.csv {stock}_sentiment_scores.csv")

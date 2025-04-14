import csv
import re

input_file = 'cnbc_headlines.csv'
output_file = 'cnbc_headlines_cleaned.csv'

with open(input_file, 'r', encoding='utf-8', newline='') as infile, \
     open(output_file, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        # Clean each cell in the row
        cleaned_row = []
        for cell in row:
            text = cell.replace('\n', ' ').replace('\t', ' ')
            text = re.sub(r'\s{2,}', ' ', text).strip()
            cleaned_row.append(text)

        # Check if entire row is empty (all cells are blank)
        if all(cell == '' for cell in cleaned_row):
            continue  # skip empty row

        writer.writerow(cleaned_row)

print(f"Cleaned CSV saved as '{output_file}'")

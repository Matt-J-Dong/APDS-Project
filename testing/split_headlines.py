import csv
from datetime import datetime
from collections import defaultdict

# Input files
headlines_file = 'headlines.csv'
availability_file = 'is_available.csv'

# Load availability data into a dictionary
availability = defaultdict(set)

with open(availability_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        date = row['Date']
        for publication, available in row.items():
            if publication != 'Date' and available.lower() == 'true':
                availability[publication].add(date)

# Prepare writers for each publication
writers = {}
files = {}

# Read and split headlines
with open(headlines_file, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        pub = row['Publication']
        raw_date = row['Date']

        # Check if article is available for this publisher and date
        if raw_date in availability.get(pub, set()):
            # Convert date to UTC Zulu time (midnight)
            try:
                dt = datetime.strptime(raw_date, "%Y%m%d")
                row['Date'] = dt.strftime("%Y-%m-%dT00:00:00Z")
            except ValueError:
                continue  # skip bad date format

            # Open output file and writer if not already open
            if pub not in writers:
                outfile = open(f'{pub.replace(" ", "_")}_headlines.csv', 'w', encoding='utf-8', newline='')
                writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                writer.writeheader()
                writers[pub] = writer
                files[pub] = outfile

            writers[pub].writerow(row)

# Close all open files
for f in files.values():
    f.close()

print("✅ Headlines split by publisher with Zulu time formatting and availability filtering.")

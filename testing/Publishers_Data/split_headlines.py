import csv
from datetime import datetime
from collections import defaultdict

# Input files
headlines_file = 'headlines.csv'
availability_file = 'is_available.csv'

# Load availability data
availability = defaultdict(set)
with open(availability_file, "r", encoding="utf-8") as af:
    reader = csv.DictReader(af)
    for row in reader:
        date = row['Date']
        for pub, is_available in row.items():
            if pub != "Date" and is_available.lower() == "true":
                availability[pub].add(date)

# Writers and file handles for each publication
writers = {}
files = {}

# Read and filter headlines
with open(headlines_file, "r", encoding="utf-8") as hf:
    reader = csv.DictReader(hf)

    # Remove URL from fieldnames
    output_fields = [f for f in reader.fieldnames if f != "URL"]

    for row in reader:
        pub = row['Publication']
        raw_date = row['Date']

        # Skip if article isn't available
        if raw_date not in availability.get(pub, set()):
            continue

        # Convert Date to Zulu time
        try:
            dt = datetime.strptime(raw_date, "%Y%m%d")
            row["Date"] = dt.strftime("%Y-%m-%dT00:00:00Z")
        except ValueError:
            continue  # Skip if date is invalid

        # Prepare writer if not yet open
        if pub not in writers:
            out_filename = f"{pub.replace(' ', '_')}_headlines.csv"
            f = open(out_filename, "w", encoding="utf-8", newline="")
            writer = csv.DictWriter(f, fieldnames=output_fields)
            writer.writeheader()
            writers[pub] = writer
            files[pub] = f

        # Write row without URL
        filtered_row = {k: v for k, v in row.items() if k in output_fields}
        writers[pub].writerow(filtered_row)

# Close all output files
for f in files.values():
    f.close()

print("✅ Files split by publication, URL removed, dates converted to UTC Zulu time.")

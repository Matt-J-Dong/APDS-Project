import csv
from datetime import datetime
from dateutil import parser as date_parser

# Custom publisher files
publishers = [
    "New York Times",
    "CNN",
    "FOX",
    "New York Post",
    "BBC",
    "Washington Post",
    "USA Today",
    "Daily Mail",
    "CNBC",
    "The Guardian",
]


# File format: e.g., CNN_headlines.csv
def publisher_filename(pub_name):
    return f"{pub_name.replace(' ', '_')}_headlines.csv"


# Add base files with special formatting
file_configs = {
    "abcnews-date-text.csv": {
        "date_field": "publish_date",
        "headline_field": "headline_text",
    },
    "reuters_headlines.csv": {"date_field": "Time", "headline_field": "Headlines"},
    "guardian_headlines_cleaned.csv": {
        "date_field": "Time",
        "headline_field": "Headlines",
    },
    "cnbc_headlines_cleaned.csv": {"date_field": "Time", "headline_field": "Headlines"},
    "WorldNewsData.csv": {
        "date_field": "Date",
        "headline_field": [f"Top{i}" for i in range(1, 26)],
    },
}

# Add publisher files dynamically
for pub in publishers:
    file_configs[publisher_filename(pub)] = {
        "date_field": "Date",
        "headline_field": "Headline",
    }

# Keyword to search
keyword = "samsung"
results = []

def convert_to_utc(date_str, source_file):
    try:
        if source_file == "abcnews-date-text.csv":
            dt = datetime.strptime(date_str.strip(), "%Y%m%d")
        else:
            dt = date_parser.parse(date_str.strip(), fuzzy=True)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return f"InvalidDate({date_str})"

# Search files
for file_name, config in file_configs.items():
    print(f"\n🔍 Searching in {file_name}...\n")
    last_headline = None  # Reset for each file
    try:
        with open(file_name, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_date = row.get(config["date_field"], "").strip()
                utc_time = convert_to_utc(raw_date, file_name)

                if isinstance(config["headline_field"], list):
                    for field in config["headline_field"]:
                        headline = row.get(field, "").strip()
                        if not headline or headline == last_headline:
                            continue
                        if keyword.lower() in headline.lower():
                            # print(f"{utc_time}: {headline}")
                            results.append(
                                {"Time Data": utc_time, "Headline": headline}
                            )
                        last_headline = headline
                else:
                    headline = row.get(config["headline_field"], "").strip()
                    if not headline or headline == last_headline:
                        continue
                    if keyword.lower() in headline.lower():
                        # print(f"{utc_time}: {headline}")
                        results.append({"Time Data": utc_time, "Headline": headline})
                    last_headline = headline
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_name}")

# Save results
output_file = f"{keyword.lower()}_data_aggregated.csv"
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["Time Data", "Headline"])
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Aggregated data saved to: {output_file}")

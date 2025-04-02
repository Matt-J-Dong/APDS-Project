import csv
import os
from datetime import datetime
from dateutil import parser as date_parser
from collections import defaultdict
import pytz

# Keyword to filter
keyword = "samsung"

# Publishers
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

# Folder paths
google_news_folder = "Google_News_Data"
publishers_folder = "Publishers_Data"
base_data_folder = "Older_Data"

# Define publisher file naming
def publisher_filename(pub_name):
    return f"{pub_name.replace(' ', '_')}_headlines.csv"

# Base files now under Older_Data
file_configs = {
    os.path.join(base_data_folder, "abcnews-date-text.csv"): {
        "date_field": "publish_date",
        "headline_field": "headline_text",
    },
    os.path.join(base_data_folder, "reuters_headlines.csv"): {
        "date_field": "Time",
        "headline_field": "Headlines",
    },
    os.path.join(base_data_folder, "guardian_headlines_cleaned.csv"): {
        "date_field": "Time",
        "headline_field": "Headlines",
    },
    os.path.join(base_data_folder, "cnbc_headlines_cleaned.csv"): {
        "date_field": "Time",
        "headline_field": "Headlines",
    },
    os.path.join(base_data_folder, "WorldNewsData.csv"): {
        "date_field": "Date",
        "headline_field": [f"Top{i}" for i in range(1, 26)],
    },
}

# Add all publisher-specific files (now in Publishers_Data)
for pub in publishers:
    file_configs[os.path.join(publishers_folder, publisher_filename(pub))] = {
        "date_field": "Date",
        "headline_field": "Headline",
    }

# Add headlines.csv base file (also now in Publishers_Data)
file_configs[os.path.join(publishers_folder, "headlines.csv")] = {
    "date_field": "Date",
    "headline_field": "Headline",
}

# Add Google News monthly files
for fname in os.listdir(google_news_folder):
    if fname.endswith(".csv"):
        file_configs[os.path.join(google_news_folder, fname)] = {
            "date_field": "DateTime",
            "headline_field": "Title",
        }

# Timezone handling
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc
tzinfos = {"ET": eastern}

# Store results and monthly counts
results = []
monthly_counter = defaultdict(int)

# Convert to UTC
def convert_to_utc(date_str, source_file):
    try:
        if source_file.endswith("abcnews-date-text.csv"):
            dt = datetime.strptime(date_str.strip(), "%Y%m%d")
            dt_utc = dt.replace(tzinfo=utc)
        else:
            dt_local = date_parser.parse(date_str.strip(), fuzzy=True, tzinfos=tzinfos)
            dt_utc = dt_local.astimezone(utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), dt_utc
    except Exception:
        return f"InvalidDate({date_str})", None

# Process all configured files
for file_name, config in file_configs.items():
    print(f"\n🔍 Searching in {file_name}...\n")
    last_headline = None
    try:
        with open(file_name, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                raw_date = row.get(config["date_field"], "").strip()
                utc_str, dt_obj = convert_to_utc(raw_date, file_name)
                if not dt_obj:
                    continue

                if isinstance(config["headline_field"], list):
                    for field in config["headline_field"]:
                        headline = row.get(field, "").strip()
                        if not headline or headline == last_headline:
                            continue
                        if keyword.lower() in headline.lower():
                            results.append({"Time Data": utc_str, "Headline": headline})
                            monthly_counter[(dt_obj.year, dt_obj.month)] += 1
                        last_headline = headline
                else:
                    headline = row.get(config["headline_field"], "").strip()
                    if not headline or headline == last_headline:
                        continue
                    if keyword.lower() in headline.lower():
                        results.append({"Time Data": utc_str, "Headline": headline})
                        monthly_counter[(dt_obj.year, dt_obj.month)] += 1
                    last_headline = headline
    except FileNotFoundError:
        print(f"⚠️ File not found: {file_name}")

# Sort results by time
results.sort(key=lambda x: x["Time Data"])

# Save headline results
output_file = f"{keyword.lower()}_data_aggregated.csv"
with open(output_file, "w", encoding="utf-8", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=["Time Data", "Headline"])
    writer.writeheader()
    writer.writerows(results)

# Save monthly headline counts
count_file = f"{keyword.lower()}_monthly_headline_count.csv"
with open(count_file, "w", encoding="utf-8", newline="") as countfile:
    writer = csv.writer(countfile)
    writer.writerow(["Year", "Month", "Headline Count"])
    for (year, month), count in sorted(monthly_counter.items()):
        writer.writerow([year, month, count])

# Console output
print(f"\n📅 Monthly headline count for keyword '{keyword}':\n")
for (year, month), count in sorted(monthly_counter.items()):
    print(f"{year}-{month:02d}: {count} headline(s)")

print(f"\n✅ Aggregated headline data saved to: {output_file}")
print(f"✅ Monthly count saved to: {count_file}")

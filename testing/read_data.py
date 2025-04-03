import csv
import os
from datetime import (
    datetime,
    timedelta,
)  # --- Changed: Added timedelta import for missing dates analysis.
from dateutil import parser as date_parser
from collections import defaultdict
import pytz
import glob

# Keyword to filter
# --- Changed: Removed the old single keyword variable.
# keyword = "samsung"

# --- Added: New keyword groups mapping to include multiple search terms (ticker symbols, aliases, etc.).
keyword_groups = {
    "samsung": ["samsung", "ssnlf"],  # For Samsung, include ticker "SSNLF"
    "google": [
        "google",
        "alphabet",
        "goog",
        "googl",
    ],  # For Google, include multiple related terms
}

# --- Removed selected_group and search_terms variables since we now iterate over each group.
# selected_group = "samsung"  # Change to desired group (e.g., "samsung" or "google")
# search_terms = keyword_groups[selected_group]

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
    "The Guardian",
]

# Folder paths
google_news_folder = "Google_News_Data"
publishers_folder = "Publishers_Data"
base_data_folder = "Older_Data"
analyst_data_folder = "Analyst_Data"

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

# # Add headlines.csv base file (also now in Publishers_Data)
# file_configs[os.path.join(publishers_folder, "headlines.csv")] = {
#     "date_field": "Date",
#     "headline_field": "Headline",
# }

# Add Google News monthly files
for fname in os.listdir(google_news_folder):
    if fname.endswith(".csv"):
        file_configs[os.path.join(google_news_folder, fname)] = {
            "date_field": "DateTime",
            "headline_field": "Title",
        }

# Add analyst data files
for analyst_file in glob.glob(
    os.path.join(analyst_data_folder, "analyst_ratings_processed_cleaned_*.csv")
):
    file_configs[analyst_file] = {
        "date_field": "date",
        "headline_field": "title",
    }

for partner_file in glob.glob(
    os.path.join(analyst_data_folder, "partner_headlines_cleaned_*.csv")
):
    file_configs[partner_file] = {
        "date_field": "date",
        "headline_field": "headline",
    }

# Timezone handling
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc
tzinfos = {"ET": eastern}

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


# --- Changed: Iterate over each keyword group to create individual file groups.
for group, search_terms in keyword_groups.items():
    print(
        f"\n===== Processing group: {group} =====\n"
    )  # --- Added: Display current group being processed.

    # Store results and monthly counts for current group
    results = []
    monthly_counter = defaultdict(int)
    seen = set()  # To remove duplicates (Zulu time + headline)
    seen_monthly_titles = defaultdict(set)  # To remove title dupes per month

    # Process all configured files for current group
    for file_name, config in file_configs.items():
        print(
            f"\n🔍 Searching in {file_name} for group '{group}'...\n"
        )  # --- Added: Indicate group in search message.
        last_headline = None
        try:
            with open(file_name, newline="", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    raw_date = row.get(config["date_field"], "").strip()
                    utc_str, dt_obj = convert_to_utc(raw_date, file_name)
                    if not dt_obj:
                        continue

                    year_month = (dt_obj.year, dt_obj.month)

                    if isinstance(config["headline_field"], list):
                        for field in config["headline_field"]:
                            headline = row.get(field, "").strip()
                            if not headline or headline == last_headline:
                                continue
                            dedup_key = (utc_str, headline)
                            # --- Changed: Use any() with search_terms list instead of a single keyword check.
                            if (
                                any(
                                    term.lower() in headline.lower()
                                    for term in search_terms
                                )  # Check multiple search terms
                                and dedup_key not in seen
                                and headline not in seen_monthly_titles[year_month]
                            ):
                                results.append(
                                    {"Time Data": utc_str, "Headline": headline}
                                )
                                monthly_counter[year_month] += 1
                                seen.add(dedup_key)
                                seen_monthly_titles[year_month].add(headline)
                            last_headline = headline
                    else:
                        headline = row.get(config["headline_field"], "").strip()
                        if not headline or headline == last_headline:
                            continue
                        dedup_key = (utc_str, headline)
                        # --- Changed: Use any() with search_terms list instead of a single keyword check.
                        if (
                            any(
                                term.lower() in headline.lower()
                                for term in search_terms
                            )  # Check multiple search terms
                            and dedup_key not in seen
                            and headline not in seen_monthly_titles[year_month]
                        ):
                            results.append({"Time Data": utc_str, "Headline": headline})
                            monthly_counter[year_month] += 1
                            seen.add(dedup_key)
                            seen_monthly_titles[year_month].add(headline)
                        last_headline = headline
        except FileNotFoundError:
            print(f"⚠️ File not found: {file_name}")

    # Sort results by time for current group
    results.sort(key=lambda x: x["Time Data"])

    # Save headline results for current group
    output_file = f"{group.lower()}_data_aggregated.csv"  # --- Changed: Use group key for filename.
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["Time Data", "Headline"])
        writer.writeheader()
        writer.writerows(results)

    # Save monthly headline counts for current group
    count_file = f"{group.lower()}_monthly_headline_count.csv"  # --- Changed: Use group key for filename.
    with open(count_file, "w", encoding="utf-8", newline="") as countfile:
        writer = csv.writer(countfile)
        writer.writerow(["Year", "Month", "Headline Count"])
        for (year, month), count in sorted(monthly_counter.items()):
            writer.writerow([year, month, count])

    # Console output for monthly counts
    print(f"\n📅 Monthly headline count for group '{group}':\n")
    for (year, month), count in sorted(monthly_counter.items()):
        print(f"{year}-{month:02d}: {count} headline(s)")

    print(f"\n✅ Aggregated headline data saved to: {output_file}")
    print(f"✅ Monthly count saved to: {count_file}")

    # --- Added: Section to analyze aggregated CSV and determine missing dates for current group.
    with open(output_file, newline="", encoding="utf-8") as agg_file:
        reader = csv.DictReader(agg_file)
        dates_present = set()
        for row in reader:
            time_str = row["Time Data"]
            # --- Changed: Parse the date portion from the timestamp.
            try:
                dt = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")
                date_only = dt.date()
                dates_present.add(date_only)
            except Exception:
                continue

    if dates_present:
        start_date = min(dates_present)
        end_date = max(dates_present)
        # --- Added: Generate all dates in range and check for missing ones.
        missing_dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date not in dates_present:
                missing_dates.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
    else:
        missing_dates = []

    # --- Added: Save missing dates to CSV file for current group.
    missing_dates_file = (
        f"{group.lower()}_missing_dates.csv"  # --- Changed: Use group key for filename.
    )
    with open(missing_dates_file, "w", newline="", encoding="utf-8") as md_file:
        writer = csv.writer(md_file)
        writer.writerow(["Missing Date"])
        for d in missing_dates:
            writer.writerow([d])

    print(f"\n✅ Missing dates saved to: {missing_dates_file}")

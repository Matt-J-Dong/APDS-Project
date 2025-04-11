import csv
import os
from datetime import (
    datetime,
    timedelta,
    timezone,
)  # --- Changed: Added timedelta and timezone imports for missing dates analysis and NYT date conversion.
from dateutil import parser as date_parser
from collections import defaultdict
import pytz
import glob
import requests  # --- Added: Import requests for NYT API calls.
from dotenv import (
    load_dotenv,
)  # --- Added: Import dotenv to load environment variables.
import time  # --- Added: Import time for rate limit handling.
import calendar  # --- Added: Import calendar to compute month end dates.
import concurrent.futures  # --- Added: Import for parallelization

load_dotenv()  # Load environment variables (e.g., NYT_API_KEY)

# --- Added: New keyword groups mapping for companies and tickers.
keyword_groups = {
    "apple": ["apple", "aapl"],
    "microsoft": ["microsoft", "msft"],
    "nvidia": ["nvidia", "nvda"],
    "amazon": ["amazon", "amzn"],
    "google": ["alphabet", "google", "goog", "googl"],
    "meta": ["meta", "facebook", "meta platforms", "meta platforms (facebook)"],
    "tesla": ["tesla", "tsla"],
    "broadcom": ["broadcom", "avgo"],
    "netflix": ["netflix", "nflx"],
    "oracle": ["oracle", "orcl"],
    "salesforce": ["salesforce", "crm"],
    "cisco": ["cisco", "csco"],
    "ibm": ["ibm"],
    "palantir": ["palantir", "pltr"],
    "intuit": ["intuit", "intu"],
    "servicenow": ["servicenow", "now"],
    "adobe": ["adobe", "adbe"],
    "qualcomm": ["qualcomm", "qcom"],
    "amd": ["amd"],
    "texas_instruments": ["texas instruments", "txn"],
    "uber": ["uber"],
    "booking": ["booking", "bkng", "booking.com"],
    "adp": ["automatic data processing", "adp"],
    "fiserv": ["fiserv", "fi"],
    "applied_materials": ["applied materials", "amat"],
    "palo_alto": ["palo alto networks", "panw"],
    "intel": ["intel", "intc"],
}

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

# --- Existing function: Convert to UTC.
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

# --- Added: Function to fetch NYT articles for a given keyword group starting from January 2020.
def fetch_nyt_articles(group, search_terms, start_year=2020, start_month=1):
    """
    Fetch New York Times articles for a given keyword group from January 2020 to the current month.
    Returns a list of dictionaries with keys "Time Data" and "Headline".
    """
    API_KEY = os.environ.get("NYT_API_KEY")
    BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    articles = []
    now = datetime.utcnow()
    current_year = now.year
    current_month = now.month

    year = start_year
    month = start_month
    # Prepare the search filter query by joining search terms with OR operator.
    joined_terms = " OR ".join(search_terms)

    while (year < current_year) or (year == current_year and month <= current_month):
        begin_date = f"{year:04d}{month:02d}01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year:04d}{month:02d}{last_day:02d}"

        params = {
            "fq": f"headline:({joined_terms})",  # Filter to only include articles with the search terms in the headline
            "sort": "newest",  # Sort articles by newest first
            "page": 0,  # Get the first page (up to 10 articles)
            "api-key": API_KEY,
            "begin_date": begin_date,
            "end_date": end_date,
        }

        print(
            f"Fetching NYT articles for {year}-{month:02d} (from {begin_date} to {end_date}) for group '{group}'..."
        )
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            data = response.json()
            docs = data.get("response", {}).get("docs", [])
            articles.extend(docs)
        else:
            print(
                f"Error fetching NYT articles for {year}-{month:02d}: {response.status_code}"
            )

        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

        if (year < current_year) or (year == current_year and month <= current_month):
            time.sleep(12)  # Respect rate limits

    print(f"Retrieved {len(articles)} NYT articles for group '{group}'.")

    # Convert the NYT articles to the common format with "Time Data" and "Headline"
    nyt_results = []
    for article in articles:
        pub_date_str = article.get("pub_date")
        if pub_date_str:
            try:
                dt = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%S%z")
                time_data = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception as e:
                print(f"Error parsing date {pub_date_str}: {e}")
                time_data = "Invalid Date"
        else:
            time_data = "No Date"
        headline = article.get("headline", {}).get("main", "No Headline")
        nyt_results.append({"Time Data": time_data, "Headline": headline})

    return nyt_results

# --- Function to process each keyword group in parallel.
def process_keyword_group(group, search_terms):
    print(f"\n===== Processing group: {group} =====\n")
    print(f"Starting process for group '{group}'...")

    # Store results and monthly counts for current group
    results = []
    monthly_counter = defaultdict(int)
    seen = set()  # To remove duplicates (Zulu time + headline)
    seen_monthly_titles = defaultdict(set)  # To remove title dupes per month

    # Process all configured files for current group
    for file_name, config in file_configs.items():
        print(f"\n🔍 Searching in {file_name} for group '{group}'...\n")
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
                                )
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
                        if (
                            any(
                                term.lower() in headline.lower()
                                for term in search_terms
                            )
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

    # --- Added: Process filtered_news_hugging_face.csv data.
    filtered_news_file = "filtered_news_hugging_face.csv"
    if os.path.exists(filtered_news_file):
        print(f"\n🔍 Searching in {filtered_news_file} for group '{group}'...\n")
        with open(filtered_news_file, newline="", encoding="utf-8") as fnf:
            reader = csv.DictReader(fnf)
            for row in reader:
                title_text = row.get("title", "").strip()
                published_at = row.get("published_at", "").strip()
                if not title_text or not published_at:
                    continue
                if any(term.lower() in title_text.lower() for term in search_terms):
                    dedup_key = (published_at, title_text)
                    try:
                        dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        continue
                    year_month = (dt.year, dt.month)
                    if (
                        dedup_key not in seen
                        and title_text not in seen_monthly_titles[year_month]
                    ):
                        results.append(
                            {"Time Data": published_at, "Headline": title_text}
                        )
                        monthly_counter[year_month] += 1
                        seen.add(dedup_key)
                        seen_monthly_titles[year_month].add(title_text)

    # --- Added: Optionally fetch NYT articles for current group from January 2020.
    # Due to the API rate limits, you can uncomment the following lines when you are ready to fetch NYT data.
    # nyt_results = fetch_nyt_articles(
    #     group, search_terms, start_year=2020, start_month=1
    # )
    # results.extend(nyt_results)
    # for article in nyt_results:
    #     try:
    #         dt = datetime.strptime(article["Time Data"], "%Y-%m-%dT%H:%M:%SZ")
    #         year_month = (dt.year, dt.month)
    #         monthly_counter[year_month] += 1
    #     except Exception:
    #         continue

    # Sort results by time for current group
    results.sort(key=lambda x: x["Time Data"])

    # Save headline results for current group
    output_file = f"{group.lower()}_data_aggregated_parallel.csv"  # --- Changed: Use group key for filename.
    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["Time Data", "Headline"])
        writer.writeheader()
        writer.writerows(results)

    # Save monthly headline counts for current group
    count_file = f"{group.lower()}_monthly_headline_count_parallel.csv"  # --- Changed: Use group key for filename.
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
        f"{group.lower()}_missing_dates_parallel.csv"  # --- Changed: Use group key for filename.
    )
    with open(missing_dates_file, "w", newline="", encoding="utf-8") as md_file:
        writer = csv.writer(md_file)
        writer.writerow(["Missing Date"])
        for d in missing_dates:
            writer.writerow([d])

    print(f"\n✅ Missing dates saved to: {missing_dates_file}")
    print(f"\n✅ Completed processing for group '{group}'")

if __name__ == '__main__':
    overall_start = time.time()
    print("Starting parallel processing of keyword groups...")
    groups = list(keyword_groups.items())
    # Limit the maximum number of workers to the number of groups (or a fixed number if preferred).
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(groups)) as executor:
        # Submit the processing tasks for each group.
        futures = {executor.submit(process_keyword_group, group, search_terms): group for group, search_terms in groups}
        for future in concurrent.futures.as_completed(futures):
            group = futures[future]
            try:
                future.result()
                print(f"Process for group '{group}' completed successfully.")
            except Exception as exc:
                print(f"⚠️ Group '{group}' generated an exception: {exc}")

    overall_end = time.time()
    total_time = overall_end - overall_start
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    # Save the total time to time_parallel.txt
    with open("time_parallel.txt", "w") as time_file:
        time_file.write(f"Total execution time: {total_time:.2f} seconds")

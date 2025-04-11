import csv
import os
from datetime import (
    datetime,
    timedelta,
    timezone,
)  # --- Added timedelta and timezone imports for missing dates analysis and NYT date conversion.
from dateutil import parser as date_parser
from collections import defaultdict
import pytz
import glob
import requests  # --- For NYT API calls.
from dotenv import load_dotenv  # --- For environment variables.
import time  # --- For rate limit handling.
import calendar  # --- For month end dates.
import multiprocessing
import sys
import os

load_dotenv()  # Load environment variables (e.g., NYT_API_KEY)

# --- Keyword groups mapping for companies and tickers.
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

# Publishers list
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

# Folder paths configuration
google_news_folder = "Google_News_Data"
publishers_folder = "Publishers_Data"
base_data_folder = "Older_Data"
analyst_data_folder = "Analyst_Data"

# Initialize file configurations at module level for multiprocessing
file_configs = {}

def initialize_file_configs():
    """Initialize file configurations once for all processes"""
    global file_configs
    
    # Base files under Older_Data
    base_files = {
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
    file_configs.update(base_files)

    # Publisher-specific files
    for pub in publishers:
        file_configs[os.path.join(publishers_folder, f"{pub.replace(' ', '_')}_headlines.csv")] = {
            "date_field": "Date",
            "headline_field": "Headline",
        }

    # Google News monthly files
    for fname in os.listdir(google_news_folder):
        if fname.endswith(".csv"):
            file_configs[os.path.join(google_news_folder, fname)] = {
                "date_field": "DateTime",
                "headline_field": "Title",
            }

    # Analyst data files
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

# Initialize file configurations before creating worker processes
initialize_file_configs()

# Timezone configuration
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc
tzinfos = {"ET": eastern}

def convert_to_utc(date_str, source_file):
    """Convert date string to UTC format with error handling"""
    try:
        if source_file.endswith("abcnews-date-text.csv"):
            dt = datetime.strptime(date_str.strip(), "%Y%m%d")
            dt_utc = dt.replace(tzinfo=utc)
        else:
            dt_local = date_parser.parse(date_str.strip(), fuzzy=True, tzinfos=tzinfos)
            dt_utc = dt_local.astimezone(utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ"), dt_utc
    except Exception as e:
        return f"InvalidDate({date_str})", None

def fetch_nyt_articles(group, search_terms, start_year=2020, start_month=1):
    """Fetch New York Times articles for a keyword group"""
    API_KEY = os.environ.get("NYT_API_KEY")
    if not API_KEY:
        print(f"⚠️ NYT_API_KEY not found in environment variables for group {group}")
        return []

    BASE_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    articles = []
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month

    year = start_year
    month = start_month
    joined_terms = " OR ".join(search_terms)

    while (year < current_year) or (year == current_year and month <= current_month):
        begin_date = f"{year:04d}{month:02d}01"
        last_day = calendar.monthrange(year, month)[1]
        end_date = f"{year:04d}{month:02d}{last_day:02d}"

        params = {
            "fq": f"headline:({joined_terms})",
            "sort": "newest",
            "page": 0,
            "api-key": API_KEY,
            "begin_date": begin_date,
            "end_date": end_date,
        }

        try:
            response = requests.get(BASE_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                articles.extend(data.get("response", {}).get("docs", []))
            else:
                print(f"⚠️ NYT API error for {group} ({year}-{month}): {response.status_code}")
        except Exception as e:
            print(f"⚠️ NYT API connection failed for {group}: {str(e)}")

        # Progress to next month
        if month == 12:
            month = 1
            year += 1
        else:
            month += 1

        time.sleep(12 / 1000)  # Basic rate limiting

    # Process articles into standard format
    processed = []
    for article in articles:
        pub_date = article.get("pub_date", "")
        headline = article.get("headline", {}).get("main", "")
        if pub_date and headline:
            try:
                dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%S%z")
                utc_time = dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                processed.append({"Time Data": utc_time, "Headline": headline})
            except Exception:
                continue
    return processed

def process_keyword_group(args):
    """Process a single keyword group with multiprocessing support"""
    group, search_terms = args
    pid = os.getpid()
    print(f"🚦 [PID:{pid}] Starting processing for {group}")

    results = []
    monthly_counter = defaultdict(int)
    seen = set()
    seen_monthly_titles = defaultdict(set)

    try:
        # Process all configured files
        for file_path, config in file_configs.items():
            print(f"🔍 [PID:{pid}] Processing {os.path.basename(file_path)} for {group}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Date processing
                        raw_date = row.get(config["date_field"], "")
                        utc_str, dt_obj = convert_to_utc(raw_date, file_path)
                        if not dt_obj:
                            continue

                        # Headline processing
                        headlines = []
                        if isinstance(config["headline_field"], list):
                            for field in config["headline_field"]:
                                if h := row.get(field, "").strip():
                                    headlines.append(h)
                        else:
                            if h := row.get(config["headline_field"], "").strip():
                                headlines.append(h)

                        # Keyword matching
                        for headline in headlines:
                            year_month = (dt_obj.year, dt_obj.month)
                            if any(term.lower() in headline.lower() for term in search_terms):
                                dedup_key = (utc_str, headline)
                                if dedup_key not in seen and headline not in seen_monthly_titles[year_month]:
                                    results.append({"Time Data": utc_str, "Headline": headline})
                                    monthly_counter[year_month] += 1
                                    seen.add(dedup_key)
                                    seen_monthly_titles[year_month].add(headline)
            except FileNotFoundError:
                print(f"⚠️ [PID:{pid}] Missing file: {file_path}")
            except Exception as e:
                print(f"⚠️ [PID:{pid}] Error processing {file_path}: {str(e)}")

        # Process filtered_news_hugging_face.csv if exists
        filtered_news_file = "filtered_news_hugging_face.csv"
        if os.path.exists(filtered_news_file):
            print(f"🔍 [PID:{pid}] Processing filtered news for {group}")
            try:
                with open(filtered_news_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        headline = row.get("title", "").strip()
                        date_str = row.get("published_at", "").strip()
                        if headline and date_str:
                            try:
                                dt = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
                                year_month = (dt.year, dt.month)
                                if any(term.lower() in headline.lower() for term in search_terms):
                                    dedup_key = (date_str, headline)
                                    if dedup_key not in seen and headline not in seen_monthly_titles[year_month]:
                                        results.append({"Time Data": date_str, "Headline": headline})
                                        monthly_counter[year_month] += 1
                                        seen.add(dedup_key)
                                        seen_monthly_titles[year_month].add(headline)
                            except Exception:
                                continue
            except Exception as e:
                print(f"⚠️ [PID:{pid}] Error processing filtered news: {str(e)}")

        # NYT API integration (uncomment to enable)
        # print(f"🌐 [PID:{pid}] Fetching NYT articles for {group}")
        # nyt_results = fetch_nyt_articles(group, search_terms)
        # results.extend(nyt_results)

        # Save results
        if results:
            output_file = f"{group}_data_aggregated.csv"
            print(f"💾 [PID:{pid}] Saving {len(results)} entries to {output_file}")
            with open(output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Time Data", "Headline"])
                writer.writeheader()
                writer.writerows(sorted(results, key=lambda x: x["Time Data"]))

            # Save monthly counts
            count_file = f"{group}_monthly_headline_count.csv"
            with open(count_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Year", "Month", "Headline Count"])
                for (y, m), c in sorted(monthly_counter.items()):
                    writer.writerow([y, m, c])

            # Missing dates analysis
            dates_present = set()
            with open(output_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        dt = datetime.strptime(row["Time Data"], "%Y-%m-%dT%H:%M:%SZ")
                        dates_present.add(dt.date())
                    except Exception:
                        continue

            if dates_present:
                start_date = min(dates_present)
                end_date = max(dates_present)
                current_date = start_date
                missing_dates = []
                while current_date <= end_date:
                    if current_date not in dates_present:
                        missing_dates.append(current_date.isoformat())
                    current_date += timedelta(days=1)
                
                if missing_dates:
                    missing_file = f"{group}_missing_dates.csv"
                    with open(missing_file, "w", encoding="utf-8", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Missing Date"])
                        writer.writerows([[d] for d in missing_dates])

        print(f"✅ [PID:{pid}] Completed processing for {group}")
    except Exception as e:
        print(f"❌ [PID:{pid}] Critical error processing {group}: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting parallel news aggregation")
    start_time = time.time()
    
    # Set up multiprocessing pool
    cpu_count = multiprocessing.cpu_count()
    print(f"⚙️ System has {cpu_count} available CPU cores")
    
    # Create process pool with 80% of available cores
    pool_size = max(1, int(cpu_count * 0.8))
    print(f"🔄 Creating process pool with {pool_size} workers")
    
    try:
        with multiprocessing.Pool(processes=pool_size) as pool:
            print("⏳ Processing keyword groups in parallel...")
            pool.map(process_keyword_group, keyword_groups.items())
    except KeyboardInterrupt:
        print("🛑 Process interrupted by user")
        pool.terminate()
    finally:
        pool.close()
        pool.join()
    
    total_time = time.time() - start_time
    print(f"🏁 All processes completed in {total_time:.2f} seconds")
    
    # Save timing information
    with open("processing_metadata.txt", "w") as f:
        f.write(f"Parallel processing completed in {total_time:.2f} seconds\n")
        f.write(f"Used {pool_size} processes\n")
        f.write(f"Processed {len(keyword_groups)} keyword groups\n")
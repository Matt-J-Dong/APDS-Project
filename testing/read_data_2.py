import os
import glob
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
from collections import defaultdict
import pytz
import requests  # --- For NYT API calls.
from dotenv import load_dotenv  # --- For environment variables.
import time  # --- For rate limit handling.
import calendar  # --- For month end dates.

load_dotenv()  # Load environment variables (e.g., NYT_API_KEY)

# --- Keyword groups mapping for companies and tickers.
keyword_groups = {
    "apple": ["apple", "aapl"],
    "microsoft": ["microsoft", "msft"],
    # "nvidia": ["nvidia", "nvda"],
    # "amazon": ["amazon", "amzn"],
    # "google": ["alphabet", "google", "goog", "googl"],
    # "meta": ["meta", "facebook", "meta platforms", "meta platforms (facebook)"],
    # "tesla": ["tesla", "tsla"],
    # "broadcom": ["broadcom", "avgo"],
    # "netflix": ["netflix", "nflx"],
    # "oracle": ["oracle", "orcl"],
    # "salesforce": ["salesforce", "crm"],
    # "cisco": ["cisco", "csco"],
    # "ibm": ["ibm"],
    # "palantir": ["palantir", "pltr"],
    # "intuit": ["intuit", "intu"],
    # "servicenow": ["servicenow", "now"],
    # "adobe": ["adobe", "adbe"],
    # "qualcomm": ["qualcomm", "qcom"],
    # "amd": ["amd"],
    # "texas_instruments": ["texas instruments", "txn"],
    # "uber": ["uber"],
    # "booking": ["booking", "bkng", "booking.com"],
    # "adp": ["automatic data processing", "adp"],
    # "fiserv": ["fiserv", "fi"],
    # "applied_materials": ["applied materials", "amat"],
    # "palo_alto": ["palo alto networks", "panw"],
    # "intel": ["intel", "intc"],
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

# Initialize file configurations at module level for reading all files.
file_configs = {}

def initialize_file_configs():
    """Initialize file configurations once for all files"""
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

    # Publisher-specific files (now in Publishers_Data)
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
    for analyst_file in glob.glob(os.path.join(analyst_data_folder, "analyst_ratings_processed_cleaned_*.csv")):
        file_configs[analyst_file] = {
            "date_field": "date",
            "headline_field": "title",
        }

    for partner_file in glob.glob(os.path.join(analyst_data_folder, "partner_headlines_cleaned_*.csv")):
        file_configs[partner_file] = {
            "date_field": "date",
            "headline_field": "headline",
        }

# Initialize file configurations before processing data
initialize_file_configs()

# --- Timezone configuration.
eastern = pytz.timezone("US/Eastern")
utc = pytz.utc
tzinfos = {"ET": eastern}

def convert_to_utc(date_str, source_file):
    """Convert date string to UTC format with error handling."""
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

def load_all_data():
    """
    Load all files as per file_configs into a single pandas DataFrame.
    Each file is transformed into a common format with columns:
    "Time Data", "Headline", and a parsed datetime ("Parsed").
    """
    all_dfs = []  # List to hold DataFrames from all files.

    for file_path, config in file_configs.items():
        if not os.path.exists(file_path):
            print(f"⚠️ File not found: {file_path}")
            continue

        try:
            # Read file into DataFrame.
            df = pd.read_csv(file_path, encoding="utf-8")
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            continue

        # Process date field: rename to "RawDate" for conversion.
        raw_date_col = config["date_field"]

        # Process headline field: if list, then stack multiple columns.
        if isinstance(config["headline_field"], list):
            # Create a DataFrame by melting the selected headline columns.
            # This produces one row per headline from the list columns.
            df_melted = df.melt(id_vars=[raw_date_col], 
                                value_vars=config["headline_field"],
                                var_name="headline_source", value_name="Headline")
            df_melted = df_melted.dropna(subset=["Headline"])
            df_processed = df_melted.rename(columns={raw_date_col: "RawDate"})
        else:
            df_processed = df[[raw_date_col, config["headline_field"]]].copy()
            df_processed = df_processed.rename(columns={raw_date_col: "RawDate", config["headline_field"]: "Headline"})
            df_processed = df_processed.dropna(subset=["Headline"])

        # Apply date conversion to create "Time Data" and "Parsed".
        # We use a lambda that calls the convert_to_utc function.
        df_processed["Time Data"], df_processed["Parsed"] = zip(*df_processed["RawDate"].astype(str).apply(lambda x: convert_to_utc(x, file_path)))
        # Filter out rows where conversion failed (Parsed is None)
        df_processed = df_processed[df_processed["Parsed"].notnull()]

        # Optionally, add a column for source if you wish to track file origin.
        df_processed["Source"] = os.path.basename(file_path)

        # Keep only the columns we need.
        df_final = df_processed[["Time Data", "Headline", "Parsed"]].copy()
        all_dfs.append(df_final)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} rows from all files.")
    else:
        combined_df = pd.DataFrame(columns=["Time Data", "Headline", "Parsed"])
        print("No data loaded.")
    return combined_df

# Load all data into a single DataFrame.
df_all = load_all_data()

# --- Approach 1: One-pass vectorized filtering per keyword group.
# For each keyword group, filter the combined DataFrame using vectorized string operations.
for group, keywords in keyword_groups.items():
    # Create a regex pattern: join keywords with the OR operator.
    # Using re.escape ensures literal matching.
    import re
    pattern = '|'.join(re.escape(term) for term in keywords)
    
    # Use vectorized string match on the "Headline" column (case-insensitive).
    mask = df_all["Headline"].str.contains(pattern, case=False, na=False)
    group_df = df_all[mask].copy()
    
    # Sort by time data for consistency.
    group_df.sort_values(by="Time Data", inplace=True)
    
    # Save aggregated headline results.
    output_file = f"{group.lower()}_data_aggregated.csv"
    group_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"✅ Group '{group}' saved with {len(group_df)} rows to {output_file}.")
    
    # --- Compute monthly headline counts.
    # Create a new column for year-month based on the parsed datetime.
    group_df["Year"] = group_df["Parsed"].dt.year
    group_df["Month"] = group_df["Parsed"].dt.month
    monthly_counts = group_df.groupby(["Year", "Month"]).size().reset_index(name="Headline Count")
    count_file = f"{group.lower()}_monthly_headline_count.csv"
    monthly_counts.to_csv(count_file, index=False, encoding="utf-8")
    print(f"📅 Monthly counts saved to {count_file}.")
    
    # --- Missing dates analysis.
    # Get unique dates from the parsed datetime.
    dates_present = group_df["Parsed"].dt.date.unique()
    if len(dates_present) > 0:
        start_date = min(dates_present)
        end_date = max(dates_present)
        all_dates = {start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)}
        missing_dates = sorted(all_dates - set(dates_present))
    else:
        missing_dates = []
    
    missing_file = f"{group.lower()}_missing_dates.csv"
    with open(missing_file, "w", encoding="utf-8", newline="") as f:
        writer = pd.ExcelWriter(f)  # Not strictly needed; we will write using pandas.
        # Instead, use pandas DataFrame for missing dates.
    # Alternatively, save using pandas:
    pd.DataFrame({"Missing Date": [d.isoformat() for d in missing_dates]}).to_csv(missing_file, index=False, encoding="utf-8")
    print(f"🔍 Missing dates saved to {missing_file}.")

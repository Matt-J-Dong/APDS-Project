import csv

# Define files and their respective formats
file_configs = {
    'abcnews-date-text.csv': {'date_field': 'publish_date', 'headline_field': 'headline_text'},
    'reuters_headlines.csv': {'date_field': 'Time', 'headline_field': 'Headlines'},
    'guardian_headlines_cleaned.csv': {'date_field': 'Time', 'headline_field': 'Headlines'},
    'cnbc_headlines_cleaned.csv': {'date_field': 'Time', 'headline_field': 'Headlines'}
}

# Search keyword
keyword = 'samsung'

for file_name, config in file_configs.items():
    print(f"\nSearching in {file_name}...\n")
    
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            headline = row[config['headline_field']]
            if keyword.lower() in headline.lower():
                print(f"{row[config['date_field']]}: {headline}")

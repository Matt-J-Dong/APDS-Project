import nasdaqdatalink
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
# Read your API key from environment or file:
nasdaqdatalink.read_key(filename=".env")

# Fetch data from the ZACKS/FC datatable for the ticker AAPL
data = nasdaqdatalink.get_table('ZACKS/FC', ticker='AAPL')

# Print all the column names available in the returned data
print("Available columns in ZACKS/FC for AAPL:")
print(data.columns.tolist())

columns_to_pull = [
    'ticker',
    'comp_name',
    'per_end_date',
    'per_fisc_year',
    'per_fisc_qtr',
    'tot_revnu',
    'oper_income',
    'net_income_parent_comp',
    'eps_basic_cont_oper',
    'eps_diluted_cont_oper',
    'filing_date'
]

# Retrieve data from the ZACKS/FC datatable for ticker AAPL using the specified columns.
# The 'paginate=True' parameter ensures that if there are multiple pages of data, they are merged.
data = nasdaqdatalink.get_table(
    'ZACKS/FC',
    ticker='AAPL',
    qopts={'columns': columns_to_pull, 'per_page': 100},
    paginate=True
)

print("Relevant ZACKS/FC data for AAPL:")
print(data)
import pandas as pd

# Load data from xlsx file
file_path = 'data/主力合约与月合约映射表.xlsx'
df = pd.read_excel(file_path, sheet_name=1)

# Use pd.melt to reshape the dataframe
# id_vars: keep the first column (date)
# var_name: name for the new column from original column names
# value_name: name for the values column
df = df.rename(columns={df.columns[0]: 'date'})
df_melted = pd.melt(df, id_vars=['date'], var_name='WINDCODE', value_name='MAPPING_WINDCODE')

# Print the result
print(df_melted)

# Load CSV data
csv_path = 'data/SI.csv'
df_csv = pd.read_csv(csv_path)

# Get rows where symbol changes
df_csv['first_symbol'] = df_csv['symbol'].ne(df_csv['symbol'].shift())
df_csv['last_symbol'] = df_csv['symbol'].ne(df_csv['symbol'].shift(-1))
symbol_first_symbol_previous_date = df_csv[df_csv['first_symbol']][['datetime', 'symbol']].copy()
symbol_last_symbol = df_csv[df_csv['last_symbol']][['datetime', 'symbol']].copy()
# For each first-symbol change row, compare its date to the immediate previous row's date in the original CSV.
# Keep the change date if it differs from the previous row's date; otherwise, pick the next available date.
symbol_first_symbol_next_date = df_csv[df_csv['first_symbol']][['datetime', 'symbol']].reset_index().rename(columns={'index': 'orig_index'}).copy()

def get_date_for_datetime(x, symbol_value, orig_idx):
    x_date = pd.to_datetime(x).date()
    # determine previous row date if available
    if orig_idx is not None and orig_idx > 0:
        prev_date = pd.to_datetime(df_csv['datetime'].iloc[orig_idx - 1]).date()
    else:
        prev_date = None

    # if current row's date differs from previous row's date, keep it
    if prev_date is None or x_date != prev_date:
        return x_date
    else:
        next_dates = df_csv[df_csv['datetime'] > x]
        for _, row in next_dates.iterrows():
            next_date = pd.to_datetime(row['datetime']).date()
            if next_date != x_date:
                print(f"Warning: Symbol '{symbol_value}' on datetime '{x}' has same date as previous row ({prev_date}); using next available date ({next_date})")
                return next_date
        next_date = (pd.to_datetime(x_date) + pd.Timedelta(days=1)).date()
        print(f"Warning: Symbol '{symbol_value}' on datetime '{x}' has same date as previous row ({prev_date}); using next available date ({next_date})")
        return next_date

symbol_first_symbol_next_date['date'] = symbol_first_symbol_next_date.apply(
    lambda row: get_date_for_datetime(row['datetime'], row['symbol'], row['orig_index']), axis=1
)
symbol_last_symbol['date'] = pd.to_datetime(symbol_last_symbol['datetime']).dt.date
# Add original index and compute 'date' as the date of the immediate previous row in df_csv
symbol_first_symbol_previous_date = symbol_first_symbol_previous_date.reset_index().rename(columns={'index': 'orig_index'}).copy()
# drop the first row (if present) and reindex rows
if not symbol_first_symbol_previous_date.empty:
    symbol_first_symbol_previous_date = symbol_first_symbol_previous_date.iloc[1:].reset_index(drop=True)

symbol_first_symbol_previous_date['date'] = symbol_first_symbol_previous_date.apply(
    lambda r: (
        pd.to_datetime(df_csv['datetime'].iloc[r['orig_index'] - 1]).date()
        if r['orig_index'] > 0
        else pd.to_datetime(df_csv['datetime'].iloc[r['orig_index']]).date()
    ),
    axis=1
)

# Merge with melted dataframe on date and symbol prefix match
symbol_first_symbol_previous_date['windcode_prefix'] = symbol_first_symbol_previous_date['symbol'].str.extract(r'(\D+)') + '_S.'
symbol_first_symbol_next_date['windcode_prefix'] = symbol_first_symbol_next_date['symbol'].str.extract(r'(\D+)') + '.'
symbol_last_symbol['windcode_prefix'] = symbol_last_symbol['symbol'].str.extract(r'(\D+)')  + '.'
# Concatenate first-change and last-change rows into one dataframe
symbol_changes = pd.concat([symbol_first_symbol_previous_date, symbol_first_symbol_next_date, symbol_last_symbol], ignore_index=True, sort=False)
# Print all rows and columns of symbol_changes
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(symbol_changes)
def check_match(row):
    # Remove only the first numeric character from symbol, keep everything else
    sym = str(row['symbol'])
    symbol_no_digits = sym
    for i, ch in enumerate(sym):
        if ch.isdigit():
            symbol_no_digits = sym[:i] + sym[i+1:]
            break
    matches = df_melted[
        (df_melted['date'].astype(str) == str(row['date'])) & 
        (df_melted['WINDCODE'].astype(str).str.startswith(str(row['windcode_prefix']))) &
        (
            (df_melted['MAPPING_WINDCODE'].astype(str).str.startswith(str(row['symbol']))) |
            (df_melted['MAPPING_WINDCODE'].astype(str).str.startswith(symbol_no_digits))
        )
    ]
    if len(matches) == 1:
        print(f"Checking: date={row['date']}, prefix={row['windcode_prefix']}, symbol={row['symbol']}, matched_row={matches.iloc[0].to_dict()}")
    elif len(matches) > 1:
        raise ValueError(f"Multiple matches found for date={row['date']}, prefix={row['windcode_prefix']}: {matches.to_dict('records')}")
    else:
        print(f"Checking: date={row['date']}, prefix={row['windcode_prefix']}, symbol={row['symbol']}, matches={len(matches)}")
    return matches.empty

mismatches = symbol_changes[symbol_changes.apply(check_match, axis=1)]

print("\nMismatches:")
print(mismatches[['datetime', 'symbol', 'date']])
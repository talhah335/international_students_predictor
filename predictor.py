import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV with headers and skip top metadata lines
df_raw = pd.read_csv(
    "data/EN_ODP-TR-Study-IS_CITZ_sign_date.csv",
    header=[0, 1, 2],
    skiprows=2,
    index_col=0
)

# Flatten columns into single strings
df_raw.columns = [
    ' '.join([str(i) for i in col if str(i) != 'nan' and not str(i).startswith('Unnamed')]).strip()
    for col in df_raw.columns
]

# Reset index to make the country name a regular column
df_raw.reset_index(inplace=True)

# Rename the index column to 'Country'
df_raw.rename(columns={df_raw.columns[0]: 'Country'}, inplace=True)

# Only keep columns with yearly totals, ignore quarters and unnamed
yearly_total_cols = [
    col for col in df_raw.columns
    if 'Total' in col and 'Q' not in col and 'Unnamed' not in col and not col.startswith('2025')
]


# Select only country and those total columns
df_yearly = df_raw[['Country'] + yearly_total_cols].copy()

# Convert column names like '2015 Total' → 2015
df_yearly.columns = ['Country'] + [int(col.split()[0]) for col in yearly_total_cols]

# Convert all year columns to numeric
for col in df_yearly.columns[1:]:
    df_yearly[col] = pd.to_numeric(df_yearly[col].astype(str).str.replace(',', ''), errors='coerce')

def predict_students_yearly(country_name, target_year):
    row = df_yearly[df_yearly['Country'].str.lower() == country_name.lower()]
    if row.empty:
        print(f"No data found for country: {country_name}")
        return

    years = np.array(df_yearly.columns[1:], dtype=float)
    values = row.iloc[0, 1:].values.astype(float)

    mask = ~np.isnan(values)
    X = years[mask].reshape(-1, 1)
    y = values[mask].reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_year = np.array([[target_year]])
    prediction = model.predict(future_year)

    plt.figure(figsize=(12, 6))
    plt.plot(X.flatten(), y.flatten(), 'o-', label='Actual Data')
    plt.plot(future_year, prediction, 'ro', label=f'Predicted {target_year}')
    plt.plot(
        np.append(X, future_year),
        model.predict(np.append(X, future_year).reshape(-1, 1)),
        '--',
        color='gray',
        label='Trend Line'
    )
    plt.xlabel("Year")
    plt.ylabel("Number of Students")
    plt.title(f"Yearly International Students from {country_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Predicted number of international students from {country_name} in {target_year}: {int(prediction[0][0])}")

country = input("Enter country: ")
year = int(input("Enter year: "))
predict_students_yearly(country, year)















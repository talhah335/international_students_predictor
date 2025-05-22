from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import os

app = Flask(__name__)

# Load CSV once at startup
df_raw = pd.read_csv(
    "data/EN_ODP-TR-Study-IS_CITZ_sign_date.csv",
    header=[0, 1, 2],
    skiprows=2,
    index_col=0
)

# Clean headers
df_raw.columns = [
    ' '.join([str(i) for i in col if str(i) != 'nan' and not str(i).startswith('Unnamed')]).strip()
    for col in df_raw.columns
]

df_raw.reset_index(inplace=True)
df_raw.rename(columns={df_raw.columns[0]: 'Country'}, inplace=True)

# Keep only yearly totals
yearly_total_cols = [
    col for col in df_raw.columns
    if 'Total' in col and 'Q' not in col and 'Unnamed' not in col and not col.startswith('2025')
]

df_yearly = df_raw[['Country'] + yearly_total_cols].copy()
df_yearly.columns = ['Country'] + [int(col.split()[0]) for col in yearly_total_cols]

# Clean numeric values
for col in df_yearly.columns[1:]:
    df_yearly[col] = pd.to_numeric(df_yearly[col].astype(str).str.replace(',', ''), errors='coerce')


def make_prediction(country_name, target_year):
    row = df_yearly[df_yearly['Country'].str.lower() == country_name.lower()]
    if row.empty:
        return -1, None

    years = np.array(df_yearly.columns[1:], dtype=float)
    values = row.iloc[0, 1:].values.astype(float)

    mask = ~np.isnan(values)
    X = years[mask].reshape(-1, 1)
    y = values[mask].reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    future_year = np.array([[target_year]])
    prediction = model.predict(future_year)

    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(X.flatten(), y.flatten(), 'o-', label='Actual Data')
    plt.plot(future_year, prediction, 'ro', label=f'Predicted {target_year}')
    plt.plot(np.append(X, future_year), model.predict(np.append(X, future_year).reshape(-1, 1)), '--', color='gray', label='Trend Line')
    plt.xlabel("Year")
    plt.ylabel("Number of Students")
    plt.title(f"International Students from {country_name}")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return int(prediction[0][0]), buf


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    country = data.get('country')
    year = int(data.get('year'))
    prediction, img_buf = make_prediction(country, year)

    if prediction == -1:
        return jsonify({'error': f"No data found for country: {country}"}), 404

    # Save image temporarily
    img_path = f"static/prediction_{country}_{year}.png"
    os.makedirs("static", exist_ok=True)
    with open(img_path, 'wb') as f:
        f.write(img_buf.read())

    return jsonify({
        'prediction': prediction,
        'image_url': f'/{img_path}'
    })


if __name__ == '__main__':
    app.run(debug=True)





















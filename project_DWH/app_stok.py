#!/usr/bin/env python3

# NAMA : KINTAN ADHIESTYARESTHU
# NIM : 235091107111001
# NO.ABSEN : 32
# MATA KULIAH: DATA WAREHOUSE

from flask import Flask, render_template, request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
import base64
import mysql.connector
import os

app = Flask(__name__)

def get_db_connection():
    return mysql.connector.connect(
        host=os.environ.get('localhost'),
        user=os.environ.get('root'),
        password=os.environ.get('Adhiestyaresthu08#'),
        database=os.environ.get('gudang_db')
    )

def save_user_input(produk, gudang, periode_awal, periode_akhir):
    cursor = None
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        # your logic here
    except Exception as e:
        print("Error:", e)
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()

# Load dataset
original_df = pd.read_csv("/workspaces/FRESHTRACK-STOK-BUAH/project_DWH/gudang_stok.csv", encoding='latin1', sep=';')
original_df['tanggal'] = pd.to_datetime(original_df['tanggal'])
original_df['bulan'] = original_df['tanggal'].dt.month
original_df['minggu'] = original_df['tanggal'].dt.isocalendar().week
original_df['musim'] = original_df['bulan'].apply(
    lambda x: 'Musim Panas' if x in [6,7,8] else 'Musim Hujan' if x in [12,1,2] else 'Musim Peralihan'
)
original_df.fillna(0, inplace=True)

list_produk = sorted(original_df['produk'].unique())
list_gudang = sorted(original_df['lokasi'].unique())
max_minggu = int(original_df['minggu'].max())

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    base64_img = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return base64_img

@app.route('/', methods=['GET', 'POST'])
def dashboard():
    produk = request.form.get('produk', list_produk[0])
    gudang = request.form.get('gudang', list_gudang[0])
    periode_awal = int(request.form.get('awal', 1))
    periode_akhir = int(request.form.get('akhir', max_minggu))

    if request.method == 'POST':
        save_user_input(produk, gudang, periode_awal, periode_akhir)

    # Filter the data based on user input
    df_filtered = original_df[
        (original_df['produk'] == produk) & 
        (original_df['lokasi'] == gudang) & 
        (original_df['minggu'].between(periode_awal, periode_akhir))
    ]

    if df_filtered.empty:
        return render_template('Dashboard.html', 
                               produk=produk, 
                               gudang=gudang, 
                               periode_awal=periode_awal, 
                               periode_akhir=periode_akhir,
                               list_produk=list_produk,
                               list_gudang=list_gudang,
                               max_minggu=max_minggu,
                               no_data=True)

    total_stok = df_filtered['stok'].sum()
    rata_rata_stok = df_filtered.groupby('minggu')['stok'].sum().mean()
    jumlah_minggu = df_filtered['minggu'].nunique()

    kapasitas_maksimum = 8000
    overload = total_stok > kapasitas_maksimum
    persen_overload = ((total_stok - kapasitas_maksimum) / kapasitas_maksimum) * 100 if overload else 0

    # Grafik Tren Stok Harian
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df_plot = df_filtered.groupby('tanggal')['stok'].sum()
    ax1.plot(df_plot.index, df_plot.values, marker='o', linestyle='-', linewidth=2)
    ax1.set_title('Grafik Tren Stok Harian')
    ax1.grid(True)
    ax1.tick_params(axis='x', rotation=45)
    tren_stok_img = plot_to_base64(fig1)
    plt.close(fig1)

    # Heatmap Musiman
    fig2, ax2 = plt.subplots()
    pivot_table = df_filtered.pivot_table(index='minggu', columns='musim', values='stok', aggfunc='sum')
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt=".0f", ax=ax2)
    ax2.set_title("Heatmap Permintaan Musiman")
    heatmap_img = plot_to_base64(fig2)
    plt.close(fig2)

    # Pie Chart Distribusi Kualitas
    fig_kualitas, ax_kualitas = plt.subplots()
    kualitas_counts = df_filtered['kualitas'].value_counts()
    ax_kualitas.pie(kualitas_counts, labels=kualitas_counts.index, autopct='%1.1f%%', startangle=90)
    ax_kualitas.set_title("Distribusi Kualitas Produk")
    pie_kualitas_img = plot_to_base64(fig_kualitas)
    plt.close(fig_kualitas)

    # Pie Chart Kategori Harga
    def kategori_harga(harga):
        if harga <= 10:
            return 'Murah'
        elif 10 < harga <= 25:
            return 'Sedang'
        else:
            return 'Mahal'

    df_filtered['kategori_harga'] = df_filtered['harga'].apply(kategori_harga)

    fig_harga, ax_harga = plt.subplots()
    harga_counts = df_filtered['kategori_harga'].value_counts()
    ax_harga.pie(harga_counts, labels=harga_counts.index, autopct='%1.1f%%', startangle=90)
    ax_harga.set_title("Distribusi Kategori Harga")
    pie_harga_img = plot_to_base64(fig_harga)
    plt.close(fig_harga)

    # Random Forest Model Prediction
    pred_img, mse, r2_score_val = None, None, None
    if len(df_filtered) >= 12:
        # Define feature and target variables
        # Using 'minggu' as a feature, and 'stok' as target variable
        X = df_filtered[['minggu']]  # Features
        y = df_filtered['stok']     # Target variable

        # Train-test split (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Random Forest Model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        r2_score_val = model.score(X_test, y_test)  # R² score

        # Plot Predicted vs Actual
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(y_test.values, label='Actual', marker='o', linestyle='-', color='blue')
        ax3.plot(predictions, label='Predicted', marker='x', linestyle='--', color='red')
        ax3.legend()
        ax3.set_title('Prediksi vs Aktual Stok')
        ax3.set_xlabel('Minggu Ke-')
        ax3.set_ylabel('Stok')
        pred_img = plot_to_base64(fig3)  # Convert plot to base64 image
        plt.close(fig3)

    # Folium Map
    m = folium.Map(location=[-2.5, 117], zoom_start=5)
    heat_data = [[row['lat'], row['lon'], row['stok']] for index, row in df_filtered.iterrows()]
    HeatMap(heat_data).add_to(m)
    map_html = m._repr_html_()

    return render_template('Dashboard.html', 
                           produk=produk, 
                           gudang=gudang, 
                           periode_awal=periode_awal, 
                           periode_akhir=periode_akhir,
                           list_produk=list_produk,
                           list_gudang=list_gudang,
                           max_minggu=max_minggu,
                           tren_stok_img=tren_stok_img,
                           heatmap_img=heatmap_img,
                           pred_img=pred_img,
                           mse=mse,
                           r2_score_val=r2_score_val,  # Pass R² score to template
                           map_html=map_html,
                           overload=overload,
                           persen_overload=persen_overload,
                           no_data=False,
                           total_stok=total_stok,
                           rata_rata_stok=rata_rata_stok,
                           jumlah_minggu=jumlah_minggu,
                           df_filtered=df_filtered,
                           pie_kualitas_img=pie_kualitas_img,
                           pie_harga_img=pie_harga_img)

@app.route('/export', methods=['POST'])
def export():
    produk = request.form.get('produk')
    gudang = request.form.get('gudang')
    periode_awal = int(request.form.get('awal'))
    periode_akhir = int(request.form.get('akhir'))

    df_filtered = original_df[
        (original_df['produk'] == produk) &
        (original_df['lokasi'] == gudang) &
        (original_df['minggu'].between(periode_awal, periode_akhir))
    ]

    csv_buffer = io.StringIO()
    df_filtered.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)

    return (
        csv_buffer.getvalue(),
        200,
        {
            'Content-Type': 'text/csv',
            'Content-Disposition': f'attachment; filename=export_{produk}_{gudang}.csv'
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


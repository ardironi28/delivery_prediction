# Delivery Time Prediction — Streamlit

Aplikasi Streamlit untuk memprediksi estimasi waktu pengiriman makanan dalam menit menggunakan XGBoost.

## File repository

- `app.py` — antarmuka Streamlit dan form input prediksi.
- `model_delivery.pkl` — pipeline preprocessing + model XGBoost.
- `metrics.json` — metrik evaluasi model.
- `requirements.txt` — dependensi aplikasi.
- `Final_Project_Kelompok_4.ipynb` — notebook project asli sebagai dokumentasi.

## Fitur input

Aplikasi menerima 8 fitur numerik:
1. `total_items`
2. `subtotal`
3. `num_distinct_items`
4. `min_item_price`
5. `max_item_price`
6. `total_onshift_partners`
7. `total_busy_partners`
8. `total_outstanding_orders`

## Menjalankan secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Upload semua isi folder repository ini ke GitHub.
2. Buka https://share.streamlit.io/ dan login dengan GitHub.
3. Klik **Create app**.
4. Pilih repository, branch `main`, dan main file path `app.py`.
5. Klik **Deploy**.

## Catatan model

Model dilatih ulang menggunakan dataset project yang dibersihkan mengikuti notebook: baris tanpa timestamp target dihapus, target durasi dihitung dari selisih `actual_delivery_time` dan `created_at`, lalu outlier target difilter dengan aturan IQR. Model menggunakan 8 fitur numerik dan target `delivery_time_minutes`.

Preprocessing median imputation dan StandardScaler disimpan bersama model dalam satu scikit-learn Pipeline. Dengan demikian, input dari Streamlit diproses dengan preprocessing yang sama seperti saat training.

Metrik uji yang tercantum di `metrics.json` berasal dari split 80:20 dengan `random_state=42`. Nilai R² sekitar 0.206 menunjukkan kemampuan penjelasan model masih terbatas; hasil prediksi sebaiknya dipakai sebagai estimasi, bukan kepastian.

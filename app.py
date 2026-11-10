import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_delivery.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

FEATURES = [
    "total_items",
    "subtotal",
    "num_distinct_items",
    "min_item_price",
    "max_item_price",
    "total_onshift_partners",
    "total_busy_partners",
    "total_outstanding_orders",
]

st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Delivery Time Prediction")
st.caption(
    "Aplikasi estimasi durasi pengiriman makanan menggunakan model XGBoost "
    "yang dilatih dari project Kelompok 4."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

try:
    model = load_model()
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()

metrics = load_metrics()

with st.sidebar:
    st.header("Tentang Model")
    st.write("**Algoritma:** XGBoost Regressor")
    st.write("**Target prediksi:** delivery time (menit)")
    st.write("**Jumlah fitur:** 8")
    if metrics:
        st.metric("MAE", f"{metrics.get('MAE_minutes', 0):.2f} menit")
        st.metric("RMSE", f"{metrics.get('RMSE_minutes', 0):.2f} menit")
        st.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
    st.caption(
        "MAE dan RMSE menunjukkan rata-rata besar kesalahan prediksi. "
        "R² menunjukkan proporsi variasi target yang dapat dijelaskan model."
    )

st.subheader("Masukkan detail pesanan")
st.write("Isi nilai sesuai kondisi pesanan dan ketersediaan mitra pengiriman.")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        total_items = st.number_input(
            "Total item dalam pesanan",
            min_value=1, max_value=500, value=3, step=1,
            help="Jumlah seluruh item yang dipesan."
        )
        subtotal = st.number_input(
            "Subtotal pesanan",
            min_value=0, max_value=100000, value=2660, step=50,
            help="Gunakan satuan harga yang sama dengan dataset training."
        )
        num_distinct_items = st.number_input(
            "Jumlah jenis item berbeda",
            min_value=1, max_value=100, value=2, step=1
        )
        min_item_price = st.number_input(
            "Harga item termurah",
            min_value=0, max_value=50000, value=600, step=50
        )

    with right:
        max_item_price = st.number_input(
            "Harga item termahal",
            min_value=0, max_value=50000, value=1100, step=50
        )
        total_onshift_partners = st.number_input(
            "Jumlah mitra yang sedang bertugas",
            min_value=0, max_value=500, value=44, step=1
        )
        total_busy_partners = st.number_input(
            "Jumlah mitra yang sedang sibuk",
            min_value=0, max_value=500, value=41, step=1
        )
        total_outstanding_orders = st.number_input(
            "Jumlah pesanan yang belum selesai",
            min_value=0, max_value=1000, value=56, step=1
        )

    submitted = st.form_submit_button(
        "🔮 Prediksi waktu pengiriman",
        type="primary",
        use_container_width=True,
    )

if submitted:
    errors = []
    if num_distinct_items > total_items:
        errors.append("Jumlah jenis item berbeda tidak boleh melebihi total item.")
    if min_item_price > max_item_price:
        errors.append("Harga item termurah tidak boleh melebihi harga item termahal.")
    if total_busy_partners > total_onshift_partners:
        errors.append("Jumlah mitra sibuk tidak boleh melebihi jumlah mitra yang bertugas.")

    if errors:
        for error in errors:
            st.error(error)
    else:
        input_data = pd.DataFrame([{
            "total_items": total_items,
            "subtotal": subtotal,
            "num_distinct_items": num_distinct_items,
            "min_item_price": min_item_price,
            "max_item_price": max_item_price,
            "total_onshift_partners": total_onshift_partners,
            "total_busy_partners": total_busy_partners,
            "total_outstanding_orders": total_outstanding_orders,
        }], columns=FEATURES)

        try:
            prediction = float(model.predict(input_data)[0])
            prediction = max(0.0, prediction)

            st.divider()
            st.subheader("Hasil Prediksi")
            st.metric(
                label="Estimasi waktu pengiriman",
                value=f"{prediction:.1f} menit",
                delta=f"Sekitar {int(round(prediction))} menit",
                delta_color="off",
            )
            st.info(
                "Hasil ini merupakan estimasi berdasarkan pola data historis, "
                "bukan jaminan waktu pengiriman aktual."
            )
            with st.expander("Lihat data yang dikirim ke model"):
                st.dataframe(input_data, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediksi gagal dijalankan: {exc}")

st.divider()
st.caption(
    "Final Project Kelompok 4 • Delivery Time Prediction • "
    "Model dan preprocessing tersimpan dalam satu pipeline."
)

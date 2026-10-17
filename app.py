import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_irigasi_tuned.pkl"
METADATA_PATH = BASE_DIR / "metadata_irigasi.json"

st.set_page_config(page_title="Prediksi Kebutuhan Irigasi", page_icon="🌱", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)

st.title("🌱 Prediksi Kebutuhan Irigasi Tanaman")
st.write("Aplikasi klasifikasi kebutuhan irigasi tanaman berbasis Machine Learning.")

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("File model atau metadata belum ditemukan. Letakkan model_irigasi_tuned.pkl dan metadata_irigasi.json dalam folder yang sama dengan app.py.")
    st.stop()

try:
    model = load_model()
    metadata = load_metadata()
except Exception as exc:
    st.error(f"Gagal memuat model atau metadata: {exc}")
    st.info("Pastikan versi scikit-learn kompatibel dengan versi yang digunakan saat model disimpan dari Google Colab.")
    st.stop()

numeric_features = metadata["numeric_features"]
categorical_features = metadata["categorical_features"]
feature_order = metadata["feature_order"]

tab_single, tab_batch = st.tabs(["🔎 Prediksi Satu Data", "📂 Prediksi Banyak Data"])

with tab_single:
    st.subheader("Masukkan karakteristik tanah, cuaca, tanaman, dan lahan")
    with st.form("single_prediction_form"):
        input_data = {}
        st.markdown("#### Fitur Numerik")
        numeric_columns = st.columns(2)

        for i, col in enumerate(numeric_features):
            info = metadata["numeric_info"][col]
            minimum, maximum, default = float(info["min"]), float(info["max"]), float(info["median"])
            margin = max((maximum - minimum) * 0.1, 1.0)
            with numeric_columns[i % 2]:
                input_data[col] = st.number_input(
                    col, min_value=minimum - margin, max_value=maximum + margin,
                    value=default, key=f"numeric_{col}"
                )

        st.markdown("#### Fitur Kategorikal")
        categorical_columns = st.columns(2)
        for i, col in enumerate(categorical_features):
            options = [str(value) for value in metadata["categorical_options"].get(col, [])]
            with categorical_columns[i % 2]:
                if options:
                    input_data[col] = st.selectbox(col, options=options, key=f"categorical_{col}")
                else:
                    input_data[col] = st.text_input(col, key=f"categorical_text_{col}")

        submitted = st.form_submit_button("Prediksi Kebutuhan Irigasi", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_data], columns=feature_order)
        for col in numeric_features:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            st.divider()
            st.subheader("Hasil Prediksi")
            label = str(prediction).strip().lower()
            if label == "high":
                st.error(f"Kategori kebutuhan irigasi: **{prediction}**")
            elif label == "medium":
                st.warning(f"Kategori kebutuhan irigasi: **{prediction}**")
            else:
                st.success(f"Kategori kebutuhan irigasi: **{prediction}**")
            st.caption("Probabilitas merupakan estimasi dari model, bukan pengukuran langsung kebutuhan air tanaman.")
            probability_df = pd.DataFrame({
                "Kategori": [str(c) for c in model.classes_],
                "Probabilitas (%)": probabilities * 100
            }).sort_values("Probabilitas (%)", ascending=False)
            st.markdown("#### Probabilitas setiap kelas")
            st.bar_chart(probability_df.set_index("Kategori")["Probabilitas (%)"])
            st.dataframe(probability_df.style.format({"Probabilitas (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
            with st.expander("Lihat data input"):
                st.dataframe(input_df, use_container_width=True)
        except Exception as exc:
            st.error(f"Prediksi gagal: {exc}")
            st.warning("Periksa nama kolom, tipe data, dan kecocokan versi scikit-learn.")

with tab_batch:
    st.subheader("Prediksi menggunakan file CSV")
    st.write("CSV harus memiliki semua kolom fitur yang sama dengan data training. Kolom target Irrigation_Need tidak wajib disertakan.")
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"], key="batch_csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.write("Pratinjau data:")
            st.dataframe(uploaded_df.head(), use_container_width=True)
            missing_columns = [col for col in feature_order if col not in uploaded_df.columns]
            if missing_columns:
                st.error("Kolom fitur berikut tidak ditemukan: " + ", ".join(missing_columns))
            elif st.button("Prediksi Semua Data", type="primary"):
                X_upload = uploaded_df[feature_order].copy()
                for col in numeric_features:
                    X_upload[col] = pd.to_numeric(X_upload[col], errors="coerce")
                predictions = model.predict(X_upload)
                probabilities = model.predict_proba(X_upload)
                result_df = uploaded_df.copy()
                result_df["Prediksi_Irrigation_Need"] = predictions
                for i, class_name in enumerate(model.classes_):
                    result_df[f"Probabilitas_{class_name}_(%)"] = np.round(probabilities[:, i] * 100, 2)
                st.success(f"Prediksi selesai untuk {len(result_df)} baris data.")
                st.dataframe(result_df, use_container_width=True)
                csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ Unduh hasil prediksi CSV", data=csv_bytes, file_name="hasil_prediksi_irigasi.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Gagal memproses CSV: {exc}")

st.divider()
st.caption("Model klasifikasi kebutuhan irigasi tanaman | Validasi hasil prediksi sebelum digunakan untuk keputusan irigasi.")

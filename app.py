
import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

# --- Import Custom Model Classes ---
# Pastikan file custom_models.py berada di direktori yang sama
try:
    from custom_models import NaiveBayes, RandomForest, DecisionTree, Node # Impor semua kelas yang mungkin dibutuhkan
    st.sidebar.success("Kelas model kustom berhasil diimpor dari custom_models.py")
except ImportError:
    st.error("Error: Tidak dapat mengimpor kelas dari custom_models.py. Pastikan file tersebut ada di direktori yang sama dengan app.py.")
    NaiveBayes = None
    RandomForest = None
except Exception as e:
    st.error(f"Error saat mengimpor custom_models.py: {e}")
    NaiveBayes = None
    RandomForest = None

# --- Download NLTK data ---
# def download_nltk_data():
#     try:
#         nltk.data.find("corpora/wordnet")
#     except nltk.downloader.DownloadError:
#         with st.spinner("Downloading NLTK data (wordnet)..."):
#             nltk.download("wordnet", quiet=True)
#     try:
#         nltk.data.find("corpora/stopwords")
#     except nltk.downloader.DownloadError:
#         with st.spinner("Downloading NLTK data (stopwords)..."):
#             nltk.download("stopwords", quiet=True)
#     try:
#         nltk.data.find("tokenizers/punkt")
#     except nltk.downloader.DownloadError:
#         with st.spinner("Downloading NLTK data (punkt)..."):
#             nltk.download("punkt", quiet=True)

# download_nltk_data()

# --- Fungsi Preprocessing ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text):
    if not isinstance(text, str):
        return []
    try:
        tokens = word_tokenize(text)
        return tokens
    except Exception as e:
        st.error(f"Error saat tokenisasi: {e}")
        return []

def remove_stopwords(tokens):
    if not isinstance(tokens, list):
        return []
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    if not isinstance(tokens, list):
        return []
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return lemmatized_tokens

def preprocess_text_pipeline(text):
    if not isinstance(text, str):
        return []
    cleaned_text = clean_text(text)
    tokens = tokenize_text(cleaned_text)
    tokens_no_stopwords = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(tokens_no_stopwords)
    return " ".join(lemmatized_tokens)

# --- Memuat Vectorizer, Selector, dan Model ---
tfidf_vectorizer_path = "tfidf_vectorizer.joblib"
selector_path = "chi2_selector.pkl"        
nb_model_path = "naive_bayes_model.joblib"       
rf_model_path = "random_forest_model.joblib"     

vectorizer = None
selector = None
nb_model = None
rf_model = None

# Muat TF-IDF Vectorizer
with st.spinner(f"Memuat TF-IDF vectorizer{tfidf_vectorizer_path}..."):
    try:
        vectorizer = joblib.load(tfidf_vectorizer_path)
        st.sidebar.success(f"TF-IDF Vectorizer{tfidf_vectorizer_path} berhasil dimuat.")
    except FileNotFoundError:
        st.sidebar.error(f"Error: File TF-IDF vectorizer{tfidf_vectorizer_path} tidak ditemukan.")
    except Exception as e:
        st.sidebar.error(f"Error saat memuat TF-IDF vectorizer{tfidf_vectorizer_path}: {e}")

# Muat Feature Selector (Chi2)
with st.spinner(f"Memuat feature selector{selector_path}..."):
    try:
        selector = joblib.load(selector_path)
        st.sidebar.success(f"Feature Selector{selector_path} berhasil dimuat.")
    except FileNotFoundError:
        st.sidebar.error(f"Error: File feature selector{selector_path} tidak ditemukan.")
    except Exception as e:
        st.sidebar.error(f"Error saat memuat feature selector{selector_path}: {e}")

# Muat Model Naive Bayes
if NaiveBayes is not None:
    with st.spinner(f"Memuat model Naive Bayes{nb_model_path}..."):
        try:
            nb_model = joblib.load(nb_model_path)
            st.sidebar.success(f"Model Naive Bayes{nb_model_path} berhasil dimuat.")
        except FileNotFoundError:
            st.sidebar.error(f"Error: File model Naive Bayes{nb_model_path} tidak ditemukan.")
        except AttributeError as e:
            st.sidebar.error(f"Error loading model Naive Bayes: {e}.")
        except Exception as e:
            st.sidebar.error(f"Error saat memuat model Naive Bayes{nb_model_path}: {e}")
else:
    st.sidebar.warning("Kelas NaiveBayes tidak dapat diimpor, pemuatan model dilewati.")

# Muat Model Random Forest
if RandomForest is not None:
    with st.spinner(f"Memuat model Random Forest{rf_model_path}..."):
        try:
            rf_model = joblib.load(rf_model_path)
            st.sidebar.success(f"Model Random Forest{rf_model_path} berhasil dimuat.")
        except FileNotFoundError:
            st.sidebar.error(f"Error: File model Random Forest{rf_model_path} tidak ditemukan.")
        except AttributeError as e:
            st.sidebar.error(f"Error loading model Random Forest: {e}.")
        except Exception as e:
            st.sidebar.error(f"Error saat memuat model Random Forest{rf_model_path}: {e}")
else:
    st.sidebar.warning("Kelas RandomForest tidak dapat diimpor, pemuatan model dilewati.")


st.title("Deteksi Pelecehan dari Data Teks")

st.markdown("""
Selamat datang di aplikasi deteksi pelecehan teks. Aplikasi ini menggunakan model Machine Learning
(yang dibangun dari scratch) untuk mengklasifikasikan apakah suatu teks mengandung unsur pelecehan atau tidak.
""")

st.header("Deskripsi Proyek & Problem Statement")
st.markdown("""
...
""") 

st.header("💡 Solusi yang Ditawarkan")
st.markdown("""
Solusi yang diimplementasikan adalah pipeline NLP dan ML:
1.  Pembersihan Teks, Tokenisasi, Stopword Removal, Lemmatisasi.
2.  Ekstraksi Fitur: Menggunakan TF-IDF (`tfidf_vectorizer.joblib`) untuk mendapatkan representasi numerik (10k fitur).
3.  **Seleksi Fitur (untuk Naive Bayes):** Menggunakan Chi2 Selector (`chi2_selector.pkl`) untuk memilih 5k fitur terbaik dari hasil TF-IDF.
4.  Klasifikasi:
    *   **Naive Bayes:** Menggunakan model (`naive_bayes_model.joblib`) pada 5k fitur hasil seleksi.
    *   **Random Forest:** Menggunakan model (`random_forest_model.joblib`) pada 5k fitur hasil seleksi.
""")

st.header("⌨Coba Deteksi Teks Anda")
user_input = st.text_area("Masukkan teks yang ingin Anda periksa:", height=150)

if st.button("Deteksi Sekarang"):
    if not user_input.strip():
        st.warning("Silakan masukkan teks terlebih dahulu.")
    # Periksa apakah semua komponen yang diperlukan ada
    elif vectorizer is None or (nb_model is None and selector is None) and rf_model is None:
        st.error("Komponen penting (vectorizer/selector/model) belum berhasil dimuat. Tidak dapat melakukan prediksi. Periksa pesan error di sidebar.")
    else:
        with st.spinner("Memproses teks dan melakukan prediksi..."):
            # 1. Preprocess teks input
            processed_input = preprocess_text_pipeline(user_input)

            # 2. Transform teks menggunakan TF-IDF Vectorizer (menghasilkan 10k fitur)
            try:
                input_vector_tfidf = vectorizer.transform([processed_input])
                st.subheader("Hasil Prediksi:")

                # 3. Prediksi menggunakan Naive Bayes (jika model & selector dimuat)
                if nb_model is not None and selector is not None:
                    try:
                        # Terapkan Feature Selection (Chi2) ke hasil TF-IDF
                        input_vector_nb = selector.transform(input_vector_tfidf)
                        # Sekarang input_vector_nb punya 5k fitur

                        nb_prediction = nb_model.predict(input_vector_nb)
                        label_nb = "Pelecehan" if nb_prediction[0] == 1 else "Non-Pelecehan"
                        if hasattr(nb_model, 'predict_proba') and callable(getattr(nb_model, 'predict_proba')):
                            nb_proba = nb_model.predict_proba(input_vector_nb)
                            st.write(f"**Model Naive Bayes (5k fitur):** {label_nb} (Probabilitas: {nb_proba[0][nb_prediction[0]]:.2f})")
                        else:
                            st.write(f"**Model Naive Bayes (5k fitur):** {label_nb} (Probabilitas tidak tersedia)")
                    except ValueError as ve:
                         st.error(f"Error dimensi saat prediksi Naive Bayes: {ve}. Pastikan selector menghasilkan jumlah fitur yang sesuai dengan model NB.")
                    except Exception as e:
                        st.error(f"Error prediksi Naive Bayes: {e}")
                elif nb_model is None:
                     st.warning("Model Naive Bayes tidak dimuat, prediksi NB dilewati.")
                elif selector is None:
                     st.warning("Feature selector tidak dimuat, prediksi NB dilewati.")

                # 4. Prediksi menggunakan Random Forest (jika model dimuat)
                if rf_model is not None and selector is not None: # Also check selector is loaded
                    try:
                        # Terapkan Feature Selection (Chi2) ke hasil TF-IDF
                        input_vector_rf = selector.transform(input_vector_tfidf)
                        # Sekarang input_vector_rf punya 5k fitur

                        rf_prediction = rf_model.predict(input_vector_rf)
                        label_rf = "Pelecehan" if rf_prediction[0] == 1 else "Non-Pelecehan"

                        if hasattr(rf_model, 'predict_proba') and callable(getattr(rf_model, 'predict_proba')):
                            rf_proba = rf_model.predict_proba(input_vector_rf)
                            st.write(f"**Model Random Forest (5k fitur):** {label_rf} (Probabilitas: {rf_proba[0][rf_prediction[0]]:.2f})")
                        else:
                            st.write(f"**Model Random Forest (5k fitur):** {label_rf} (Probabilitas tidak tersedia)")
                    except ValueError as ve:
                         st.error(f"Error dimensi saat prediksi Random Forest: {ve}. Pastikan selector menghasilkan jumlah fitur yang sesuai dengan model RF.") # More specific error
                    except Exception as e:
                        st.error(f"Error prediksi Random Forest: {e}")
                elif rf_model is None:
                     st.warning("Model Random Forest tidak dimuat, prediksi RF dilewati.")
                elif selector is None: # Added check for selector missing
                     st.warning("Feature selector tidak dimuat, prediksi RF dilewati.")

            except ValueError as ve:
                 if "actual number of features 10000 does not match expected number of features 5000" in str(ve):
                     st.error(f"Error transformasi TF-IDF: {ve}. Sepertinya vectorizer yang dimuat tidak sesuai dengan yang diharapkan (mungkin tertukar dengan selector?).")
                 else:
                     st.error(f"Error saat transformasi TF-IDF: {ve}")
            except Exception as e:
                st.error(f"Error saat transformasi teks atau prediksi awal: {e}")

st.markdown("---")
st.caption("Dikembangkan menggunakan Streamlit, NLTK, dan model ML dari scratch.")


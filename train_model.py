# training.py
import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from my_model import NaiveBayes, RandomForest

# 1. Muat dataset
data = pd.read_csv('nh_h_all.csv')  # Ganti dengan path ke file CSV kamu

# 2. Pisahkan fitur (X) dan label (y)
X = data.drop('label', axis=1)  # Ganti 'label' dengan nama kolom target di dataset kamu
y = data['label']

# 3. Bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# - test_size=0.2 berarti 20% data untuk testing, 80% untuk training
# - random_state=42 untuk konsistensi hasil

# 4. Inisialisasi dan latih model Naive Bayes
nb = NaiveBayes(alpha=1.0)
nb.fit(X_train, y_train)  # Latih model Naive Bayes

# 5. Inisialisasi dan latih model Random Forest
rf = RandomForest(
    n_trees=40,
    max_depth=10,
    min_samples_split=20,
    n_features_subset=0.4,  # Pakai 40% fitur tiap tree
    max_samples=0.8,        # Pakai 80% sampel tiap bootstrap
)
rf.fit(X_train, y_train)  # Latih model Random Forest

# 6. Evaluasi model (opsional, tapi disarankan)
nb_preds = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_preds)
print(f"Akurasi Naive Bayes: {nb_accuracy}")

rf_preds = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)
print(f"Akurasi Random Forest: {rf_accuracy}")

# 7. Simpan model yang sudah dilatih
os.makedirs('models', exist_ok=True)  # Buat folder 'models' jika belum ada
with open("models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(nb, f)
with open("models/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model telah disimpan!")
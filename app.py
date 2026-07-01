from flask import Flask, render_template, request

import pandas as pd
import re
import string
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =====================================================
# FLASK
# =====================================================

app = Flask(__name__)

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("MEMBACA DATASET")
print("=" * 60)

DATASET_PATH = "dataset/Twitter_Emotion_Dataset.csv"
KAMUS_PATH = "dataset/kamus_singkatan.csv"
CLEAN_DATASET = "dataset/clean_dataset.csv"

df = pd.read_csv(DATASET_PATH)

df.rename(
    columns={
        "tweet": "text",
        "label": "emotion"
    },
    inplace=True
)

print(df.head())
print("\nJumlah Data :", len(df))

# =====================================================
# LOAD KAMUS SINGKATAN
# =====================================================

kamus = pd.read_csv(
    KAMUS_PATH,
    sep=";",
    names=["slang", "formal"]
)

normalization_dict = dict(
    zip(
        kamus["slang"],
        kamus["formal"]
    )
)

print("Jumlah Kamus :", len(normalization_dict))

# =====================================================
# SASTRAWI
# =====================================================

stop_factory = StopWordRemoverFactory()
stopword = stop_factory.create_stop_word_remover()

stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

# =====================================================
# PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text).lower()

    # hapus url
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # hapus mention
    text = re.sub(r"@\w+", "", text)

    # hapus hashtag
    text = re.sub(r"#\w+", "", text)

    # hapus angka
    text = re.sub(r"\d+", "", text)

    # hapus tanda baca
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()

    # tokenisasi
    words = text.split()

    hasil = []

    for word in words:

        word = normalization_dict.get(word, word)

        hasil.append(word)

    text = " ".join(hasil)

    # Stopword Removal
    text = stopword.remove(text)

    # Tidak memakai stemming
    return text

    # stemming
    text = stemmer.stem(text)

    return text

# =====================================================
# CEK CLEAN DATASET
# =====================================================

if os.path.exists(CLEAN_DATASET):

    print()
    print("=" * 60)
    print("CLEAN DATASET SUDAH ADA")
    print("=" * 60)

    df = pd.read_csv(CLEAN_DATASET)

else:

    print()
    print("=" * 60)
    print("MEMPROSES DATASET...")
    print("=" * 60)

    df["clean_text"] = df["text"].apply(clean_text)

    df.to_csv(
        CLEAN_DATASET,
        index=False
    )

    print("Clean Dataset berhasil disimpan.")

print()
print(df.head())

# =====================================================
# SPLIT DATASET
# =====================================================

train_df, temp_df = train_test_split(

    df,

    test_size=0.20,

    random_state=42,

    stratify=df["emotion"]

)

val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=42,

    stratify=temp_df["emotion"]

)

print()

print("=" * 60)
print("JUMLAH DATA")
print("=" * 60)

print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))

print()

print(train_df["emotion"].value_counts())

# =====================================================
# TF-IDF & MODEL
# =====================================================

MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# =====================================================
# CEK MODEL
# =====================================================

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):

    print()
    print("=" * 60)
    print("MODEL SUDAH ADA")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    print("Model berhasil dimuat.")
    print("Vectorizer berhasil dimuat.")

else:

    print()
    print("=" * 60)
    print("MODEL BELUM ADA")
    print("=" * 60)

    print("Membuat TF-IDF...")

    vectorizer = TfidfVectorizer(

        max_features=3000,

        ngram_range=(1,1),

        min_df=3
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])

    X_val = vectorizer.transform(val_df["clean_text"])

    X_test = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["emotion"]

    y_val = val_df["emotion"]

    y_test = test_df["emotion"]

    print("Training Logistic Regression...")

    model = LogisticRegression(

        max_iter=1000,

        random_state=42

    )

    model.fit(

        X_train,

        y_train

    )

    print("Training selesai.")

    joblib.dump(

        model,

        MODEL_PATH

    )

    joblib.dump(

        vectorizer,

        VECTORIZER_PATH

    )

    print()

    print("Model berhasil disimpan.")

    print("Vectorizer berhasil disimpan.")

# =====================================================
# EVALUASI
# =====================================================

print()

print("=" * 60)
print("EVALUASI MODEL")
print("=" * 60)

X_val = vectorizer.transform(val_df["clean_text"])

X_test = vectorizer.transform(test_df["clean_text"])

y_val = val_df["emotion"]

y_test = test_df["emotion"]

val_pred = model.predict(X_val)

test_pred = model.predict(X_test)

validation_accuracy = accuracy_score(

    y_val,

    val_pred

)

testing_accuracy = accuracy_score(

    y_test,

    test_pred

)

print()

print("Validation Accuracy : {:.4f}".format(validation_accuracy))

print("Testing Accuracy    : {:.4f}".format(testing_accuracy))

print()

print("Classification Report")

print()

print(

    classification_report(

        y_test,

        test_pred

    )

)

# =====================================================
# EMOJI
# =====================================================

emoji = {

    "happy": "😊",

    "sadness": "😢",

    "anger": "😠",

    "fear": "😨",

    "love": "❤️",

    "surprise": "😲",

    "neutral": "😐"

}

# =====================================================
# FUNGSI PREDIKSI
# =====================================================

def predict_emotion(text):

    clean = clean_text(text)

    vector = vectorizer.transform(

        [clean]

    )

    prediction = model.predict(

        vector

    )[0]

    probability = model.predict_proba(

        vector

    )[0]

    confidence = round(

        max(probability) * 100,

        2

    )

    probability_dict = {}

    for label, prob in zip(

        model.classes_,

        probability

    ):

        probability_dict[label] = round(

            prob * 100,

            2

        )

    return (

        prediction,

        confidence,

        probability_dict

    )

print()

print("=" * 60)
print("MODEL SIAP DIGUNAKAN")
print("=" * 60)

# =====================================================
# ROUTE HOME
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    probability = None
    icon = ""
    user_text = ""

    if request.method == "POST":

        user_text = request.form.get("text", "").strip()

        if user_text != "":

            prediction, confidence, probability = predict_emotion(user_text)

            icon = emoji.get(prediction, "🙂")

    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        probability=probability,

        emoji=icon,

        user_text=user_text

    )

# =====================================================
# ABOUT
# =====================================================

@app.route("/about")
def about():

    return render_template("about.html")

# =====================================================
# API PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict_api():

    text = request.form.get("text", "").strip()

    if text == "":

        return {
            "status": "error",
            "message": "Teks tidak boleh kosong"
        }

    prediction, confidence, probability = predict_emotion(text)

    return {

        "status": "success",

        "text": text,

        "prediction": prediction,

        "confidence": confidence,

        "probability": probability

    }

# =====================================================
# ERROR PAGE
# =====================================================

@app.errorhandler(404)
def page_not_found(e):

    return (

        "<h2>404 | Halaman tidak ditemukan</h2>",

        404

    )

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print()

    print("=" * 60)
    print("FLASK BERHASIL DIJALANKAN")
    print("=" * 60)

    print()

    print("Buka browser:")

    print("http://127.0.0.1:5000")

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )
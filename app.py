
# ============================================================
# PROJECT: IMDb MOVIE REVIEW SENTIMENT ANALYSIS
# TECHNIQUES: WORD2VEC (SKIP-GRAM) + ANN
# ============================================================

# 1. IMPORT LIBRARIES

import os
import re
import random
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from gensim.models import Word2Vec

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 2. CONFIGURATION
# ============================================================

SEED = 42
VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 2
EPOCHS_WORD2VEC = 10
EPOCHS_ANN = 15
BATCH_SIZE = 64

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("NumPy Version:", np.__version__)
print("TensorFlow Version:", tf.__version__)


# ============================================================
# 3. LOAD DATASET
# ============================================================

DATASET_PATH = "IMDB Dataset 2.csv"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)
print("Dataset Columns:", df.columns.tolist())

df.columns = df.columns.str.strip().str.lower()

if "review" not in df.columns or "sentiment" not in df.columns:
    raise ValueError(
        "Dataset must contain review and sentiment columns."
    )

df = df[["review", "sentiment"]].dropna().copy()

df["review"] = df["review"].astype(str)
df["sentiment"] = (
    df["sentiment"].astype(str).str.lower().str.strip()
)

df = df[
    df["sentiment"].isin(["positive", "negative"])
].reset_index(drop=True)

print("\nFinal Dataset Shape:", df.shape)
print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_review"] = df["review"].apply(clean_text)

print("\nText Cleaning Completed!")


# ============================================================
# 5. TOKENIZATION AND STOPWORD REMOVAL
# ============================================================

# Preserve negation words for sentiment meaning

negation_words = {
    "no",
    "not",
    "nor",
    "never",
    "neither",
    "none",
    "cannot",
    "without"
}

stop_words = set(ENGLISH_STOP_WORDS) - negation_words


def tokenize_text(text):

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


df["tokens"] = df["clean_review"].apply(tokenize_text)

print("\nTokenization Completed!")
print("Sample Tokens:", df["tokens"].iloc[0][:20])


# ============================================================
# 6. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df["sentiment"])

print("\nLabel Encoder Classes:")
print(label_encoder.classes_)

if list(label_encoder.classes_) != ["negative", "positive"]:
    raise ValueError("Unexpected sentiment class mapping.")


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_tokens = df["tokens"].tolist()

X_train_tokens, X_test_tokens, y_train, y_test = train_test_split(
    X_tokens,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nTraining Records:", len(X_train_tokens))
print("Testing Records:", len(X_test_tokens))


# ============================================================
# 8. WORD2VEC MODEL - SKIP GRAM
# ============================================================

print("\nTraining Word2Vec Model...")

word2vec_model = Word2Vec(
    sentences=X_train_tokens,
    vector_size=VECTOR_SIZE,
    window=WINDOW_SIZE,
    min_count=MIN_COUNT,
    workers=4,
    sg=1,
    epochs=EPOCHS_WORD2VEC,
    seed=SEED
)

print("\nWord2Vec Training Completed!")

print(
    "Vocabulary Size:",
    len(word2vec_model.wv)
)


# ============================================================
# 9. SAVE WORD2VEC MODEL
# ============================================================

word2vec_model.save("word2vec.model")

print("\nWord2Vec Model Saved!")


# ============================================================
# 10. SENTENCE EMBEDDING FUNCTION
# ============================================================

def get_sentence_embedding(tokens):

    vectors = []

    for word in tokens:

        if word in word2vec_model.wv:

            vectors.append(word2vec_model.wv[word])

    if len(vectors) == 0:

        return np.zeros(
            VECTOR_SIZE,
            dtype=np.float32
        )

    sentence_vector = np.mean(
        vectors,
        axis=0
    )

    return sentence_vector.astype(np.float32)


# ============================================================
# 11. GENERATE SENTENCE EMBEDDINGS
# ============================================================

print("\nGenerating Training Embeddings...")

X_train = np.array(
    [
        get_sentence_embedding(tokens)
        for tokens in X_train_tokens
    ],
    dtype=np.float32
)

print("Generating Testing Embeddings...")

X_test = np.array(
    [
        get_sentence_embedding(tokens)
        for tokens in X_test_tokens
    ],
    dtype=np.float32
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)


# ============================================================
# 12. BUILD ARTIFICIAL NEURAL NETWORK
# ============================================================

model = Sequential([

    Input(shape=(VECTOR_SIZE,)),

    Dense(128, activation="relu"),

    Dropout(0.3),

    Dense(64, activation="relu"),

    Dropout(0.2),

    Dense(1, activation="sigmoid")

])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nANN Model Summary:")
model.summary()


# ============================================================
# 13. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


# ============================================================
# 14. TRAIN ANN
# ============================================================

print("\nANN Model Training Started...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=EPOCHS_ANN,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping],
    verbose=1
)

print("\nANN Training Completed!")


# ============================================================
# 15. MODEL EVALUATION
# ============================================================

print("\nEvaluating ANN Model...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTest Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")


# ============================================================
# 16. PREDICTION AND CLASSIFICATION REPORT
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_pred = (y_probability >= 0.5).astype(int)

print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 17. SAVE ANN MODEL
# ============================================================

model.save("word2vec_ann.keras")

print("\nANN Model Saved Successfully!")


# ============================================================
# 18. SAVE LABEL ENCODER AND STOPWORDS
# ============================================================

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

joblib.dump(
    stop_words,
    "stop_words.pkl"
)

print("\nLabel Encoder Saved!")
print("Stopwords Saved!")


# ============================================================
# 19. CUSTOM REVIEW PREDICTION
# ============================================================

def predict_sentiment(review):

    cleaned_review = clean_text(review)

    tokens = tokenize_text(cleaned_review)

    embedding = get_sentence_embedding(tokens)

    embedding = embedding.reshape(1, VECTOR_SIZE)

    probability = float(
        model.predict(
            embedding,
            verbose=0
        )[0][0]
    )

    predicted_class = int(probability >= 0.5)

    sentiment = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    )

    return {
        "Sentiment": sentiment.upper(),
        "Confidence": round(confidence * 100, 2),
        "Positive Probability": round(probability * 100, 2),
        "Negative Probability": round(
            (1 - probability) * 100, 2
        )
    }


# ============================================================
# 20. TEST REVIEWS
# ============================================================

positive_review = """
This movie is amazing. The acting is excellent.
The story is wonderful and I really enjoyed it.
"""

negative_review = """
This movie is boring and terrible.
The acting is bad and I did not enjoy watching it.
"""

print("\nPositive Review Prediction:")
print(predict_sentiment(positive_review))

print("\nNegative Review Prediction:")
print(predict_sentiment(negative_review))


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n==========================================")
print("WORD2VEC + ANN TRAINING COMPLETED")
print("==========================================")

print("""
Generated Model Files:

1. word2vec.model
2. word2vec_ann.keras
3. label_encoder.pkl
4. stop_words.pkl

Project Training Completed Successfully!
""")


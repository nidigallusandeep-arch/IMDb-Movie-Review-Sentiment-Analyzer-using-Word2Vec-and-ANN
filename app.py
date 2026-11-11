
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import re
import random

from gensim.models import KeyedVectors


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="IMDb Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #a78bfa;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 20px;
}

.positive {
    background-color: #064e3b;
    border: 2px solid #10b981;
}

.negative {
    background-color: #7f1d1d;
    border: 2px solid #ef4444;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🎬 IMDb Movie Review Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by Word2Vec + Artificial Neural Network</div>',
    unsafe_allow_html=True
)

st.write("")


# ==========================================
# 4. SESSION STATE
# ==========================================

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ==========================================
# 5. SAMPLE REVIEWS
# ==========================================

sample_reviews = {
    "Positive Review":
        "This movie is absolutely fantastic. The acting was excellent and the story was amazing. I really enjoyed watching it.",

    "Negative Review":
        "This movie was terrible and boring. The acting was poor and the story was disappointing. I did not enjoy it.",

    "Positive Review 2":
        "Wonderful movie with great performances. The direction and screenplay were excellent.",

    "Negative Review 2":
        "Worst movie I have ever watched. Very slow story and terrible acting. Complete waste of time."
}


# ==========================================
# 6. CALLBACK FUNCTIONS
# ==========================================

def load_sample_review():

    selected = st.session_state.sample_choice

    st.session_state.review_text = sample_reviews[selected]

    st.session_state.prediction_result = None


def random_review():

    selected = random.choice(
        list(sample_reviews.values())
    )

    st.session_state.review_text = selected

    st.session_state.prediction_result = None


def clear_review():

    st.session_state.review_text = ""

    st.session_state.prediction_result = None


def clear_history():

    st.session_state.history = []


# ==========================================
# 7. LOAD TRAINED MODELS
# ==========================================

@st.cache_resource
def load_models():

    # Load ANN model
    ann_model = tf.keras.models.load_model(
        "word2vec_ann.keras",
        compile=False
    )

    # Load Word2Vec using memory mapping
    word2vec_model = KeyedVectors.load(
        "word2vec.kv",
        mmap="r"
    )

    # Load Label Encoder
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    # Load Stopwords
    with open("stop_words.pkl", "rb") as f:
        stop_words = pickle.load(f)

    return (
        ann_model,
        word2vec_model,
        label_encoder,
        stop_words
    )


# ==========================================
# 8. MODEL INITIALIZATION
# ==========================================

try:

    with st.spinner("Loading trained models..."):

        (
            ann_model,
            word2vec_model,
            label_encoder,
            stop_words
        ) = load_models()

    st.success("All models loaded successfully!")

except Exception as e:

    st.error(f"Model loading error: {e}")

    st.info(
        "Please check model files and requirements.txt."
    )

    st.stop()


# ==========================================
# 9. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):

    cleaned_text = clean_text(text)

    tokens = cleaned_text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


# ==========================================
# 10. WORD2VEC SENTENCE EMBEDDING
# ==========================================

def get_sentence_embedding(tokens):

    vectors = []

    for word in tokens:

        if word in word2vec_model:

            vectors.append(word2vec_model[word])

    if len(vectors) == 0:

        return np.zeros(
            word2vec_model.vector_size,
            dtype=np.float32
        )

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)


# ==========================================
# 11. SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(review):

    tokens = tokenize_text(review)

    if len(tokens) == 0:

        raise ValueError(
            "No valid words found after preprocessing."
        )

    # Word2Vec embedding
    embedding = get_sentence_embedding(tokens)

    # ANN input
    embedding = np.expand_dims(
        embedding,
        axis=0
    )

    embedding = embedding.astype(np.float32)

    # ANN prediction
    prediction_probability = float(
        ann_model.predict(
            embedding,
            verbose=0
        ).reshape(-1)[0]
    )

    # Convert probability into class index
    predicted_index = (
        1 if prediction_probability >= 0.5 else 0
    )

    # Label Encoder
    predicted_label = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Probabilities
    positive_probability = prediction_probability * 100

    negative_probability = (
        1 - prediction_probability
    ) * 100

    confidence = max(
        positive_probability,
        negative_probability
    )

    return {
        "sentiment": predicted_label,
        "confidence": confidence,
        "positive_probability": positive_probability,
        "negative_probability": negative_probability
    }


# ==========================================
# 12. SAMPLE REVIEW SECTION
# ==========================================

st.subheader("📝 Try a Sample Review")

col1, col2 = st.columns([3, 1])

with col1:

    st.selectbox(
        "Choose a sample review",
        list(sample_reviews.keys()),
        key="sample_choice"
    )

with col2:

    st.write("")

    st.button(
        "Load Sample",
        on_click=load_sample_review,
        use_container_width=True
    )


# ==========================================
# 13. REVIEW INPUT
# ==========================================

st.subheader("Enter Your Movie Review")

review = st.text_area(
    "Write your review here:",
    key="review_text",
    height=150,
    placeholder="Example: This movie was amazing..."
)


# ==========================================
# 14. ACTION BUTTONS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    predict_button = st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    )

with col2:

    st.button(
        "🎲 Random Review",
        on_click=random_review,
        use_container_width=True
    )

with col3:

    st.button(
        "🗑️ Clear",
        on_click=clear_review,
        use_container_width=True
    )


# ==========================================
# 15. PREDICTION RESULT
# ==========================================

if predict_button:

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        try:

            with st.spinner("Analyzing your review..."):

                result = predict_sentiment(review)

                st.session_state.prediction_result = result

                # Save history
                st.session_state.history.append({
                    "Review": review,
                    "Sentiment": result["sentiment"],
                    "Confidence": round(
                        result["confidence"], 2
                    )
                })

        except Exception as e:

            st.error(f"Prediction error: {e}")


# ==========================================
# 16. DISPLAY RESULT
# ==========================================

result = st.session_state.prediction_result

if result is not None:

    sentiment = result["sentiment"]

    if sentiment.lower() == "positive":

        st.markdown(
            f"""
            <div class="result-box positive">
                <h1>😊 Positive Review</h1>
                <h3>Confidence: {result["confidence"]:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="result-box negative">
                <h1>😞 Negative Review</h1>
                <h3>Confidence: {result["confidence"]:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # Probability metrics
    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Positive Probability",
            f'{result["positive_probability"]:.2f}%'
        )

        st.progress(
            result["positive_probability"] / 100
        )

    with col2:

        st.metric(
            "Negative Probability",
            f'{result["negative_probability"]:.2f}%'
        )

        st.progress(
            result["negative_probability"] / 100
        )


# ==========================================
# 17. PREDICTION HISTORY
# ==========================================

st.divider()

st.subheader("📊 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download History CSV",
        data=csv_data,
        file_name="sentiment_prediction_history.csv",
        mime="text/csv"
    )

    st.button(
        "Clear History",
        on_click=clear_history
    )

else:

    st.info(
        "No predictions yet. Enter a review to get started."
    )


# ==========================================
# 18. FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <center>
        <p style="color:#a78bfa;">
            IMDb Movie Sentiment Analyzer |
            Word2Vec + ANN |
            Built with Streamlit
        </p>
    </center>
    """,
    unsafe_allow_html=True
)

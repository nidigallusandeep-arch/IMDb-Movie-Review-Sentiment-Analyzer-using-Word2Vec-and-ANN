
import streamlit as st
import numpy as np
import pandas as pd
import re
import random
import joblib
import tensorflow as tf

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e1b4b, #312e81);
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #67e8f9;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #172554 !important;
    color: white !important;
    border: 1px solid #06b6d4 !important;
    border-radius: 12px;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #172554;
    color: white;
}

.stButton button {
    background: linear-gradient(90deg, #0891b2, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    width: 100%;
}

.positive-message {
    background: linear-gradient(90deg, #047857, #0891b2);
    color: white !important;
    border: 1px solid #5eead4;
    border-radius: 12px;
    padding: 18px;
    font-size: 17px;
    font-weight: bold;
    margin: 15px 0;
}

.negative-message {
    background: linear-gradient(90deg, #be123c, #ea580c);
    color: white !important;
    border: 1px solid #fda4af;
    border-radius: 12px;
    padding: 18px;
    font-size: 17px;
    font-weight: bold;
    margin: 15px 0;
}

.metric-card {
    background: #172554;
    border: 1px solid #06b6d4;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# TITLE
# ==========================================

st.markdown(
    '<div class="main-title">🎬 Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Analyze movie reviews using Word2Vec and Artificial Neural Networks</div>',
    unsafe_allow_html=True
)


# ==========================================
# LOAD MODEL AND ARTIFACTS
# ==========================================

@st.cache_resource
def load_artifacts():

    # Display versions in logs if needed
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", tf.keras.__version__)

    # Load ANN model
    ann_model = load_model(
        "word2vec_ann.keras",
        compile=False
    )

    # Load Word2Vec model
    w2v_model = Word2Vec.load("word2vec.model")

    # Load Label Encoder
    label_encoder = joblib.load("label_encoder.pkl")

    # Load stopwords
    stop_words = joblib.load("stop_words.pkl")

    return ann_model, w2v_model, label_encoder, stop_words


try:
    model, word2vec_model, label_encoder, stop_words = load_artifacts()

    st.success("✅ All models and files loaded successfully!")

except Exception as e:

    st.error(f"Model loading error: {e}")

    st.stop()


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ==========================================
# TOKENIZATION
# ==========================================

def tokenize_text(text):

    cleaned_text = clean_text(text)

    tokens = cleaned_text.split()

    # Preserve negation words
    negation_words = {
        "no", "not", "nor", "never", "neither",
        "none", "cannot", "without"
    }

    tokens = [
        word for word in tokens
        if word not in stop_words or word in negation_words
    ]

    return tokens


# ==========================================
# WORD2VEC SENTENCE EMBEDDING
# ==========================================

def get_sentence_embedding(text):

    tokens = tokenize_text(text)

    vectors = []

    for word in tokens:

        if word in word2vec_model.wv:

            vectors.append(word2vec_model.wv[word])

    if len(vectors) == 0:

        return np.zeros(100, dtype=np.float32)

    embedding = np.mean(vectors, axis=0)

    return embedding.astype(np.float32)


# ==========================================
# SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(review):

    embedding = get_sentence_embedding(review)

    embedding = embedding.reshape(1, -1)

    prediction = model.predict(
        embedding,
        verbose=0
    )

    probability = float(np.asarray(prediction).reshape(-1)[0])

    # ANN sigmoid predicts probability for class index 1
    predicted_class_index = 1 if probability >= 0.5 else 0

    sentiment = str(
        label_encoder.inverse_transform(
            [predicted_class_index]
        )[0]
    )

    confidence = (
        probability if predicted_class_index == 1
        else 1 - probability
    )

    return sentiment.upper(), confidence, probability


# ==========================================
# SAMPLE REVIEWS
# ==========================================

sample_reviews = {

    "Positive Review":
    "This movie is absolutely fantastic. The acting is brilliant, "
    "the story is interesting and the direction is excellent. "
    "I really enjoyed watching this movie.",

    "Negative Review":
    "This movie is very boring and disappointing. "
    "The story is weak, the acting is terrible and "
    "I did not enjoy watching it at all."
}


# ==========================================
# SESSION STATE
# ==========================================

if "movie_review" not in st.session_state:
    st.session_state.movie_review = ""

if "history" not in st.session_state:
    st.session_state.history = []


def load_sample():

    selected = st.session_state.sample_option

    st.session_state.movie_review = sample_reviews[selected]


def random_review():

    st.session_state.movie_review = random.choice(
        list(sample_reviews.values())
    )


def clear_review():

    st.session_state.movie_review = ""


# ==========================================
# SAMPLE REVIEW SELECTION
# ==========================================

st.subheader("📝 Enter Movie Review")

col1, col2 = st.columns([2, 1])

with col1:

    st.selectbox(
        "Choose Sample Review",
        list(sample_reviews.keys()),
        key="sample_option"
    )

with col2:

    st.write("")

    st.button(
        "Load Sample Review",
        on_click=load_sample
    )


# ==========================================
# REVIEW INPUT
# ==========================================

review = st.text_area(
    "Write your movie review here:",
    key="movie_review",
    height=180,
    placeholder="Enter your movie review..."
)


# ==========================================
# BUTTONS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    predict_button = st.button(
        "🔍 Predict Sentiment"
    )

with col2:

    st.button(
        "🎲 Random Review",
        on_click=random_review
    )

with col3:

    st.button(
        "🗑️ Clear Review",
        on_click=clear_review
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        with st.spinner("Analyzing movie review..."):

            try:

                sentiment, confidence, positive_probability = (
                    predict_sentiment(review)
                )

                negative_probability = 1 - positive_probability

                st.markdown("---")

                st.subheader("📊 Prediction Result")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Sentiment</h3>
                            <h2>{sentiment}</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{confidence * 100:.2f}%</h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                if sentiment == "POSITIVE":

                    st.markdown(
                        """
                        <div class="positive-message">
                            😊 This movie review has POSITIVE sentiment!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        """
                        <div class="negative-message">
                            😞 This movie review has NEGATIVE sentiment!
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.subheader("Sentiment Probability")

                st.write(
                    f"Positive: {positive_probability * 100:.2f}%"
                )

                st.progress(positive_probability)

                st.write(
                    f"Negative: {negative_probability * 100:.2f}%"
                )

                st.progress(negative_probability)

                # Save prediction history
                st.session_state.history.append({

                    "Review": review,

                    "Sentiment": sentiment,

                    "Confidence": round(
                        confidence * 100, 2
                    )

                })

            except Exception as e:

                st.error(f"Prediction error: {e}")


# ==========================================
# PREDICTION HISTORY
# ==========================================

st.markdown("---")

st.subheader("📚 Prediction History")

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
        "⬇️ Download Prediction History",
        data=csv_data,
        file_name="movie_sentiment_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info("No prediction history available yet.")


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        🎬 Movie Sentiment Analyzer |
        Word2Vec + Artificial Neural Network
        <br>
        Built using Python and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

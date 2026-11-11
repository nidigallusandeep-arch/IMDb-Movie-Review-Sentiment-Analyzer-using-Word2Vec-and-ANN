
# ============================================================
# IMDb MOVIE SENTIMENT ANALYZER
# WORD2VEC + ARTIFICIAL NEURAL NETWORK
# STREAMLIT WEB APPLICATION
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import re
import random

from gensim.models import KeyedVectors


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e1b4b,
        #312e81
    );
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.result-card {
    background: rgba(255,255,255,0.10);
    padding: 22px;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.2);
    text-align: center;
}

.positive {
    background: rgba(34,197,94,0.18);
    border: 1px solid #22c55e;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.negative {
    background: rgba(239,68,68,0.18);
    border: 1px solid #ef4444;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}

.footer {
    text-align: center;
    color: #cbd5e1;
    padding: 20px;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_models():

    # Load ANN model
    model = tf.keras.models.load_model(
        "word2vec_ann.keras",
        compile=False
    )

    # Load Word2Vec vectors without old pickle RNG state
    word2vec_model = KeyedVectors.load_word2vec_format(
        "word2vec_vectors.txt",
        binary=False
    )

    # Load Label Encoder
    label_encoder = joblib.load(
        "label_encoder.pkl"
    )

    # Load Stopwords
    stop_words = joblib.load(
        "stop_words.pkl"
    )

    # Validate model dimensions
    if word2vec_model.vector_size != 100:
        raise ValueError(
            "Word2Vec vector size must be 100."
        )

    if model.input_shape[-1] != 100:
        raise ValueError(
            "ANN input dimension must be 100."
        )

    return (
        model,
        word2vec_model,
        label_encoder,
        stop_words
    )


try:

    (
        model,
        word2vec_model,
        label_encoder,
        stop_words
    ) = load_models()

    st.sidebar.success("All models loaded successfully!")

except Exception as e:

    st.error(f"Model loading error: {e}")

    st.info(
        "Check whether all model files are available "
        "in your GitHub repository."
    )

    st.stop()


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Remove special characters and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# 5. TOKENIZATION
# ============================================================

def tokenize_text(text):

    cleaned_text = clean_text(text)

    tokens = cleaned_text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    return tokens


# ============================================================
# 6. SENTENCE EMBEDDING
# ============================================================

def get_sentence_embedding(tokens):

    vectors = []

    for word in tokens:

        if word in word2vec_model:
            vectors.append(word2vec_model[word])

    if len(vectors) == 0:
        return np.zeros(100, dtype=np.float32)

    return np.mean(
        vectors,
        axis=0
    ).astype(np.float32)


# ============================================================
# 7. SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(review):

    tokens = tokenize_text(review)

    embedding = get_sentence_embedding(tokens)

    embedding = embedding.reshape(1, 100)

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

    positive_probability = probability * 100

    negative_probability = (1 - probability) * 100

    confidence = (
        probability
        if predicted_class == 1
        else 1 - probability
    ) * 100

    return {
        "sentiment": sentiment.upper(),
        "confidence": round(confidence, 2),
        "positive_probability": round(
            positive_probability, 2
        ),
        "negative_probability": round(
            negative_probability, 2
        )
    }


# ============================================================
# 8. APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered IMDb Movie Review Classification '
    'Using Word2Vec and Artificial Neural Network'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 9. SAMPLE REVIEWS
# ============================================================

positive_samples = [

    "This movie is absolutely amazing. The acting and story are excellent.",

    "I really enjoyed this film. The direction and performances were fantastic.",

    "One of the best movies I have ever watched. Highly recommended.",

    "Wonderful movie with excellent acting and beautiful storytelling."

]

negative_samples = [

    "This movie is boring and terrible. I did not enjoy watching it.",

    "Very disappointing film. The acting was poor and the story was bad.",

    "Worst movie ever. Completely wasted my time.",

    "The film was slow, boring and not worth watching."

]


# ============================================================
# 10. SAMPLE REVIEW SELECTOR
# ============================================================

st.subheader("📝 Try Sample Movie Reviews")

sample_type = st.selectbox(
    "Choose Sample Review",
    [
        "Select a sample",
        "Positive Review",
        "Negative Review"
    ]
)

if st.button("Load Sample Review"):

    if sample_type == "Positive Review":

        st.session_state.review_text = random.choice(
            positive_samples
        )

    elif sample_type == "Negative Review":

        st.session_state.review_text = random.choice(
            negative_samples
        )

    else:

        st.warning("Please select a review type.")


# ============================================================
# 11. REVIEW INPUT
# ============================================================

st.subheader("✍️ Enter Your Movie Review")

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

review = st.text_area(
    "Write your movie review here:",
    key="review_text",
    height=180,
    placeholder="Example: This movie was amazing..."
)


# ============================================================
# 12. BUTTONS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    predict_button = st.button(
        "🔍 Predict Sentiment",
        use_container_width=True
    )

with col2:

    random_button = st.button(
        "🎲 Random Review",
        use_container_width=True
    )

with col3:

    clear_button = st.button(
        "🗑️ Clear Review",
        use_container_width=True
    )


# ============================================================
# 13. SESSION STATE
# ============================================================

if "history" not in st.session_state:

    st.session_state.history = []


# ============================================================
# 14. RANDOM REVIEW
# ============================================================

if random_button:

    all_samples = positive_samples + negative_samples

    st.session_state.review_text = random.choice(
        all_samples
    )

    st.rerun()


# ============================================================
# 15. CLEAR REVIEW
# ============================================================

if clear_button:

    st.session_state.review_text = ""

    st.rerun()


# ============================================================
# 16. PREDICT SENTIMENT
# ============================================================

if predict_button:

    if not review.strip():

        st.warning(
            "Please enter a movie review first."
        )

    else:

        result = predict_sentiment(review)

        sentiment = result["sentiment"]

        confidence = result["confidence"]

        positive_probability = result[
            "positive_probability"
        ]

        negative_probability = result[
            "negative_probability"
        ]

        st.divider()

        st.subheader("📊 Prediction Results")

        # Positive Result
        if sentiment == "POSITIVE":

            st.markdown(
                """
                <div class="positive">
                    <h2>😊 POSITIVE REVIEW</h2>
                    <p>This movie review expresses positive sentiment.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Negative Result
        else:

            st.markdown(
                """
                <div class="negative">
                    <h2>😞 NEGATIVE REVIEW</h2>
                    <p>This movie review expresses negative sentiment.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # Confidence Cards
        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Predicted Sentiment",
                sentiment
            )

        with c2:

            st.metric(
                "Confidence",
                f"{confidence}%"
            )

        with c3:

            st.metric(
                "Positive Probability",
                f"{positive_probability}%"
            )

        # Probability Bars
        st.subheader("📈 Sentiment Probability")

        st.write(
            f"Positive: {positive_probability}%"
        )

        st.progress(
            int(positive_probability)
        )

        st.write(
            f"Negative: {negative_probability}%"
        )

        st.progress(
            int(negative_probability)
        )

        # Save Prediction History
        st.session_state.history.append({

            "Review": review,

            "Sentiment": sentiment,

            "Confidence": confidence,

            "Positive Probability": positive_probability,

            "Negative Probability": negative_probability

        })


# ============================================================
# 17. PREDICTION HISTORY
# ============================================================

st.divider()

st.subheader("📚 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    # Download History
    csv_data = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Prediction History",
        data=csv_data,
        file_name="movie_sentiment_history.csv",
        mime="text/csv"
    )

    if st.button("Clear Prediction History"):

        st.session_state.history = []

        st.rerun()

else:

    st.info("No prediction history available yet.")


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        <h3>🎬 IMDb Movie Sentiment Analyzer</h3>

        <p>
        Developed using Python, Word2Vec,
        Artificial Neural Network and Streamlit.
        </p>

        <p>
        Machine Learning | Natural Language Processing | Deep Learning
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import re
import random

from gensim.models import Word2Vec
from tensorflow.keras.models import load_model


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background:
        radial-gradient(
            ellipse at 10% 10%,
            #421078 0%,
            transparent 35%
        ),
        radial-gradient(
            ellipse at 90% 90%,
            #30105e 0%,
            transparent 35%
        ),
        linear-gradient(
            135deg,
            #080d2b,
            #101b48,
            #090d2b
        );

    background-attachment: fixed;
    color: white;
}

/* Main Container */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95%;
}

/* Headings */

h1, h2, h3 {
    color: white !important;
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
}

.main-title span {
    color: #22d3ee;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Text Area */

.stTextArea textarea {
    background: rgba(17, 29, 65, 0.95) !important;
    color: white !important;
    border: 2px solid #22d3ee !important;
    border-radius: 12px !important;
}

.stTextArea textarea::placeholder {
    color: #cbd5e1 !important;
}

/* Select Box */

.stSelectbox div[data-baseweb="select"] > div {
    background: #15234d !important;
    color: white !important;
    border: 1px solid #22d3ee !important;
    border-radius: 10px;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #9333ea
    ) !important;

    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    min-height: 42px;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    box-shadow: 0 0 15px #9333ea;
    transform: translateY(-2px);
}

/* Prediction Metric Cards */

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        rgba(20, 40, 85, 0.95),
        rgba(65, 25, 105, 0.85)
    ) !important;

    border: 1px solid #22d3ee !important;
    border-radius: 15px !important;
    padding: 22px !important;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #22d3ee !important;
}

/* POSITIVE MESSAGE */

.positive-message {
    background: linear-gradient(
        90deg,
        #047857,
        #0891b2
    );

    color: white !important;
    border: 1px solid #5eead4;
    border-radius: 12px;
    padding: 18px;
    font-size: 17px;
    font-weight: bold;
    margin: 15px 0;
}

/* NEGATIVE MESSAGE */

.negative-message {
    background: linear-gradient(
        90deg,
        #be123c,
        #ea580c
    );

    color: white !important;
    border: 1px solid #fda4af;
    border-radius: 12px;
    padding: 18px;
    font-size: 17px;
    font-weight: bold;
    margin: 15px 0;
}

/* Progress Bar */

.stProgress > div > div > div > div {
    background: linear-gradient(
        90deg,
        #06b6d4,
        #9333ea
    );
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border: 1px solid #22d3ee;
    border-radius: 10px;
}

/* Footer */

.footer {
    text-align: center;
    color: #cbd5e1;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MODEL FILES
# ==========================================

@st.cache_resource
def load_artifacts():

    model = load_model(
        os.path.join(BASE_DIR, "word2vec_ann.keras")
    )

    word2vec_model = Word2Vec.load(
        os.path.join(BASE_DIR, "word2vec.model")
    )

    label_encoder = joblib.load(
        os.path.join(BASE_DIR, "label_encoder.pkl")
    )

    stop_words = joblib.load(
        os.path.join(BASE_DIR, "stop_words.pkl")
    )

    return model, word2vec_model, label_encoder, stop_words


try:

    model, word2vec_model, label_encoder, stop_words = (
        load_artifacts()
    )

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# 4. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text):

    text = clean_text(text)

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    return tokens


# ==========================================
# 5. WORD2VEC SENTENCE VECTOR
# ==========================================

def get_sentence_vector(tokens):

    vectors = [
        word2vec_model.wv[word]
        for word in tokens
        if word in word2vec_model.wv
    ]

    if len(vectors) == 0:

        return np.zeros(
            word2vec_model.vector_size
        )

    return np.mean(vectors, axis=0)


# ==========================================
# 6. SENTIMENT PREDICTION
# ==========================================

def predict_sentiment(review):

    tokens = preprocess_text(review)

    sentence_vector = get_sentence_vector(tokens)

    input_data = np.array(
        [sentence_vector],
        dtype=np.float32
    )

    probability = float(
        model.predict(input_data, verbose=0)[0][0]
    )

    # Sigmoid probability for class index 1

    predicted_index = int(probability >= 0.5)

    predicted_sentiment = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    classes = [
        str(c).lower()
        for c in label_encoder.classes_
    ]

    positive_index = next(
        (
            i for i, c in enumerate(classes)
            if "positive" in c
        ),
        1
    )

    # Positive probability

    if positive_index == 1:

        positive_probability = probability * 100

    else:

        positive_probability = (1 - probability) * 100

    negative_probability = 100 - positive_probability

    confidence = max(
        positive_probability,
        negative_probability
    )

    return (
        str(predicted_sentiment).upper(),
        confidence,
        positive_probability,
        negative_probability
    )


# ==========================================
# 7. SAMPLE REVIEWS
# ==========================================

positive_reviews = [
    "This movie is amazing. I love the story and acting!",
    "Excellent movie with wonderful performances.",
    "I really enjoyed this film. It was fantastic!",
    "The movie was emotional and beautiful.",
    "Great direction, amazing music, and brilliant acting."
]

negative_reviews = [
    "This movie is very bad. I hate it!",
    "Worst movie ever. Very boring and disappointing!",
    "I did not enjoy this film. The acting was terrible.",
    "The story was boring and completely disappointing.",
    "I wasted my time watching this movie."
]

all_reviews = positive_reviews + negative_reviews


# ==========================================
# 8. SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []

if "movie_review" not in st.session_state:
    st.session_state.movie_review = ""

if "sample_option" not in st.session_state:
    st.session_state.sample_option = "Select a sample review"


def clear_review():

    st.session_state.movie_review = ""


def set_random_review():

    st.session_state.movie_review = random.choice(
        all_reviews
    )


def load_sample_review():

    selected = st.session_state.sample_option

    if selected == "Positive Review":

        st.session_state.movie_review = random.choice(
            positive_reviews
        )

    elif selected == "Negative Review":

        st.session_state.movie_review = random.choice(
            negative_reviews
        )


def clear_history():

    st.session_state.history = []


# ==========================================
# 9. HEADER
# ==========================================

st.markdown("""
<div class="main-title">
    🎬 Movie Sentiment <span>Analyzer</span>
</div>

<div class="subtitle">
    AI Powered Movie Review Classification using Word2Vec + ANN
</div>
""", unsafe_allow_html=True)


# ==========================================
# 10. SAMPLE REVIEW SECTION
# ==========================================

st.markdown("## 📝 Try Sample Reviews")

st.selectbox(
    "Choose a sample review",
    [
        "Select a sample review",
        "Positive Review",
        "Negative Review"
    ],
    key="sample_option"
)

st.button(
    "📥 Load Sample Review",
    on_click=load_sample_review
)


# ==========================================
# 11. REVIEW INPUT
# ==========================================

st.markdown("## 🎥 Enter Your Movie Review")

review = st.text_area(
    "Write your movie review here",
    key="movie_review",
    height=120,
    placeholder="Write your movie review here..."
)


# ==========================================
# 12. ACTION BUTTONS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:

    predict_button = st.button(
        "🚀 Predict Sentiment",
        use_container_width=True
    )

with col2:

    st.button(
        "🎲 Random Review",
        on_click=set_random_review,
        use_container_width=True
    )

with col3:

    st.button(
        "🗑️ Clear Review",
        on_click=clear_review,
        use_container_width=True
    )


# ==========================================
# 13. PREDICTION RESULT
# ==========================================

if predict_button:

    if not review.strip():

        st.warning("Please enter a movie review!")

    else:

        with st.spinner("Analyzing your review..."):

            sentiment, confidence, positive, negative = (
                predict_sentiment(review)
            )

        st.markdown("## 📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Sentiment",
                sentiment
            )

        with col2:

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

        # Positive and Negative Messages

        if sentiment == "POSITIVE":

            st.markdown("""
            <div class="positive-message">
                😊 This movie review has POSITIVE sentiment!
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="negative-message">
                😞 This movie review has NEGATIVE sentiment!
            </div>
            """, unsafe_allow_html=True)

        # Prediction Probabilities

        st.markdown("## 📈 Prediction Probabilities")

        st.write(
            f"😊 Positive: {positive:.2f}%"
        )

        st.progress(
            min(max(positive / 100, 0.0), 1.0)
        )

        st.write(
            f"😞 Negative: {negative:.2f}%"
        )

        st.progress(
            min(max(negative / 100, 0.0), 1.0)
        )

        # Save History

        st.session_state.history.append({

            "Review": review,
            "Sentiment": sentiment,
            "Confidence": round(confidence, 2),
            "Positive Probability": round(positive, 2),
            "Negative Probability": round(negative, 2)

        })


# ==========================================
# 14. PREDICTION HISTORY
# ==========================================

st.markdown("## 📚 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    col1, col2 = st.columns(2)

    with col1:

        st.download_button(
            "⬇️ Download History (CSV)",
            data=csv,
            file_name="movie_sentiment_history.csv",
            mime="text/csv"
        )

    with col2:

        st.button(
            "🗑️ Clear History",
            on_click=clear_history
        )

else:

    st.info("No prediction history available yet.")


# ==========================================
# 15. FOOTER
# ==========================================

st.markdown("""
<div class="footer">

<hr>

🎬 Movie Sentiment Analyzer

<br><br>

Built using Python | Word2Vec | ANN | NLP | Streamlit

</div>
""", unsafe_allow_html=True)

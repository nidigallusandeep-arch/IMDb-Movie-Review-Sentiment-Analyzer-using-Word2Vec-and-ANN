
import streamlit as st
import numpy as np
import pandas as pd
import re
import html
import random
import joblib
import tensorflow as tf

from gensim.models import KeyedVectors


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="IMDb Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 2. PROFESSIONAL CSS
# =====================================================

st.markdown("""
<style>

/* Background */

.stApp {
    background: linear-gradient(
        135deg,
        #0b1020 0%,
        #111827 55%,
        #17152d 100%
    );
    color: #f8fafc;
}

/* FULL WIDTH MAIN CONTENT */

.block-container {
    max-width: 1600px !important;
    width: 100% !important;
    padding: 2rem 4rem 3rem 4rem !important;
    margin: auto !important;
}

/* Header */

.hero {
    background: linear-gradient(
        120deg,
        #312e81,
        #6d28d9,
        #9333ea
    );
    padding: 45px 30px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(109, 40, 217, 0.3);
}

.hero h1 {
    color: white;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 12px;
}

.hero p {
    color: #e9d5ff;
    font-size: 17px;
    margin: 0;
}

/* Section headings */

.section-heading {
    color: #c4b5fd;
    font-size: 27px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 20px;
}

/* Info cards */

.info-card {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    min-height: 130px;
    transition: 0.3s;
}

.info-card:hover {
    border-color: #a78bfa;
    transform: translateY(-3px);
}

.info-card h3 {
    color: #a78bfa;
    font-size: 27px;
    margin: 0;
}

.info-card p {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 10px;
}

/* Result cards */

.result-card {
    padding: 40px 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
    border: 1px solid #475569;
}

.positive-card {
    background: linear-gradient(
        120deg,
        #064e3b,
        #065f46
    );
}

.negative-card {
    background: linear-gradient(
        120deg,
        #7f1d1d,
        #991b1b
    );
}

.result-card h1 {
    color: white;
    font-size: 38px;
    margin: 8px;
}

.result-card p {
    color: #e2e8f0;
    font-size: 16px;
}

/* Input */

.stTextArea textarea {
    background-color: #1e293b !important;
    color: white !important;
    border: 1px solid #64748b !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    min-height: 180px !important;
}

label {
    color: #e2e8f0 !important;
    font-size: 15px !important;
}

/* Select box */

.stSelectbox div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    color: white !important;
    border-color: #64748b !important;
    border-radius: 10px !important;
}

/* Buttons */

.stButton button,
.stDownloadButton button {
    border-radius: 11px;
    min-height: 48px;
    font-size: 15px;
    font-weight: 700;
    border: 1px solid #7c3aed;
    background: linear-gradient(
        90deg,
        #6d28d9,
        #9333ea
    );
    color: white;
    transition: 0.3s;
}

.stButton button:hover,
.stDownloadButton button:hover {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #a855f7
    );
    color: white;
    border-color: #c4b5fd;
}

/* Metric cards */

div[data-testid="stMetric"] {
    background: #1e293b;
    padding: 22px;
    border-radius: 14px;
    border: 1px solid #334155;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

div[data-testid="stMetricValue"] {
    color: #c4b5fd;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

/* Footer */

.footer {
    text-align: center;
    color: #94a3b8;
    padding: 30px;
    font-size: 14px;
}

/* Mobile responsive */

@media only screen and (max-width: 768px) {

    .block-container {
        padding: 1rem !important;
    }

    .hero h1 {
        font-size: 27px;
    }

    .section-heading {
        font-size: 21px;
    }

    .result-card h1 {
        font-size: 28px;
    }

}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. HEADER
# =====================================================

st.markdown("""
<div class="hero">
    <h1>🎬 IMDb Movie Review Sentiment Analyzer</h1>
    <p>
        AI-Powered Movie Review Classification
        <br>
        Word2Vec Embeddings + Artificial Neural Network
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 4. SESSION STATE
# =====================================================

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =====================================================
# 5. SAMPLE REVIEWS
# =====================================================

sample_reviews = {

    "Positive Review 1":
        "This movie is absolutely fantastic. The acting was amazing and the story was wonderful.",

    "Positive Review 2":
        "I really enjoyed this film. Excellent direction, beautiful visuals and great performances.",

    "Positive Review 3":
        "One of the best movies I have ever watched. Highly recommended.",

    "Negative Review 1":
        "This movie was terrible. The story was boring and the acting was very poor.",

    "Negative Review 2":
        "I wasted my time watching this film. Very disappointing and poorly directed.",

    "Negative Review 3":
        "The worst movie ever. Bad screenplay and horrible performances."
}


# =====================================================
# 6. LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    # Load Word2Vec
    word2vec_model = KeyedVectors.load(
        "word2vec.kv",
        mmap="r"
    )

    # Rebuild ANN architecture
    ann_model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(100,)),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            64,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    # Load trained ANN weights
    ann_model.load_weights(
        "word2vec_ann.weights.h5"
    )

    # Load Label Encoder
    label_encoder = joblib.load(
        "label_encoder.pkl"
    )

    # Load stopwords
    stop_words = set(
        joblib.load("stop_words.pkl")
    )

    return (
        word2vec_model,
        ann_model,
        label_encoder,
        stop_words
    )


# =====================================================
# 7. MODEL STATUS
# =====================================================

try:

    (
        word2vec_model,
        ann_model,
        label_encoder,
        stop_words
    ) = load_models()

    classes = [
        str(value).lower()
        for value in label_encoder.classes_
    ]

    if "positive" not in classes or "negative" not in classes:
        raise ValueError(
            "Label encoder must contain positive and negative classes."
        )

    st.success(
        "✅ Word2Vec and ANN models loaded successfully!"
    )

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# =====================================================
# 8. SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("## 🎬 Movie Analyzer")

    st.write(
        "Analyze movie reviews using Artificial Intelligence."
    )

    st.divider()

    st.markdown("### 🧠 Model Architecture")

    st.info(
        """
        **Word2Vec**

        Converts words into 100-dimensional vectors.

        **ANN**

        Dense Layer: 128 neurons

        Dense Layer: 64 neurons

        Output: Sigmoid
        """
    )

    st.divider()

    st.markdown("### 🛠️ Technologies")

    st.write("• Python")
    st.write("• NLP")
    st.write("• Word2Vec")
    st.write("• Artificial Neural Network")
    st.write("• TensorFlow")
    st.write("• Streamlit")

    st.divider()

    st.caption(
        "IMDb Movie Review Sentiment Analyzer"
    )


# =====================================================
# 9. PROJECT OVERVIEW
# =====================================================

st.markdown(
    '<div class="section-heading">🚀 Project Overview</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    <div class="info-card">
        <h3>Word2Vec</h3>
        <p>Word Embedding Technique</p>
    </div>
    """, unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="info-card">
        <h3>ANN</h3>
        <p>Deep Learning Classifier</p>
    </div>
    """, unsafe_allow_html=True)

with col3:

    st.markdown("""
    <div class="info-card">
        <h3>100</h3>
        <p>Embedding Dimensions</p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# 10. TEXT PREPROCESSING
# =====================================================

def preprocess_text(text):

    text = html.unescape(text)

    text = text.lower()

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Stopword removal
    words = [
        word for word in words
        if word not in stop_words
    ]

    return words


# =====================================================
# 11. WORD2VEC EMBEDDING
# =====================================================

def get_sentence_vector(words):

    vectors = []

    for word in words:

        if word in word2vec_model:

            vectors.append(
                word2vec_model[word]
            )

    if len(vectors) == 0:

        return np.zeros(
            100,
            dtype=np.float32
        )

    sentence_vector = np.mean(
        vectors,
        axis=0
    )

    return sentence_vector.astype(
        np.float32
    )


# =====================================================
# 12. SENTIMENT PREDICTION
# =====================================================

def predict_sentiment(review):

    words = preprocess_text(review)

    if not words:

        raise ValueError(
            "Please enter a valid movie review."
        )

    sentence_vector = get_sentence_vector(words)

    input_data = np.array(
        [sentence_vector],
        dtype=np.float32
    )

    prediction = float(
        ann_model.predict(
            input_data,
            verbose=0
        )[0][0]
    )

    classes = [
        str(value).lower()
        for value in label_encoder.classes_
    ]

    # Sigmoid output corresponds to class index 1
    if classes[1] == "positive":

        positive_probability = prediction

    else:

        positive_probability = 1 - prediction

    negative_probability = 1 - positive_probability

    sentiment = (
        "Positive"
        if positive_probability >= 0.5
        else "Negative"
    )

    confidence = max(
        positive_probability,
        negative_probability
    ) * 100

    return {

        "sentiment": sentiment,

        "confidence": confidence,

        "positive_probability": positive_probability * 100,

        "negative_probability": negative_probability * 100

    }


# =====================================================
# 13. REVIEW INPUT
# =====================================================

st.markdown(
    '<div class="section-heading">📝 Analyze Your Movie Review</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter a movie review or select a sample review below."
)

col1, col2 = st.columns([3, 1])

with col1:

    selected_sample = st.selectbox(
        "Choose a sample review",
        list(sample_reviews.keys())
    )

with col2:

    st.write("")

    if st.button(
        "📥 Load Sample",
        use_container_width=True
    ):

        st.session_state.review_text = (
            sample_reviews[selected_sample]
        )

        st.session_state.prediction_result = None

        st.rerun()


col1, col2 = st.columns(2)

with col1:

    if st.button(
        "🎲 Random Review",
        use_container_width=True
    ):

        st.session_state.review_text = random.choice(
            list(sample_reviews.values())
        )

        st.session_state.prediction_result = None

        st.rerun()

with col2:

    if st.button(
        "🧹 Clear Review",
        use_container_width=True
    ):

        st.session_state.review_text = ""

        st.session_state.prediction_result = None

        st.rerun()


review = st.text_area(
    "Your Movie Review",
    key="review_text",
    height=180,
    placeholder="Example: This movie was amazing. The acting and story were excellent..."
)


# =====================================================
# 14. ANALYZE SENTIMENT
# =====================================================

if st.button(
    "🚀 ANALYZE SENTIMENT",
    type="primary",
    use_container_width=True
):

    if not review.strip():

        st.warning(
            "Please enter a movie review first."
        )

    else:

        try:

            result = predict_sentiment(review)

            st.session_state.prediction_result = result

            st.session_state.history.append({

                "Review": review,

                "Sentiment": result["sentiment"],

                "Confidence": round(
                    result["confidence"],
                    2
                ),

                "Positive Probability": round(
                    result["positive_probability"],
                    2
                ),

                "Negative Probability": round(
                    result["negative_probability"],
                    2
                )

            })

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# =====================================================
# 15. RESULT DISPLAY
# =====================================================

result = st.session_state.prediction_result

if result is not None:

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Sentiment Analysis Result</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Sentiment",
            result["sentiment"].upper()
        )

    with col2:

        st.metric(
            "Model Confidence",
            f'{result["confidence"]:.2f}%'
        )

    with col3:

        st.metric(
            "Positive Probability",
            f'{result["positive_probability"]:.2f}%'
        )

    if result["sentiment"] == "Positive":

        st.markdown(f"""
        <div class="result-card positive-card">
            <h1>😊 POSITIVE REVIEW</h1>
            <p>The model predicts a positive movie review.</p>
            <h1>{result["confidence"]:.2f}%</h1>
            <p>Prediction Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="result-card negative-card">
            <h1>😞 NEGATIVE REVIEW</h1>
            <p>The model predicts a negative movie review.</p>
            <h1>{result["confidence"]:.2f}%</h1>
            <p>Prediction Confidence</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-heading">📈 Probability Distribution</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write("🟢 Positive Probability")

        st.progress(
            int(result["positive_probability"])
        )

        st.write(
            f'{result["positive_probability"]:.2f}%'
        )

    with col2:

        st.write("🔴 Negative Probability")

        st.progress(
            int(result["negative_probability"])
        )

        st.write(
            f'{result["negative_probability"]:.2f}%'
        )


# =====================================================
# 16. PREDICTION HISTORY
# =====================================================

st.divider()

st.markdown(
    '<div class="section-heading">📚 Prediction History</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        csv = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download History CSV",
            data=csv,
            file_name="movie_sentiment_history.csv",
            mime="text/csv",
            use_container_width=True
        )

    with col2:

        if st.button(
            "🗑️ Clear History",
            use_container_width=True
        ):

            st.session_state.history = []

            st.rerun()

else:

    st.info(
        "Your prediction history will appear here."
    )


# =====================================================
# 17. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    <hr>
    <h3>🎬 IMDb Movie Review Sentiment Analyzer</h3>
    <p>
        Built with Python | NLP | Word2Vec |
        ANN | TensorFlow | Streamlit
    </p>
    <p>Machine Learning Project</p>
</div>
""", unsafe_allow_html=True)

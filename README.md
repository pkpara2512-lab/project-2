import re
import joblib
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/sentiment_model.pkl"
)


# ==========================================
# LOAD TF-IDF VECTORIZER
# ==========================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


# ==========================================
# NLTK
# ==========================================

stop_words = set(
    stopwords.words("english")
)


lemmatizer = WordNetLemmatizer()


# ==========================================
# CLEAN TEXT
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    return text


# ==========================================
# PREPROCESS TEXT
# ==========================================

def preprocess_text(text):

    text = clean_text(
        text
    )

    tokens = word_tokenize(
        text
    )

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(
        tokens
    )


# ==========================================
# STREAMLIT UI
# ==========================================

st.title(
    "ShopKart Sentiment Analyzer"
)


st.write(
    "Enter a customer support message "
    "and predict its sentiment."
)


message = st.text_area(
    "Customer Message"
)


# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "Predict Sentiment"
):

    if message.strip() == "":

        st.warning(
            "Please enter a message."
        )

    else:

        processed = (
            preprocess_text(
                message
            )
        )


        vector = (
            vectorizer.transform(
                [processed]
            )
        )


        prediction = (
            model.predict(
                vector
            )[0]
        )


        st.success(
            f"Predicted Sentiment: {prediction}"
        )
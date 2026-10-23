# ==========================================
# SHOPKART SENTIMENT ANALYSIS PROJECT
# ==========================================


# ==========================================
# STEP 1: IMPORT LIBRARIES
# ==========================================
import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# STEP 2: LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/shopkart_support_tickets_synthetic.csv"
)

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())


# ==========================================
# STEP 3: TEXT CLEANING
# ==========================================

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    return text


df["cleaned_text"] = (
    df["message"]
    .apply(clean_text)
)


print("\nCleaned Text:")
print(
    df[
        ["message", "cleaned_text"]
    ].head()
)


# ==========================================
# STEP 4: TOKENIZATION
# ==========================================

df["tokens"] = (
    df["cleaned_text"]
    .apply(word_tokenize)
)


print("\nTokenized Text:")
print(
    df[
        ["cleaned_text", "tokens"]
    ].head()
)


# ==========================================
# STEP 5: STOP WORD REMOVAL
# ==========================================

stop_words = set(
    stopwords.words("english")
)


def remove_stopwords(tokens):

    return [
        word
        for word in tokens
        if word not in stop_words
    ]


df["no_stopwords"] = (
    df["tokens"]
    .apply(remove_stopwords)
)


print("\nAfter Stop Word Removal:")
print(
    df[
        ["tokens", "no_stopwords"]
    ].head()
)


# ==========================================
# STEP 6: STEMMING
# ==========================================

stemmer = PorterStemmer()


def stem_words(words):

    return [
        stemmer.stem(word)
        for word in words
    ]


df["stemmed"] = (
    df["no_stopwords"]
    .apply(stem_words)
)


print("\nAfter Stemming:")
print(
    df[
        ["no_stopwords", "stemmed"]
    ].head()
)


# ==========================================
# STEP 7: LEMMATIZATION
# ==========================================

lemmatizer = WordNetLemmatizer()


def lemmatize_words(words):

    return [
        lemmatizer.lemmatize(word)
        for word in words
    ]


df["lemmatized"] = (
    df["no_stopwords"]
    .apply(lemmatize_words)
)


print("\nAfter Lemmatization:")
print(
    df[
        ["no_stopwords", "lemmatized"]
    ].head()
)


# ==========================================
# STEP 8: FINAL PROCESSED TEXT
# ==========================================

df["processed_text"] = (
    df["lemmatized"]
    .apply(
        lambda words:
        " ".join(words)
    )
)


print("\nProcessed Text:")
print(
    df[
        [
            "message",
            "processed_text",
            "sentiment"
        ]
    ].head()
)


# ==========================================
# STEP 9: X AND Y
# ==========================================

X = df["processed_text"]

y = df["sentiment"]


# ==========================================
# STEP 10: TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining Rows:")
print(len(X_train))

print("\nTesting Rows:")
print(len(X_test))


# ==========================================
# STEP 11: TF-IDF
# ==========================================

vectorizer = TfidfVectorizer()


X_train_tfidf = (
    vectorizer.fit_transform(
        X_train
    )
)


X_test_tfidf = (
    vectorizer.transform(
        X_test
    )
)


print("\nTF-IDF Training Shape:")
print(X_train_tfidf.shape)

print("\nTF-IDF Testing Shape:")
print(X_test_tfidf.shape)


# ==========================================
# STEP 12: MODEL TRAINING
# ==========================================

model = LogisticRegression(
    max_iter=1000
)


model.fit(
    X_train_tfidf,
    y_train
)


print("\nModel Training Completed!")


# ==========================================
# STEP 13: PREDICTION
# ==========================================

y_pred = model.predict(
    X_test_tfidf
)


# ==========================================
# STEP 14: EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nInitial Accuracy:")
print(accuracy)


print("\nInitial Classification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# STEP 15: CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure(
    figsize=(7, 5)
)


sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)


plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title(
    "Sentiment Confusion Matrix"
)

plt.savefig("confusion_matrix.png")
plt.close()

print("Confusion matrix saved. Continuing training...")


# ==========================================
# STEP 16: HYPERPARAMETER TUNING
# ==========================================

# ==========================================
# STEP 16: HYPERPARAMETER TUNING
# ==========================================

parameters = {
    "C": [0.1, 1, 10],
    "solver": ["lbfgs"]
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=2000),
    parameters,
    cv=3,
    scoring="accuracy",
    n_jobs=-1
)

print("\nStarting hyperparameter tuning...", flush=True)

grid_search.fit(
    X_train_tfidf,
    y_train
)

print("\nHyperparameter tuning completed!", flush=True)

print("\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_


# ==========================================
# STEP 17: FINAL PREDICTION
# ==========================================

y_pred_tuned = (
    best_model.predict(
        X_test_tfidf
    )
)


# ==========================================
# STEP 18: FINAL EVALUATION
# ==========================================

final_accuracy = (
    accuracy_score(
        y_test,
        y_pred_tuned
    )
)


print("\nFinal Accuracy:")

print(
    final_accuracy
)


print(
    "\nFinal Classification Report:"
)


print(
    classification_report(
        y_test,
        y_pred_tuned
    )
)


# ==========================================
# STEP 19: ERROR ANALYSIS
# ==========================================

results = pd.DataFrame({

    "message": X_test,

    "actual": y_test,

    "predicted": y_pred_tuned

})


errors = results[
    results["actual"]
    != results["predicted"]
]


print("\nNumber of Errors:")

print(
    len(errors)
)


print("\nSample Errors:")

print(
    errors.head(20)
)


# ==========================================
# STEP 20: EXPLAINABILITY
# ==========================================

feature_names = (
    vectorizer
    .get_feature_names_out()
)


coefficients = (
    best_model.coef_
)


print(
    "\nImportant Words:"
)


for i, class_name in enumerate(
    best_model.classes_
):

    print(
        "\nClass:",
        class_name
    )

    top_indices = (
        coefficients[i]
        .argsort()[-10:]
    )

    for index in reversed(
        top_indices
    ):

        print(
            feature_names[index],
            coefficients[i][index]
        )


# ==========================================
# STEP 21: SENTIMENT PREDICTION FUNCTION
# ==========================================

def predict_sentiment(text):

    text = clean_text(text)

    tokens = word_tokenize(
        text
    )

    tokens = remove_stopwords(
        tokens
    )

    tokens = lemmatize_words(
        tokens
    )

    processed = " ".join(
        tokens
    )

    vector = vectorizer.transform(
        [processed]
    )

    prediction = best_model.predict(
        vector
    )

    return prediction[0]


# Test prediction

print("\nCustom Predictions:")

print(
    predict_sentiment(
        "I am very happy with the service"
    )
)

print(
    predict_sentiment(
        "I am very angry and disappointed"
    )
)

print(
    predict_sentiment(
        "When will my order arrive?"
    )
)


# ==========================================
# STEP 22: SAVE MODEL
# ==========================================
# STEP 22: SAVE MODEL

# ==========================================
# STEP 22: SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/sentiment_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)

print("\nModel and Vectorizer Saved Successfully!")

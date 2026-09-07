import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.data import find


# ============================================================
# NLTK RESOURCE SETUP
# ============================================================

def ensure_nltk_resource(resource_path, download_name):
    """
    Check whether an NLTK resource exists.
    Download it only when it is missing.
    """
    try:
        find(resource_path)
    except LookupError:
        nltk.download(
            download_name,
            quiet=True
        )


ensure_nltk_resource(
    "corpora/stopwords",
    "stopwords"
)

ensure_nltk_resource(
    "corpora/wordnet",
    "wordnet"
)

ensure_nltk_resource(
    "corpora/omw-1.4",
    "omw-1.4"
)


# ============================================================
# NLP OBJECTS
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


# ============================================================
# TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Clean and normalize storyline text.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Keep English letters and spaces
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

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)
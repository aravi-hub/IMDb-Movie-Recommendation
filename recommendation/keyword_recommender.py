import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.text_preprocessing import clean_text


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed_movies_2024.csv"

TOP_N = 5

# Hybrid score weights
COSINE_WEIGHT = 0.70
KEYWORD_WEIGHT = 0.20
PHRASE_WEIGHT = 0.10

# Number of important keywords / phrases
TOP_KEYWORDS = 7
TOP_PHRASES = 5


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not os.path.exists(DATA_FILE):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required_columns = [
        "Movie Name",
        "Storyline",
        "Cleaned Storyline"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df.dropna(
        subset=[
            "Movie Name",
            "Storyline",
            "Cleaned Storyline"
        ]
    )

    df = df.reset_index(drop=True)

    return df


# ============================================================
# TF-IDF MODEL
# ============================================================

def create_tfidf_model(df):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["Cleaned Storyline"]
    )

    return vectorizer, tfidf_matrix


# ============================================================
# EXTRACT IMPORTANT KEYWORDS
# ============================================================

def extract_keywords(text, vectorizer, top_n=TOP_KEYWORDS):

    cleaned = clean_text(text)

    words = cleaned.split()

    if not words:
        return []

    # TF-IDF scores for user storyline
    vector = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()

    scores = vector.toarray()[0]

    ranked_indices = scores.argsort()[::-1]

    keywords = []

    for idx in ranked_indices:

        term = feature_names[idx]

        # Keep single-word terms only
        if " " in term:
            continue

        # Remove very short words
        if len(term) < 4:
            continue

        # Make sure the word exists in the cleaned input
        if term not in words:
            continue

        if term not in keywords:

            keywords.append(term)

        if len(keywords) == top_n:
            break

    return keywords


# ============================================================
# EXTRACT IMPORTANT PHRASES
# ============================================================

def extract_phrases(text, vectorizer, top_n=TOP_PHRASES):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    feature_names = vectorizer.get_feature_names_out()

    scores = vector.toarray()[0]

    ranked_indices = scores.argsort()[::-1]

    phrases = []

    for idx in ranked_indices:

        term = feature_names[idx]

        # Only bigrams
        if " " not in term:
            continue

        parts = term.split()

        if len(parts) != 2:
            continue

        # Ignore very short tokens
        if any(len(word) < 4 for word in parts):
            continue

        # Verify both words belong to cleaned input
        if all(word in cleaned.split() for word in parts):

            if term not in phrases:
                phrases.append(term)

        if len(phrases) == top_n:
            break

    return phrases


# ============================================================
# KEYWORD MATCHING
# ============================================================

def calculate_keyword_score(
    keywords,
    movie_text
):

    movie_cleaned = clean_text(movie_text)

    movie_words = set(
        movie_cleaned.split()
    )

    if not keywords:
        return 0.0, []

    matched_keywords = []

    for keyword in keywords:

        if keyword in movie_words:

            matched_keywords.append(
                keyword
            )

    score = (
        len(matched_keywords)
        /
        len(keywords)
    )

    return score, matched_keywords


# ============================================================
# PHRASE MATCHING
# ============================================================

def calculate_phrase_score(
    phrases,
    movie_text
):

    movie_cleaned = clean_text(movie_text)

    if not phrases:
        return 0.0, []

    matched_phrases = []

    for phrase in phrases:

        if phrase in movie_cleaned:

            matched_phrases.append(
                phrase
            )

    score = (
        len(matched_phrases)
        /
        len(phrases)
    )

    return score, matched_phrases


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    cosine_score,
    keyword_score,
    phrase_score,
    matched_keywords,
    matched_phrases
):

    reasons = []

    if cosine_score >= 0.25:

        reasons.append(
            "strong storyline similarity"
        )

    elif cosine_score >= 0.12:

        reasons.append(
            "moderate storyline similarity"
        )

    else:

        reasons.append(
            "limited direct storyline similarity"
        )

    if matched_phrases:

        reasons.append(
            "matching phrases: "
            + ", ".join(matched_phrases)
        )

    if matched_keywords:

        reasons.append(
            "matching keywords: "
            + ", ".join(matched_keywords)
        )

    return (
        "Recommended because of "
        + "; ".join(reasons)
        + "."
    )


# ============================================================
# HYBRID RECOMMENDATION
# ============================================================

def recommend_movies(
    user_storyline,
    df,
    vectorizer,
    tfidf_matrix,
    top_n=TOP_N
):

    # --------------------------------------------------------
    # Clean user input
    # --------------------------------------------------------

    cleaned_input = clean_text(
        user_storyline
    )

    # --------------------------------------------------------
    # User TF-IDF vector
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [cleaned_input]
    )

    # --------------------------------------------------------
    # Cosine similarity
    # --------------------------------------------------------

    cosine_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # --------------------------------------------------------
    # Important keywords
    # --------------------------------------------------------

    keywords = extract_keywords(
        user_storyline,
        vectorizer
    )

    # --------------------------------------------------------
    # Important phrases
    # --------------------------------------------------------

    phrases = extract_phrases(
        user_storyline,
        vectorizer
    )

    recommendations = []

    # --------------------------------------------------------
    # Calculate hybrid score for each movie
    # --------------------------------------------------------

    for index, row in df.iterrows():

        movie_storyline = row["Storyline"]

        keyword_score, matched_keywords = (
            calculate_keyword_score(
                keywords,
                movie_storyline
            )
        )

        phrase_score, matched_phrases = (
            calculate_phrase_score(
                phrases,
                movie_storyline
            )
        )

        cosine_score = cosine_scores[index]

        final_score = (
            COSINE_WEIGHT * cosine_score
            +
            KEYWORD_WEIGHT * keyword_score
            +
            PHRASE_WEIGHT * phrase_score
        )

        explanation = generate_explanation(
            cosine_score,
            keyword_score,
            phrase_score,
            matched_keywords,
            matched_phrases
        )

        recommendations.append(
            {
                "Movie Name": row["Movie Name"],
                "Storyline": movie_storyline,
                "Cosine Similarity": cosine_score,
                "Keyword Score": keyword_score,
                "Phrase Score": phrase_score,
                "Final Score": final_score,
                "Matched Keywords": matched_keywords,
                "Matched Phrases": matched_phrases,
                "Why Recommended": explanation
            }
        )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        recommendations
    )

    # --------------------------------------------------------
    # Remove movies with zero relevance
    # --------------------------------------------------------

    result = result[
        result["Final Score"] > 0
    ]

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    result = result.sort_values(
        by="Final Score",
        ascending=False
    )

    result = result.head(top_n)

    return (
        result,
        keywords,
        phrases
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def main():

    print("=" * 60)
    print("IMDb HYBRID MOVIE RECOMMENDER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print("\nDataset shape:")
    print(df.shape)

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    vectorizer, tfidf_matrix = (
        create_tfidf_model(df)
    )

    print("\nTF-IDF configuration:")
    print("ngram_range = (1, 2)")
    print("max_features = 10000")

    print("\nTF-IDF matrix shape:")
    print(tfidf_matrix.shape)

    # --------------------------------------------------------
    # User Input
    # --------------------------------------------------------

    user_storyline = input(
        "\nEnter movie storyline:\n"
    ).strip()

    if not user_storyline:

        print(
            "\nPlease enter a valid storyline."
        )

        return

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    results, keywords, phrases = (
        recommend_movies(
            user_storyline,
            df,
            vectorizer,
            tfidf_matrix
        )
    )

    # --------------------------------------------------------
    # Display NLP features
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("IMPORTANT KEYWORDS")

    print("=" * 60)

    print(keywords)

    print("\nIMPORTANT PHRASES")

    print("=" * 60)

    print(phrases)

    # --------------------------------------------------------
    # Display Recommendations
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP 5 HYBRID RECOMMENDATIONS")
    print("=" * 60)

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1
    ):

        print(
            f"\n{rank}. {row['Movie Name']}"
        )

        print(
            f"Cosine Similarity: "
            f"{row['Cosine Similarity']:.2%}"
        )

        print(
            f"Keyword Score: "
            f"{row['Keyword Score']:.2%}"
        )

        print(
            f"Phrase Score: "
            f"{row['Phrase Score']:.2%}"
        )

        print(
            f"Final Score: "
            f"{row['Final Score']:.2%}"
        )

        if row["Matched Keywords"]:

            print(
                "Matched Keywords: "
                + ", ".join(
                    row["Matched Keywords"]
                )
            )

        else:

            print(
                "Matched Keywords: None"
            )

        if row["Matched Phrases"]:

            print(
                "Matched Phrases: "
                + ", ".join(
                    row["Matched Phrases"]
                )
            )

        else:

            print(
                "Matched Phrases: None"
            )

        print(
            "Why recommended: "
            + row["Why Recommended"]
        )

        print(
            "Storyline: "
            + row["Storyline"]
        )

        print("-" * 60)
        


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
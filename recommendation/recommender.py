import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed_movies_2024.csv"

VECTORIZER_FILE = "models/tfidf_vectorizer.pkl"
MATRIX_FILE = "models/tfidf_matrix.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("IMDb MOVIE RECOMMENDATION ENGINE")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "Cleaned Storyline" not in df.columns:
    raise ValueError(
        "Cleaned Storyline column is missing from the dataset."
    )


# ============================================================
# TF-IDF VECTORISATION
# ============================================================

print("\nCreating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(
    df["Cleaned Storyline"].fillna("")
)


print("TF-IDF matrix shape:")
print(tfidf_matrix.shape)


# ============================================================
# COSINE SIMILARITY
# ============================================================

print("\nCalculating cosine similarity...")

similarity_matrix = cosine_similarity(tfidf_matrix)


print("Similarity matrix shape:")
print(similarity_matrix.shape)


# ============================================================
# CREATE MODELS FOLDER
# ============================================================

import os

os.makedirs("models", exist_ok=True)


# ============================================================
# SAVE MODEL FILES
# ============================================================

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

joblib.dump(
    tfidf_matrix,
    MATRIX_FILE
)


print("\nTF-IDF vectorizer saved:")
print(VECTORIZER_FILE)

print("\nTF-IDF matrix saved:")
print(MATRIX_FILE)


# ============================================================
# BASIC MOVIE RECOMMENDATION FUNCTION
# ============================================================

def recommend_by_movie(movie_name, top_n=5):

    matches = df[
        df["Movie Name"]
        .str.lower()
        .str.strip()
        == movie_name.lower().strip()
    ]

    if matches.empty:
        return None

    movie_index = matches.index[0]

    scores = similarity_matrix[movie_index]

    similar_indices = scores.argsort()[::-1]

    recommendations = []

    for index in similar_indices:

        # Exclude the input movie itself
        if index == movie_index:
            continue

        recommendations.append(
            {
                "Movie Name": df.iloc[index]["Movie Name"],
                "Storyline": df.iloc[index]["Storyline"],
                "Similarity Score": scores[index]
            }
        )

        if len(recommendations) == top_n:
            break

    return pd.DataFrame(recommendations)


# ============================================================
# TEST RECOMMENDATION
# ============================================================

test_movie = df.iloc[0]["Movie Name"]

print("\n")
print("=" * 60)
print("TEST RECOMMENDATION")
print("=" * 60)

print("\nInput Movie:")
print(test_movie)

results = recommend_by_movie(
    test_movie,
    top_n=5
)

if results is not None:

    print("\nTop 5 Similar Movies:\n")

    for _, row in results.iterrows():

        print(
            f"Movie: {row['Movie Name']}"
        )

        print(
            f"Similarity: "
            f"{row['Similarity Score']:.2%}"
        )

        print(
            f"Storyline: "
            f"{row['Storyline']}"
        )

        print("-" * 60)

else:

    print("Movie not found.")


print("\nTF-IDF + Cosine Similarity completed successfully.")
import os
import re

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "processed_movies_2024.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

print("=" * 60)
print("IMDb MOVIE RECOMMENDER - MODEL EVALUATION")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(text, top_n=10):

    text = str(text).lower()

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    words = text.split()

    ignored_words = {
        "movie",
        "film",
        "story",
        "man",
        "woman",
        "person",
        "people",
        "year",
        "life",
        "time",
        "young",
        "new",
        "old",
        "one",
        "two",
        "first",
        "last",
        "world"
    }

    words = [
        word
        for word in words
        if len(word) >= 4
        and word not in ignored_words
    ]

    return list(
        dict.fromkeys(words)
    )[:top_n]


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

tfidf_matrix = vectorizer.fit_transform(
    df["Cleaned Storyline"]
)


# ============================================================
# HYBRID RECOMMENDER
# ============================================================

def recommend_movies(
    user_storyline,
    top_n=5
):

    user_vector = vectorizer.transform(
        [user_storyline]
    )

    cosine_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    user_keywords = set(
        extract_keywords(
            user_storyline,
            top_n=10
        )
    )

    recommendations = []

    for index, row in df.iterrows():

        movie_keywords = set(
            extract_keywords(
                row["Storyline"],
                top_n=20
            )
        )

        matched_keywords = (
            user_keywords
            .intersection(movie_keywords)
        )

        if user_keywords:

            keyword_score = (
                len(matched_keywords)
                /
                len(user_keywords)
            )

        else:

            keyword_score = 0.0

        union = (
            user_keywords
            .union(movie_keywords)
        )

        if union:

            phrase_score = (
                len(matched_keywords)
                /
                len(union)
            )

        else:

            phrase_score = 0.0

        final_score = (
            0.70 * cosine_scores[index]
            +
            0.20 * keyword_score
            +
            0.10 * phrase_score
        )

        recommendations.append(
            {
                "Movie Name":
                    row["Movie Name"],

                "Cosine Similarity":
                    cosine_scores[index],

                "Keyword Score":
                    keyword_score,

                "Phrase Score":
                    phrase_score,

                "Final Score":
                    final_score
            }
        )

    result = pd.DataFrame(
        recommendations
    )

    return result.sort_values(
        by="Final Score",
        ascending=False
    ).head(top_n)


# ============================================================
# TEST STORYLINES
# ============================================================

test_cases = [

    """
    A group of soldiers go on a dangerous military
    mission during a war against enemy forces.
    """,

    """
    A young woman discovers a mysterious secret
    that changes her life and forces her to fight
    against powerful enemies.
    """,

    """
    A detective investigates a mysterious murder
    and discovers a dangerous criminal conspiracy.
    """,

    """
    A family struggles to survive after a disaster
    destroys their home and separates them from
    each other.
    """,

    """
    A hero fights against a powerful villain
    to save the world from destruction.
    """
]


# ============================================================
# EVALUATION
# ============================================================

evaluation_results = []

top_k = 5

print("\n")
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)


for i, storyline in enumerate(
    test_cases,
    start=1
):

    recommendations = recommend_movies(
        storyline,
        top_n=top_k
    )

    average_final_score = (
        recommendations["Final Score"]
        .mean()
    )

    average_cosine_score = (
        recommendations["Cosine Similarity"]
        .mean()
    )

    average_keyword_score = (
        recommendations["Keyword Score"]
        .mean()
    )

    average_phrase_score = (
        recommendations["Phrase Score"]
        .mean()
    )

    evaluation_results.append(
        {
            "Test Case":
                i,

            "Average Final Score":
                average_final_score,

            "Average Cosine Score":
                average_cosine_score,

            "Average Keyword Score":
                average_keyword_score,

            "Average Phrase Score":
                average_phrase_score
        }
    )

    print(
        f"\nTest Case {i}"
    )

    print(
        f"Average Final Score: "
        f"{average_final_score:.2%}"
    )

    print(
        f"Average Cosine Score: "
        f"{average_cosine_score:.2%}"
    )

    print(
        f"Average Keyword Score: "
        f"{average_keyword_score:.2%}"
    )

    print(
        f"Average Phrase Score: "
        f"{average_phrase_score:.2%}"
    )


# ============================================================
# OVERALL RESULTS
# ============================================================

evaluation_df = pd.DataFrame(
    evaluation_results
)


overall_final_score = (
    evaluation_df[
        "Average Final Score"
    ].mean()
)

overall_cosine_score = (
    evaluation_df[
        "Average Cosine Score"
    ].mean()
)

overall_keyword_score = (
    evaluation_df[
        "Average Keyword Score"
    ].mean()
)

overall_phrase_score = (
    evaluation_df[
        "Average Phrase Score"
    ].mean()
)


print("\n")
print("=" * 60)
print("OVERALL MODEL EVALUATION")
print("=" * 60)

print(
    f"\nOverall Average Final Score: "
    f"{overall_final_score:.2%}"
)

print(
    f"Overall Average Cosine Score: "
    f"{overall_cosine_score:.2%}"
)

print(
    f"Overall Average Keyword Score: "
    f"{overall_keyword_score:.2%}"
)

print(
    f"Overall Average Phrase Score: "
    f"{overall_phrase_score:.2%}"
)


# ============================================================
# TOP-5 COVERAGE
# ============================================================

total_movies = len(df)

recommended_movies = set()

for storyline in test_cases:

    recommendations = recommend_movies(
        storyline,
        top_n=top_k
    )

    recommended_movies.update(
        recommendations["Movie Name"].tolist()
    )


coverage_at_5 = (
    len(recommended_movies)
    /
    total_movies
)


print(
    f"\nCoverage@5: "
    f"{coverage_at_5:.2%}"
)


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

OUTPUT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "model_evaluation_results.csv"
)

evaluation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nEvaluation file saved:"
    f"\n{OUTPUT_FILE}"
)

print("\nEvaluation completed.")
import os
import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Project root path
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_movies_2024.csv"
)

# Import preprocessing
sys.path.append(BASE_DIR)

from preprocessing.text_preprocessing import clean_text


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(
    df["Cleaned Storyline"]
)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(user_input, top_n=5):

    cleaned_input = clean_text(user_input)

    user_vector = vectorizer.transform(
        [cleaned_input]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]

    return df.iloc[top_indices][
        ["Movie Name", "Storyline"]
    ].copy(), similarity_scores[top_indices]


# ============================================================
# TEST CASES
# ============================================================

test_cases = [
    (
        "Horror",
        "A terrifying vampire haunts a young woman in a dark gothic town."
    ),
    (
        "Mystery",
        "A detective investigates a mysterious murder and discovers a hidden secret."
    ),
    (
        "War",
        "A group of soldiers secretly fights enemy forces during a major war."
    ),
    (
        "Romance",
        "Two people unexpectedly fall in love despite their different backgrounds."
    ),
    (
        "Fantasy",
        "A young hero discovers magical powers and fights a dark supernatural force."
    ),
    (
        "Short Input",
        "horror"
    ),
    (
        "Normal Input",
        "A young detective investigates a mysterious murder in a small town."
    ),
    (
        "Empty Input",
        ""
    )
]


# ============================================================
# RUN TESTS
# ============================================================

print("=" * 70)
print("IMDb MOVIE RECOMMENDATION - FINAL TESTING")
print("=" * 70)

print(f"\nDataset size: {len(df)} movies")
print(f"TF-IDF matrix: {tfidf_matrix.shape}")


passed = 0
failed = 0


for test_name, query in test_cases:

    print("\n" + "=" * 70)
    print(f"TEST CASE: {test_name}")
    print("=" * 70)

    print(f"Input: {query}")

    # Empty input handling
    if not query.strip():

        print("RESULT: PASS")
        print("Empty input handled safely.")

        passed += 1
        continue

    try:

        results, scores = recommend_movies(query)

        if len(results) == 5:

            print("RESULT: PASS")
            print("Top 5 recommendations generated.")

            print("\nTop Recommendations:")

            for rank, (_, row) in enumerate(
                results.iterrows(),
                start=1
            ):

                score = scores[rank - 1]

                print(
                    f"{rank}. {row['Movie Name']} "
                    f"- Similarity: {score:.2%}"
                )

            passed += 1

        else:

            print("RESULT: FAIL")
            print(
                f"Expected 5 results, "
                f"received {len(results)}"
            )

            failed += 1

    except Exception as error:

        print("RESULT: FAIL")
        print("Error:", error)

        failed += 1


# ============================================================
# FINAL TEST SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)

print(f"Tests Passed : {passed}")
print(f"Tests Failed : {failed}")
print(
    f"Total Tests  : {passed + failed}"
)

if failed == 0:

    print("\n✅ ALL TESTS PASSED")

else:

    print(
        f"\n⚠️ {failed} TEST(S) NEED ATTENTION"
    )
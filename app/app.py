import os
import sys

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed_movies_2024.csv"
)

sys.path.append(BASE_DIR)

from preprocessing.text_preprocessing import clean_text


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="IMDb Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .movie-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .movie-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .score-box {
        padding: 10px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.20);
    }

    .small-label {
        font-size: 13px;
    }

    .keyword-tag {
        display: inline-block;
        padding: 4px 9px;
        margin: 3px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    df = pd.read_csv(DATA_FILE)

    df["Movie Name"] = (
        df["Movie Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Storyline"] = (
        df["Storyline"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["Cleaned Storyline"] = (
        df["Cleaned Storyline"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


df = load_dataset()


# ============================================================
# TF-IDF MODEL
# ============================================================

@st.cache_resource
def create_tfidf_model(data):

    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        data["Cleaned Storyline"]
    )

    return vectorizer, tfidf_matrix


vectorizer, tfidf_matrix = create_tfidf_model(df)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def recommend_movies(
    user_input,
    top_n=5
):

    cleaned_input = clean_text(
        user_input
    )

    # --------------------------------------------------------
    # TF-IDF similarity
    # --------------------------------------------------------

    user_vector = vectorizer.transform(
        [cleaned_input]
    )

    cosine_scores = cosine_similarity(
        user_vector,
        tfidf_matrix
    ).flatten()

    # --------------------------------------------------------
    # User keywords
    # --------------------------------------------------------

    user_words = set(
        cleaned_input.split()
    )

    user_keywords = {
        word
        for word in user_words
        if len(word) >= 4
    }

    # --------------------------------------------------------
    # Important words from original order
    # --------------------------------------------------------

    input_words = cleaned_input.split()

    # --------------------------------------------------------
    # Build results
    # --------------------------------------------------------

    results = []

    for index, row in df.iterrows():

        movie_text = row[
            "Cleaned Storyline"
        ]

        movie_words = set(
            movie_text.split()
        )

        # ----------------------------------------------------
        # Keyword matching
        # ----------------------------------------------------

        matched_keywords = sorted(
            user_keywords.intersection(
                movie_words
            )
        )

        if user_keywords:

            keyword_score = (
                len(matched_keywords)
                /
                len(user_keywords)
            )

        else:

            keyword_score = 0.0

        # ----------------------------------------------------
        # Phrase matching
        # ----------------------------------------------------

        phrases = []

        for i in range(
            len(input_words) - 1
        ):

            phrase = (
                input_words[i]
                + " "
                + input_words[i + 1]
            )

            phrases.append(
                phrase
            )

        matched_phrases = []

        for phrase in phrases:

            if phrase in movie_text:

                matched_phrases.append(
                    phrase
                )

        if phrases:

            phrase_score = (
                len(matched_phrases)
                /
                len(phrases)
            )

        else:

            phrase_score = 0.0

        # ----------------------------------------------------
        # Hybrid score
        # ----------------------------------------------------

        final_score = (
            0.70 * cosine_scores[index]
            +
            0.20 * keyword_score
            +
            0.10 * phrase_score
        )

        # ----------------------------------------------------
        # Recommendation reason
        # ----------------------------------------------------

        if cosine_scores[index] >= 0.25:

            reason = (
                "The storyline has strong textual similarity "
                "with the input."
            )

        elif cosine_scores[index] >= 0.10:

            reason = (
                "The storyline has moderate textual similarity "
                "with the input."
            )

        else:

            reason = (
                "The movie contains related concepts "
                "from the input storyline."
            )

        if matched_keywords:

            reason += (
                " Matching keywords: "
                + ", ".join(
                    matched_keywords
                )
                + "."
            )

        if matched_phrases:

            reason += (
                " Matching phrases: "
                + ", ".join(
                    matched_phrases
                )
                + "."
            )

        results.append(
            {
                "movie": row["Movie Name"],
                "storyline": row["Storyline"],
                "cosine": cosine_scores[index],
                "keyword": keyword_score,
                "phrase": phrase_score,
                "final": final_score,
                "matched_keywords":
                    matched_keywords,
                "matched_phrases":
                    matched_phrases,
                "reason": reason
            }
        )

    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["final"],
        reverse=True
    )

    return results[:top_n]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🎬 IMDb Movie Recommendation System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Storyline-based recommendation using NLP, TF-IDF, '
    'Cosine Similarity and Hybrid Scoring'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("System Details")

    st.write(
        f"**Dataset:** IMDb Movies 2024"
    )

    st.write(
        f"**Movies:** {len(df)}"
    )

    st.write(
        f"**TF-IDF Features:** "
        f"{tfidf_matrix.shape[1]}"
    )

    st.write(
        "**Recommendation:** Top 5"
    )

    st.divider()

    st.subheader("Hybrid Score")

    st.write("70% — Cosine Similarity")
    st.write("20% — Keyword Score")
    st.write("10% — Phrase Score")

    st.divider()

    st.subheader("Techniques")

    st.write("• Python")
    st.write("• Pandas")
    st.write("• NLTK")
    st.write("• TF-IDF")
    st.write("• Cosine Similarity")
    st.write("• Streamlit")


# ============================================================
# DATASET METRICS
# ============================================================

m1, m2, m3 = st.columns(3)

with m1:

    st.metric(
        "Movies Available",
        len(df)
    )

with m2:

    st.metric(
        "TF-IDF Features",
        tfidf_matrix.shape[1]
    )

with m3:

    st.metric(
        "Results",
        5
    )


st.write("")


# ============================================================
# USER INPUT
# ============================================================

st.subheader(
    "📝 Enter Movie Storyline"
)

st.caption(
    "Enter at least a few meaningful words describing "
    "the movie story."
)

user_storyline = st.text_area(
    "Storyline",
    height=170,
    placeholder=(
        "Example: A young detective investigates a "
        "mysterious murder in a small town and discovers "
        "a hidden secret."
    ),
    label_visibility="collapsed"
)


# ============================================================
# EXAMPLE INPUTS
# ============================================================

st.write("**Try an example:**")

example1, example2, example3 = st.columns(3)

with example1:

    if st.button(
        "🕵️ Mystery Story",
        use_container_width=True
    ):

        st.session_state[
            "storyline"
        ] = (
            "A young detective investigates a "
            "mysterious murder in a small town "
            "and discovers a hidden secret."
        )

        st.rerun()


with example2:

    if st.button(
        "🧛 Horror Story",
        use_container_width=True
    ):

        st.session_state[
            "storyline"
        ] = (
            "A terrifying vampire haunts a young "
            "woman and spreads horror through "
            "a dark gothic town."
        )

        st.rerun()


with example3:

    if st.button(
        "🚀 Science Fiction",
        use_container_width=True
    ):

        st.session_state[
            "storyline"
        ] = (
            "A group of astronauts travel into "
            "deep space and encounter an unknown "
            "alien civilization."
        )

        st.rerun()


st.write("")


# ============================================================
# RECOMMEND BUTTON
# ============================================================

recommend_clicked = st.button(
    "🔍 Recommend Movies",
    type="primary",
    use_container_width=True
)


# ============================================================
# RESULT SECTION
# ============================================================

if recommend_clicked:

    if not user_storyline.strip():

        st.warning(
            "Please enter a movie storyline."
        )

    elif len(
        user_storyline.strip().split()
    ) < 2:

        st.warning(
            "Please enter at least two meaningful words."
        )

    else:

        with st.spinner(
            "Finding similar movies..."
        ):

            recommendations = recommend_movies(
                user_storyline,
                top_n=5
            )

        st.success(
            "Top 5 recommendations generated."
        )

        st.divider()

        # ====================================================
        # TOP RESULTS
        # ====================================================

        st.subheader(
            "🏆 Top 5 Recommendations"
        )

        # ----------------------------------------------------
        # Score chart
        # ----------------------------------------------------

        chart_data = pd.DataFrame(
            {
                "Movie": [
                    item["movie"]
                    for item in recommendations
                ],
                "Final Score": [
                    item["final"]
                    for item in recommendations
                ]
            }
        )

        st.write(
            "**Final Recommendation Score**"
        )

        st.bar_chart(
            chart_data.set_index("Movie")
        )

        st.write("")

        # ====================================================
        # MOVIE CARDS
        # ====================================================

        for rank, result in enumerate(
            recommendations,
            start=1
        ):

            st.markdown(
                f"""
                <div class="movie-card">

                <div class="movie-title">
                {rank}. 🎬 {result["movie"]}
                </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # SCORE METRICS
            # ------------------------------------------------

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Cosine Similarity",
                    f"{result['cosine']:.2%}"
                )

            with c2:

                st.metric(
                    "Keyword Score",
                    f"{result['keyword']:.2%}"
                )

            with c3:

                st.metric(
                    "Phrase Score",
                    f"{result['phrase']:.2%}"
                )

            with c4:

                st.metric(
                    "Final Score",
                    f"{result['final']:.2%}"
                )

            # ------------------------------------------------
            # MATCHED KEYWORDS
            # ------------------------------------------------

            if result[
                "matched_keywords"
            ]:

                st.write(
                    "**Matched Keywords**"
                )

                keyword_html = ""

                for keyword in result[
                    "matched_keywords"
                ]:

                    keyword_html += (
                        f'<span class="keyword-tag">'
                        f'{keyword}'
                        f'</span>'
                    )

                st.markdown(
                    keyword_html,
                    unsafe_allow_html=True
                )

            else:

                st.write(
                    "**Matched Keywords:** None"
                )

            # ------------------------------------------------
            # MATCHED PHRASES
            # ------------------------------------------------

            if result[
                "matched_phrases"
            ]:

                st.write(
                    "**Matched Phrases:** "
                    +
                    ", ".join(
                        result[
                            "matched_phrases"
                        ]
                    )
                )

            # ------------------------------------------------
            # WHY RECOMMENDED
            # ------------------------------------------------

            st.info(
                "💡 **Why recommended:** "
                + result["reason"]
            )

            # ------------------------------------------------
            # STORYLINE
            # ------------------------------------------------

            with st.expander(
                "View Storyline"
            ):

                st.write(
                    result["storyline"]
                )

            st.divider()
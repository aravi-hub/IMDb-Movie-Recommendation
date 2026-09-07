import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/processed_movies_2024.csv"
OUTPUT_DIR = "outputs"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("IMDb MOVIE DATA ANALYSIS & VISUALIZATION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)


# ============================================================
# BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nColumns:")
print(df.columns.tolist())

print("\nTotal movies:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# STORYLINE LENGTH
# ============================================================

df["Storyline Length"] = (
    df["Storyline"]
    .astype(str)
    .str.split()
    .str.len()
)

print("\n" + "=" * 60)
print("STORYLINE LENGTH ANALYSIS")
print("=" * 60)

print(
    "\nAverage storyline length:",
    round(df["Storyline Length"].mean(), 2),
    "words"
)

print(
    "Shortest storyline:",
    df["Storyline Length"].min(),
    "words"
)

print(
    "Longest storyline:",
    df["Storyline Length"].max(),
    "words"
)


# ============================================================
# TOP WORD FREQUENCY
# ============================================================

print("\n" + "=" * 60)
print("MOST FREQUENT WORDS")
print("=" * 60)

all_words = []

for text in df["Cleaned Storyline"].dropna():

    words = str(text).split()

    all_words.extend(words)


word_counts = Counter(all_words)


top_words = word_counts.most_common(10)

for word, count in top_words:

    print(f"{word}: {count}")


# ============================================================
# CHART 1 - STORYLINE LENGTH DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    df["Storyline Length"],
    bins=10
)

plt.title(
    "Distribution of Movie Storyline Length"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Number of Movies"
)

plt.tight_layout()

chart1 = os.path.join(
    OUTPUT_DIR,
    "storyline_length_distribution.png"
)

plt.savefig(
    chart1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nSaved:",
    chart1
)


# ============================================================
# CHART 2 - TOP WORD FREQUENCY
# ============================================================

words = [
    item[0]
    for item in top_words
]

counts = [
    item[1]
    for item in top_words
]

plt.figure(figsize=(10, 6))

plt.bar(
    words,
    counts
)

plt.title(
    "Top 10 Frequent Words in Movie Storylines"
)

plt.xlabel(
    "Words"
)

plt.ylabel(
    "Frequency"
)

plt.xticks(
    rotation=45
)

plt.tight_layout()

chart2 = os.path.join(
    OUTPUT_DIR,
    "top_10_words.png"
)

plt.savefig(
    chart2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    chart2
)


# ============================================================
# CHART 3 - TOP MOVIES BY STORYLINE LENGTH
# ============================================================

top_storylines = (
    df.sort_values(
        "Storyline Length",
        ascending=False
    )
    .head(10)
)

plt.figure(figsize=(12, 6))

plt.barh(
    top_storylines["Movie Name"],
    top_storylines["Storyline Length"]
)

plt.title(
    "Top 10 Movies by Storyline Length"
)

plt.xlabel(
    "Number of Words"
)

plt.ylabel(
    "Movie"
)

plt.gca().invert_yaxis()

plt.tight_layout()

chart3 = os.path.join(
    OUTPUT_DIR,
    "top_movies_storyline_length.png"
)

plt.savefig(
    chart3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved:",
    chart3
)


# ============================================================
# SAVE ANALYSIS SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "Metric": [
            "Total Movies",
            "Average Storyline Length",
            "Minimum Storyline Length",
            "Maximum Storyline Length",
            "Missing Values",
            "Duplicate Rows"
        ],
        "Value": [
            len(df),
            round(
                df["Storyline Length"].mean(),
                2
            ),
            df["Storyline Length"].min(),
            df["Storyline Length"].max(),
            int(
                df[
                    ["Movie Name", "Storyline"]
                ]
                .isnull()
                .sum()
                .sum()
            ),
            int(df.duplicated().sum())
        ]
    }
)

summary_file = os.path.join(
    OUTPUT_DIR,
    "analysis_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    "Saved:",
    summary_file
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("EDA AND VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nOutput files:")

print(
    "1.",
    chart1
)

print(
    "2.",
    chart2
)

print(
    "3.",
    chart3
)

print(
    "4.",
    summary_file
)
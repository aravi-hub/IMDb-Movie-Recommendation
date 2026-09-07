import pandas as pd

from text_preprocessing import clean_text


INPUT_FILE = "data/imdb_movies_2024.csv"
OUTPUT_FILE = "data/processed_movies_2024.csv"


def main():

    print("=" * 60)
    print("IMDb MOVIE DATA CLEANING")
    print("=" * 60)

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("\nOriginal dataset shape:")
    print(df.shape)

    # ---------------------------------------------------------
    # Select required columns
    # ---------------------------------------------------------

    df = df[
        [
            "Movie Name",
            "Storyline"
        ]
    ]

    # ---------------------------------------------------------
    # Check missing values
    # ---------------------------------------------------------

    print("\nMissing values before cleaning:")

    print(
        df.isnull().sum()
    )

    # ---------------------------------------------------------
    # Remove missing values
    # ---------------------------------------------------------

    df = df.dropna(
        subset=[
            "Movie Name",
            "Storyline"
        ]
    )

    # ---------------------------------------------------------
    # Convert to string
    # ---------------------------------------------------------

    df["Movie Name"] = (
        df["Movie Name"]
        .astype(str)
        .str.strip()
    )

    df["Storyline"] = (
        df["Storyline"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # Remove empty storylines
    # ---------------------------------------------------------

    df = df[
        df["Storyline"] != ""
    ]

    # ---------------------------------------------------------
    # Remove duplicate movies
    # ---------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=["Movie Name"]
    )

    after_duplicates = len(df)

    print(
        "\nDuplicate movies removed:",
        before_duplicates - after_duplicates
    )

    # ---------------------------------------------------------
    # NLP Cleaning
    # ---------------------------------------------------------

    print(
        "\nApplying NLP preprocessing..."
    )

    df["Cleaned Storyline"] = (
        df["Storyline"]
        .apply(clean_text)
    )

    # ---------------------------------------------------------
    # Remove empty cleaned text
    # ---------------------------------------------------------

    df = df[
        df["Cleaned Storyline"].str.strip() != ""
    ]

    # ---------------------------------------------------------
    # Reset index
    # ---------------------------------------------------------

    df = df.reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # Save processed dataset
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)

    print(
        "\nFinal dataset shape:",
        df.shape
    )

    print(
        "\nSaved file:",
        OUTPUT_FILE
    )

    print("\nFirst 5 processed records:")

    print(
        df.head().to_string(index=False)
    )


if __name__ == "__main__":
    main()
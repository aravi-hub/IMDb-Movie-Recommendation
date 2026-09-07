from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time
import os


# ============================================================
# CONFIGURATION
# ============================================================

IMDB_URL = (
    "https://www.imdb.com/search/title/"
    "?title_type=feature"
    "&release_date=2024-01-01,2024-12-31"
)

OUTPUT_FILE = "data/imdb_movies_2024.csv"


# ============================================================
# CREATE CHROME DRIVER
# ============================================================

def create_driver():
    options = Options()

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-notifications")

    driver = webdriver.Chrome(options=options)

    return driver


# ============================================================
# FIND TEXT USING MULTIPLE SELECTORS
# ============================================================

def find_text(element, selectors):
    """
    Try multiple CSS selectors and return the first
    non-empty text found.
    """

    for selector in selectors:
        try:
            child = element.find_element(By.CSS_SELECTOR, selector)

            text = child.text.strip()

            if text:
                return text

        except NoSuchElementException:
            continue
        except Exception:
            continue

    return ""


# ============================================================
# SCRAPE IMDb
# ============================================================

def scrape_imdb():

    driver = create_driver()

    movies = []

    try:

        print("Opening IMDb...")

        driver.get(IMDB_URL)

        time.sleep(5)

        print("IMDb page opened")
        print("Title:", driver.title)

        # ----------------------------------------------------
        # FIND MOVIE CARDS
        # ----------------------------------------------------

        card_selectors = [
            "li.ipc-metadata-list-summary-item",
            "li[data-testid='title-pc-list-item']",
            "div[data-testid='list-item']",
            "article"
        ]

        movie_cards = []

        for selector in card_selectors:

            try:

                elements = driver.find_elements(
                    By.CSS_SELECTOR,
                    selector
                )

                if elements:

                    movie_cards = elements

                    print(
                        f"Movie cards found using: {selector}"
                    )

                    print(
                        "Movie cards found:",
                        len(movie_cards)
                    )

                    break

            except Exception:
                continue

        if not movie_cards:

            print("ERROR: No movie cards found.")

            return

        # ----------------------------------------------------
        # EXTRACT MOVIE INFORMATION
        # ----------------------------------------------------

        for index, card in enumerate(movie_cards, start=1):

            try:

                # ------------------------------------------------
                # MOVIE NAME
                # ------------------------------------------------

                movie_name = find_text(
                    card,
                    [
                        "h3.ipc-title__text",
                        "h3",
                        "a.ipc-title-link-wrapper",
                        "[data-testid='title']"
                    ]
                )

                # IMDb sometimes gives number + title
                # Example:
                # 1. Movie Name
                if movie_name:

                    if ". " in movie_name:
                        parts = movie_name.split(". ", 1)

                        if parts[0].isdigit():
                            movie_name = parts[1].strip()

                # ------------------------------------------------
                # STORYLINE
                # ------------------------------------------------

                storyline = find_text(
                    card,
                    [
                        "div.ipc-html-content-inner-div",
                        "[data-testid='plot-xs-to-m']",
                        "[data-testid='plot-xl']",
                        "[data-testid='plot-l']",
                        "div.ipc-html-content",
                        "p"
                    ]
                )

                # ------------------------------------------------
                # FALLBACK: SEARCH ALL DIV TEXT
                # ------------------------------------------------

                if not storyline:

                    try:

                        divs = card.find_elements(
                            By.TAG_NAME,
                            "div"
                        )

                        possible_texts = []

                        for div in divs:

                            text = div.text.strip()

                            if (
                                text
                                and len(text) > 30
                                and text != movie_name
                            ):
                                possible_texts.append(text)

                        if possible_texts:
                            storyline = max(
                                possible_texts,
                                key=len
                            )

                    except Exception:
                        pass

                # ------------------------------------------------
                # SAVE ONLY IF BOTH VALUES EXIST
                # ------------------------------------------------

                if movie_name and storyline:

                    movies.append(
                        {
                            "Movie Name": movie_name,
                            "Storyline": storyline
                        }
                    )

                    print(
                        f"[{index}/{len(movie_cards)}] "
                        f"{movie_name}"
                    )

                else:

                    print(
                        f"[{index}/{len(movie_cards)}] "
                        "Movie data incomplete - skipped"
                    )

            except Exception as e:

                print(
                    f"[{index}/{len(movie_cards)}] "
                    f"Extraction error: {e}"
                )

        # ========================================================
        # CREATE DATAFRAME
        # ========================================================

        if not movies:

            print("\nERROR: No movie data extracted.")

            print(
                "The IMDb page structure may have changed "
                "or the storyline selector needs adjustment."
            )

            return

        df = pd.DataFrame(movies)

        # --------------------------------------------------------
        # SAFETY CHECK
        # --------------------------------------------------------

        required_columns = [
            "Movie Name",
            "Storyline"
        ]

        for column in required_columns:

            if column not in df.columns:
                df[column] = ""

        # --------------------------------------------------------
        # CLEAN DATA
        # --------------------------------------------------------

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

        # Remove empty values
        df = df[
            (df["Movie Name"] != "")
            &
            (df["Storyline"] != "")
        ]

        # Remove duplicate movies
        df = df.drop_duplicates(
            subset=["Movie Name"]
        )

        # Reset index
        df = df.reset_index(drop=True)

        # ========================================================
        # CREATE DATA FOLDER
        # ========================================================

        os.makedirs(
            os.path.dirname(OUTPUT_FILE),
            exist_ok=True
        )

        # ========================================================
        # SAVE CSV
        # ========================================================

        df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

        # ========================================================
        # FINAL OUTPUT
        # ========================================================

        print("\n" + "=" * 60)

        print("SCRAPING COMPLETED")

        print("=" * 60)

        print(
            "Movies saved:",
            len(df)
        )

        print(
            "CSV file:",
            OUTPUT_FILE
        )

        print("\nFirst 5 records:\n")

        print(
            df.head().to_string(index=False)
        )

        print("\nColumns:")
        print(df.columns.tolist())

        print("\nDataset shape:")
        print(df.shape)

    finally:

        driver.quit()

        print("\nChrome closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    scrape_imdb()
# ============================================================
# backend.py  —  THE BRAIN OF THE BOOK RECOMMENDATION PROJECT 📚
# ============================================================
#
# 📌 HOW TO RUN THIS FILE:
#     python backend.py
#
# 📌 WHAT THIS FILE DOES (just like Colab cells, but as functions):
#
#   STEP 1  → Import the libraries we need
#   STEP 2  → Download the dataset from Kaggle
#   STEP 3  → Load the CSV file into a table (DataFrame)
#   STEP 4  → Look at the data (summary, counts, etc.)
#   STEP 5  → Clean the data & prepare it for the recommender
#   STEP 6  → Turn book text into numbers (Bag-of-Words)
#   STEP 7  → Build the recommender "model" (similarity matrix)
#   STEP 8  → Get book recommendations for a chosen book
#   STEP 9  → Draw charts to visualize everything
#   STEP 10 → Self-test: run everything when this file is executed
#
# Nothing fancy here - every function does ONE simple job,
# so it's easy to read from top to bottom.
#
# HOW THE RECOMMENDER WORKS (in plain English):
#   We don't "predict a number" like in a diabetes model. Instead we:
#     1. Combine each book's Genres + Author + Description into one
#        block of text (this is the book's "fingerprint").
#     2. Convert that text into a bag-of-words vector (word counts) —
#        just plain word-count overlap, no TF-IDF weighting and no
#        neural network involved.
#     3. Compare every book's word-count vector to every other book's
#        vector using "cosine similarity" (a way of measuring how
#        similar two lists of numbers are).
#     4. When a user picks a book they like, we look up the books with
#        the MOST similar vectors and recommend those.
# ============================================================


# ------------------------------------------------------------
# STEP 1 — Import the libraries we need
# ------------------------------------------------------------
import os                  # for working with files & folders
import glob                # for finding files that match a pattern
import ast                 # bstract Syntax Trees -> for safely turning "['Fantasy', 'Fiction']" (text) into a real list
import difflib              # for matching a typo-ish book title to the closest real title
from pathlib import Path

import joblib               # for saving/loading our trained recommender
import kagglehub            # for downloading the dataset from Kaggle

import numpy as np          # numbers & arrays
import pandas as pd         # data tables (DataFrames)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings("ignore")   # hide noisy warning messages

# Where we will save our trained recommender on disk
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "book_recommender.joblib"

# The columns we expect to find in the Kaggle dataset:
#   Book, Author, Description, Genres, Avg_Rating, Num_Ratings, URL
TEXT_FEATURE_COLUMNS = ["Genres", "Author", "Description"]   # used to build each book's "fingerprint"




# ------------------------------------------------------------
# STEP 2 — Download the dataset from Kaggle
# ------------------------------------------------------------
def download_dataset():
    """
    Download the "Best Books (10k) Multi-Genre Data" dataset from Kaggle.
    Returns the full path to the CSV file.
    """
    print(" Downloading dataset from Kaggle...")
    folder_path = kagglehub.dataset_download(
        "ishikajohari/best-books-10k-multi-genre-data"
    )
    print(f"Dataset folder: {folder_path}")

    # Find the CSV file inside the downloaded folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {folder_path}")

    csv_path = csv_files[0]
    print(f"CSV file: {csv_path}")
    return csv_path


# ------------------------------------------------------------
# STEP 3 — Load the CSV into a DataFrame
# ------------------------------------------------------------
def load_data(csv_path):
    """Read the CSV file into a pandas table (DataFrame)."""
    df = pd.read_csv(csv_path)
    print(f" Data loaded! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ------------------------------------------------------------
# STEP 4 — Explore the data
# ------------------------------------------------------------
def get_data_summary(df):
    """
    Build a simple dictionary of useful facts about the dataset,
    so the frontend (app.py) can display them easily.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "total_rows": len(df),
        "total_cols": df.shape[1],
        "unique_authors": df["Author"].nunique() if "Author" in df.columns else 0,
    }


# ------------------------------------------------------------
# STEP 5 — Clean & preprocess the data
# ------------------------------------------------------------
def _parse_genres(genres_value):
    """
    The 'Genres' column looks like a Python list written as text, e.g.
        "['Fantasy', 'Fiction', 'Young Adult']"
    This little helper turns that text into a REAL Python list.
    If anything goes wrong, it just returns an empty list.
    """
    try:
        parsed = ast.literal_eval(genres_value)
        if isinstance(parsed, list):
            # Return a list of cleaned genre strings (remove whitespace)
            return [str(g).strip() for g in parsed if str(g).strip()]
        return []
    except (ValueError, SyntaxError):
        return []


def preprocess_data(df):
    """
    Get the raw data ready for the recommender:
      1. Remove duplicate books
      2. Drop rows missing the important text columns
      3. Turn the 'Genres' text into a real list, then a clean string
      4. Clean up the rating columns (make sure they are numbers)
      5. Build one "content" column that combines Genres + Author +
         Description — this is what the recommender will compare.
    """
    df_clean = df.copy()   # never touch the original data!

    # 1. Remove duplicate books (same title + author)
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["Book", "Author"])
    print(f" Removed {before - len(df_clean)} duplicate books")

    # 2. Drop rows that are missing the text we need to compare books
    df_clean = df_clean.dropna(subset=["Book", "Author", "Description", "Genres"])

    # 3. Turn "['Fantasy', 'Fiction']" (text) into a real list, then a clean string
    df_clean["genres_list"] = df_clean["Genres"].apply(_parse_genres)
    df_clean = df_clean[df_clean["genres_list"].map(len) > 0]   # drop books with no genres
    df_clean["genres_clean"] = df_clean["genres_list"].apply(lambda g: " ".join(g))

    # 4. Clean up the numeric columns
    #    Avg_Rating should be a float like 4.32
    df_clean["Avg_Rating"] = pd.to_numeric(df_clean["Avg_Rating"], errors="coerce")
    #    Num_Ratings sometimes has commas, e.g. "1,234,567" -> 1234567
    df_clean["Num_Ratings"] = (
        df_clean["Num_Ratings"].astype(str).str.replace(",", "", regex=False)
    )
    df_clean["Num_Ratings"] = pd.to_numeric(df_clean["Num_Ratings"], errors="coerce")

    # Drop rows where the ratings didn't clean up properly
    df_clean = df_clean.dropna(subset=["Avg_Rating", "Num_Ratings"])

    # 5. Build the "content" column — each book's text fingerprint.
    #    We repeat the genres twice so they count a bit more than the
    #    description when comparing books (genres = strong hint of "similar" books).
    df_clean["content"] = (
        df_clean["genres_clean"] + " " + df_clean["genres_clean"] + " "
        + df_clean["Author"].astype(str) + " "
        + df_clean["Description"].astype(str)
    )

    # Give every book a clean, simple index (0, 1, 2, ...)
    df_clean = df_clean.reset_index(drop=True)

    print(f"After cleaning: {len(df_clean)} books remain")
    return df_clean


# ------------------------------------------------------------
# STEP 6 & 7 — Turn text into numbers (Bag-of-Words) and build
#               the recommender
# ------------------------------------------------------------
def build_recommender(df_clean):
    """
    Build the "model" for our recommender, just plain word-count overlap:
      1. Convert every book's 'content' text into a bag-of-words
         vector (word counts).
      2. Compare every book's vector to every other book's vector
         directly with cosine similarity.

    Returns:
        vectorizer  -> the bag-of-words tool (kept so we could reuse it later)
        cosine_sim  -> a big table of "how similar is book i to book j"
    """
    print("Converting book text into bag-of-words vectors...")
    #CountVectorizer-> convert into numbers
    #the vectorizer keeps only the 8,000 most informative/frequent words.
    vectorizer = CountVectorizer(stop_words="english", max_features=8000)
    bow_matrix = vectorizer.fit_transform(df_clean["content"]).toarray().astype("float32")

    print("Comparing every book to every other book (cosine similarity)...")
    cosine_sim = cosine_similarity(bow_matrix, bow_matrix)

    print(f"Recommender ready! Similarity table shape: {cosine_sim.shape}")
    return vectorizer, cosine_sim


def save_model(vectorizer, cosine_sim, df_clean, model_path=MODEL_PATH):
    """
    Save everything the app needs to make recommendations later:
    the vectorizer, the similarity table, and the cleaned book list.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": vectorizer,
        "cosine_sim": cosine_sim,
        "books": df_clean[["Book", "Author", "Genres", "genres_clean",
                            "Avg_Rating", "Num_Ratings", "URL"]],
    }
    joblib.dump(bundle, model_path)
    print(f"💾 Recommender saved to: {model_path}")
    return model_path


def load_saved_model(model_path=MODEL_PATH):
    """Load a recommender that was already built and saved."""
    if not model_path.exists():
        raise FileNotFoundError(f"No saved recommender found at {model_path}. Build one first!")
    return joblib.load(model_path)


def train_and_save_model():
    """Run the whole pipeline from start to finish, and save the result."""
    csv_path = download_dataset()
    df = load_data(csv_path)
    df_clean = preprocess_data(df)

    vectorizer, cosine_sim = build_recommender(df_clean)
    save_model(vectorizer, cosine_sim, df_clean)

    return {"vectorizer": vectorizer, "cosine_sim": cosine_sim, "books": df_clean}, df_clean


# ------------------------------------------------------------
# STEP 8 — Get recommendations for a chosen book
# ------------------------------------------------------------
def find_closest_title(book_title, books_df):
    """
    People rarely type a book title with perfect spelling/casing.
    This finds the closest matching title we actually have in the data.
    """
    # This gets all the book titles from the DataFrame and stores them in a Python list
    all_titles = books_df["Book"].tolist()
    # It compares the user's input with every title in all_titles and returns the closest matching title
    # book_title -> the title the user typed in
    # n ->Return only one match (the best one)
    # 0.4 -> Moderately similar
    matches = difflib.get_close_matches(book_title, all_titles, n=1, cutoff=0.4)
    return matches[0] if matches else None


def get_recommendations(book_title, bundle, top_n=5):
    """
    The main "prediction" function of this project!

    Given a book the user likes, find the books with the most similar
    "fingerprint" (genres + author + description) and return them.
    """
    books_df = bundle["books"]
    cosine_sim = bundle["cosine_sim"]

    # Find the row number (index) of the chosen book
    matches = books_df.index[books_df["Book"] == book_title].tolist()
    if not matches:
        # Try to auto-correct small typos in the title
        closest = find_closest_title(book_title, books_df)
        if closest is None:
            return pd.DataFrame()   # no match found — return an empty table
        matches = books_df.index[books_df["Book"] == closest].tolist()

    book_idx = matches[0]

    # Look up how similar every other book is to this one... compare selected book with others
    similarity_scores = list(enumerate(cosine_sim[book_idx]))

    # Sort books from MOST similar to LEAST similar
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Skip index 0 because that's the book itself (100% similar to itself!)
    top_matches = similarity_scores[1: top_n + 1]

    result_rows = []
    for idx, score in top_matches:
        row = books_df.iloc[idx]
        result_rows.append({
            "Book": row["Book"],
            "Author": row["Author"],
            "Genres": row["genres_clean"],
            "Avg_Rating": row["Avg_Rating"],
            "Num_Ratings": int(row["Num_Ratings"]),
            "URL": row["URL"],
            "Similarity %": round(score * 100, 1),
        })

    return pd.DataFrame(result_rows)


# ------------------------------------------------------------
# STEP 10 — Self-test (runs only when you do: python backend.py)
# ------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  backend.py - Full Pipeline Self-Test")
    print("=" * 60)

    print("\n[1/5] Downloading & loading data...")
    csv_path = download_dataset()
    df = load_data(csv_path)

    print("\n[2/5] Data summary...")
    summary = get_data_summary(df)
    print(f"   Rows: {summary['total_rows']:,} | Columns: {summary['total_cols']}")

    print("\n[3/5] Cleaning & preprocessing...")
    df_clean = preprocess_data(df)

    print("\n[4/5] Building the recommender...")
    vectorizer, cosine_sim = build_recommender(df_clean)
    save_model(vectorizer, cosine_sim, df_clean)

    print("\n[5/5] Trying out a recommendation...")
    bundle = load_saved_model()
    sample_title = df_clean["Book"].iloc[0]
    recs = get_recommendations(sample_title, bundle, top_n=5)
    print(f"   Because you liked: '{sample_title}'")
    print(recs[["Book", "Author", "Similarity %"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("  ALL STEPS PASSED — backend.py works correctly!")
    print("  Now run: streamlit run app.py")
    print("=" * 60 + "\n")
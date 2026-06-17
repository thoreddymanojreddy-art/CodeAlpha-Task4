# ============================================================
# CodeAlpha Internship — Task 4: Sentiment Analysis
# Description: Analyze sentiment of Amazon product reviews
#              using VADER (rule-based) and TextBlob (NLP)
# Libraries  : pandas, nltk, textblob, matplotlib, seaborn,
#              wordcloud
# ============================================================

# Install dependencies:
# pip install pandas nltk textblob matplotlib seaborn wordcloud

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# NLTK setup
import nltk
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from textblob import TextBlob

sns.set_theme(style="whitegrid")


# ── 1. SAMPLE DATA ──────────────────────────────────────────
# (Replace this with a real CSV if you have one)
SAMPLE_REVIEWS = [
    "This product is absolutely amazing! Best purchase ever.",
    "Terrible quality. Broke after one day. Do not buy.",
    "It's okay, nothing special. Works as described.",
    "Exceeded my expectations! Highly recommend to everyone.",
    "Worst product I have ever used. Complete waste of money.",
    "Pretty decent for the price. Would buy again maybe.",
    "Fantastic! Fast delivery and great packaging.",
    "Not worth it at all. Very disappointed with the quality.",
    "Average product. Does the job but nothing impressive.",
    "Love it! Using it every day. Super happy with my purchase.",
    "Poor build quality. Returned immediately.",
    "Neutral experience overall. Some good, some bad parts.",
    "Outstanding performance! Totally worth every penny.",
    "Horrible experience. Customer service was useless too.",
    "Quite good actually. Pleasantly surprised.",
    "Mediocre at best. Expected much more for this price.",
    "Brilliant product! Changed my daily routine completely.",
    "Garbage. Stopped working in a week.",
    "So-so. Not impressed but not terrible either.",
    "Incredible value for money. Will definitely buy again!",
    "Delivery was late and product was damaged on arrival.",
    "Works perfectly. Simple setup and easy to use.",
    "The product feels cheap and looks nothing like the photos.",
    "Really happy with this. Great quality materials.",
    "Returned within a day. Completely useless product.",
]


# ── 2. LOAD / CREATE DATAFRAME ──────────────────────────────
def load_data():
    """Load reviews from CSV or use sample data."""
    # Uncomment to load your own CSV:
    # df = pd.read_csv("reviews.csv")
    # return df

    df = pd.DataFrame({"review": SAMPLE_REVIEWS})
    print(f"Loaded {len(df)} reviews.")
    return df


# ── 3. TEXT CLEANING ────────────────────────────────────────
def clean_text(text):
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)          # Remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()          # Normalize whitespace
    return text


# ── 4. VADER SENTIMENT ──────────────────────────────────────
def apply_vader(df):
    """Apply VADER sentiment analysis."""
    sia = SentimentIntensityAnalyzer()

    def get_vader_sentiment(text):
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return pd.Series([compound, label])

    df[["vader_score", "vader_label"]] = df["review"].apply(get_vader_sentiment)
    return df


# ── 5. TEXTBLOB SENTIMENT ───────────────────────────────────
def apply_textblob(df):
    """Apply TextBlob sentiment analysis."""
    def get_textblob_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0.05:
            label = "Positive"
        elif polarity < -0.05:
            label = "Negative"
        else:
            label = "Neutral"
        return pd.Series([polarity, subjectivity, label])

    df[["tb_polarity", "tb_subjectivity", "tb_label"]] = \
        df["clean_review"].apply(get_textblob_sentiment)
    return df


# ── 6. VISUALIZATIONS ───────────────────────────────────────
def plot_sentiment_distribution(df):
    """Bar chart of sentiment counts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentiment Distribution", fontsize=15, fontweight="bold")

    colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}

    for ax, col, title in zip(
        axes,
        ["vader_label", "tb_label"],
        ["VADER Sentiment", "TextBlob Sentiment"]
    ):
        counts = df[col].value_counts()
        bar_colors = [colors.get(label, "gray") for label in counts.index]
        bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="black")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=150)
    plt.show()
    print("Saved: sentiment_distribution.png")


def plot_score_histogram(df):
    """Histogram of VADER compound scores."""
    plt.figure(figsize=(10, 5))
    plt.hist(df["vader_score"], bins=20, color="steelblue",
             edgecolor="black", alpha=0.8)
    plt.axvline(x=0.05, color="green", linestyle="--", label="Positive threshold (0.05)")
    plt.axvline(x=-0.05, color="red", linestyle="--", label="Negative threshold (-0.05)")
    plt.xlabel("VADER Compound Score", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.title("Distribution of VADER Sentiment Scores", fontsize=14, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("sentiment_scores_histogram.png", dpi=150)
    plt.show()
    print("Saved: sentiment_scores_histogram.png")


def plot_polarity_vs_subjectivity(df):
    """Scatter: TextBlob polarity vs subjectivity."""
    colors = {"Positive": "#2ecc71", "Neutral": "#f39c12", "Negative": "#e74c3c"}
    color_list = df["tb_label"].map(colors)

    plt.figure(figsize=(10, 6))
    plt.scatter(df["tb_polarity"], df["tb_subjectivity"],
                c=color_list, s=100, alpha=0.8, edgecolors="black")

    # Legend
    handles = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
    plt.legend(handles=handles, title="Sentiment")
    plt.xlabel("Polarity (Negative ←→ Positive)", fontsize=12)
    plt.ylabel("Subjectivity (Objective ←→ Subjective)", fontsize=12)
    plt.title("TextBlob: Polarity vs Subjectivity", fontsize=14, fontweight="bold")
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    plt.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("sentiment_polarity_subjectivity.png", dpi=150)
    plt.show()
    print("Saved: sentiment_polarity_subjectivity.png")


def plot_wordcloud(df):
    """Word cloud of all review words."""
    try:
        from wordcloud import WordCloud
        stop_words = set(stopwords.words("english"))
        all_text = " ".join(df["clean_review"].tolist())
        words = [w for w in all_text.split() if w not in stop_words and len(w) > 2]
        text_for_cloud = " ".join(words)

        wc = WordCloud(width=900, height=450, background_color="white",
                       colormap="RdYlGn", max_words=100).generate(text_for_cloud)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud — Most Frequent Review Words", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sentiment_wordcloud.png", dpi=150)
        plt.show()
        print("Saved: sentiment_wordcloud.png")

    except ImportError:
        print("wordcloud not installed — skipping. Run: pip install wordcloud")


# ── 7. SUMMARY STATS ────────────────────────────────────────
def print_summary(df):
    print("\n" + "=" * 50)
    print("  SENTIMENT ANALYSIS SUMMARY")
    print("=" * 50)

    print("\n── VADER Results ──")
    print(df["vader_label"].value_counts().to_string())
    print(f"Average Compound Score: {df['vader_score'].mean():.3f}")

    print("\n── TextBlob Results ──")
    print(df["tb_label"].value_counts().to_string())
    print(f"Average Polarity     : {df['tb_polarity'].mean():.3f}")
    print(f"Average Subjectivity : {df['tb_subjectivity'].mean():.3f}")

    print("\n── Sample Predictions ──")
    print(df[["review", "vader_label", "tb_label"]].head(8).to_string(index=False))


# ── MAIN ────────────────────────────────────────────────────
def main():
    import matplotlib.patches as mpatches
    globals()["mpatches"] = mpatches

    print("=" * 50)
    print("  CodeAlpha — Task 4: Sentiment Analysis")
    print("=" * 50)

    df = load_data()
    df["clean_review"] = df["review"].apply(clean_text)
    df = apply_vader(df)
    df = apply_textblob(df)

    print_summary(df)
    plot_sentiment_distribution(df)
    plot_score_histogram(df)
    plot_polarity_vs_subjectivity(df)
    plot_wordcloud(df)

    # Save results
    df.to_csv("sentiment_results.csv", index=False)
    print("\nResults saved to 'sentiment_results.csv'")
    print("Sentiment Analysis complete!")


if __name__ == "__main__":
    main()

# HOTEL REVIEW NLP + LDA PIPELINE
# AUTHOR: Havva Aygün
# DESCRIPTION: Full end-to-end cleaning, topic modeling, sentiment analysis,
#              trend extraction, and Power BI export pipeline.
"""
CLEAN LDA HOTEL ANALYSIS (ENGLISH-ONLY)
=======================================
- English-only filtering
- Remove "read more / less"
- Aggressive cleaning
- LDA topic modeling
- Topic sentiment classification
- Power BI için: reviews.csv, trend.csv, topic_trend.csv
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ===================== PATHS ===================== #
BASE_DIR = Path("/Users/havvaaygun/Desktop/OTEL ANALİZ/yeni_otel")
DATA_FILE = BASE_DIR / "Hotel Reviews Data.csv"

# ÇIKTI DOSYALARI
OUTPUT_EXCEL  = BASE_DIR / "otel_rapor.xlsx"
OUTPUT_TXT    = BASE_DIR / "otel_rapor.txt"
REVIEWS_CSV   = BASE_DIR / "reviews.csv"
TREND_CSV     = BASE_DIR / "trend.csv"
TOPIC_TREND_CSV = BASE_DIR / "topic_trend.csv"
TOPIC_SUMMARY_CSV = BASE_DIR / "topic_summary_clean.csv"

FIG_TREND    = BASE_DIR / "trend_yorum_sayisi.png"
FIG_SENT     = BASE_DIR / "trend_sentiment.png"
FIG_TOPICS   = BASE_DIR / "topic_distribution.png"


# ================================================= #
# =============== CLEANING FUNCTION =============== #
def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # "read more / read less" çöplüğünü sil
    text = text.replace("read more", " ")
    text = text.replace("read less", " ")

    # İngilizce olmayan karakterleri tamamen at (sadece ascii kalsın)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # sadece harf
    text = re.sub(r"[^a-z\s]", " ", text)

    # çok kısa saçma kelimeleri at (la, en, q vs.)
    text = " ".join([w for w in text.split() if len(w) > 2])

    # stopwords at
    stops = set(stopwords.words("english"))
    text = " ".join([w for w in text.split() if w not in stops])

    return text.strip()


# ================================================= #
# ===================== LOAD DATA ================= #
print("📥 CSV yükleniyor...")

try:
    df = pd.read_csv(DATA_FILE, encoding="utf-8")
except:
    df = pd.read_csv(DATA_FILE, encoding="latin1")

df.columns = ["review", "date", "location"]

# Tarihi parse et + year_month üret
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year_month"] = df["date"].dt.to_period("M").astype(str)

print("Toplam ham yorum:", len(df))

# CLEANING
df["clean_text"] = df["review"].apply(clean_text)

# Boş kalanları at
df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)

print("Temizlenmiş yorum sayısı:", len(df))


# ================================================= #
# =============== SENTIMENT ANALYSIS ============== #
print("\n❤️  Sentiment analizi...")

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(x):
    s = analyzer.polarity_scores(x)["compound"]
    return "positive" if s >= 0.2 else "negative" if s <= -0.2 else "neutral"

df["sentiment"] = df["clean_text"].apply(get_sentiment)

print(df["sentiment"].value_counts())


# ================================================= #
# =============== LDA TOPIC MODELING ============== #
print("\n🧠 LDA Topic modeling başlıyor...")

NUM_TOPICS = 8  # istersen değiştir

vectorizer = CountVectorizer(
    max_df=0.90,
    min_df=20,
    stop_words="english"
)

X = vectorizer.fit_transform(df["clean_text"])

lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    random_state=42,
    learning_method="batch"
)

lda.fit(X)
terms = vectorizer.get_feature_names_out()

topic_keywords = []
for idx, topic in enumerate(lda.components_):
    words = [terms[i] for i in topic.argsort()[:-15:-1]]
    topic_keywords.append(", ".join(words))

topic_df = pd.DataFrame({
    "topic_id": range(NUM_TOPICS),
    "keywords": topic_keywords
})

# her yorumun topic'i
topic_values = lda.transform(X)
df["topic"] = topic_values.argmax(axis=1)

# yorumlara topic_keywords ekle (Power BI için çok iyi)
topic_map = topic_df.set_index("topic_id")["keywords"].to_dict()
df["topic_keywords"] = df["topic"].map(topic_map)

print("\n✔ TEMİZ TOPİCLER:")
print(topic_df)


# ================================================= #
# ========== TOPIC SENTIMENT CLASSIFICATION ======= #
topic_sentiment = df.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
topic_sentiment["total"]    = topic_sentiment.sum(axis=1)
topic_sentiment["pos_rate"] = topic_sentiment["positive"] / topic_sentiment["total"]
topic_sentiment["neg_rate"] = topic_sentiment["negative"] / topic_sentiment["total"]

def classify_topic(row):
    if row["neg_rate"] > 0.35:
        return "NEGATIVE TOPIC"
    elif row["pos_rate"] > 0.45:
        return "POSITIVE TOPIC"
    else:
        return "MIXED"

topic_sentiment["label"] = topic_sentiment.apply(classify_topic, axis=1)

topic_summary = pd.concat(
    [topic_df.set_index("topic_id"), topic_sentiment],
    axis=1
).reset_index().rename(columns={"index": "topic"})

print("\n📌 TOPIC–SENTIMENT ÖZET (EN TEMİZ HALİ):")
print(topic_summary)

topic_summary.to_csv(TOPIC_SUMMARY_CSV, index=False)
print("💾 topic_summary_clean.csv kaydedildi.")


# ================================================= #
# =============== ZAMAN ANALİZİ (TREND) =========== #
print("\n📈 Zaman analizi hesaplanıyor...")

trend = df.groupby("year_month").agg(
    yorum_sayisi=("review", "count"),
    pozitif=("sentiment", lambda x: (x == "positive").sum()),
    negatif=("sentiment", lambda x: (x == "negative").sum())
).reset_index()

trend["negatif_oran"] = trend["negatif"] / trend["yorum_sayisi"]

print(trend.head())

# Power BI için trend.csv
trend.to_csv(TREND_CSV, index=False)
print(f"💾 trend.csv kaydedildi → {TREND_CSV}")


# ================================================= #
# =============== TOPIC TREND (AY-TOPIC) ========== #
topic_trend = (
    df.groupby(["year_month", "topic"])
      .size()
      .reset_index(name="count")
)

topic_trend.to_csv(TOPIC_TREND_CSV, index=False)
print(f"💾 topic_trend.csv kaydedildi → {TOPIC_TREND_CSV}")


# ================================================= #
# =============== REVIEWS (ANA TABLO) ============= #
# Power BI için ana tablo
reviews_cols = [
    "review",
    "clean_text",
    "date",
    "year_month",
    "location",
    "sentiment",
    "topic",
    "topic_keywords"
]

reviews_out = df[reviews_cols].copy()
reviews_out.to_csv(REVIEWS_CSV, index=False)
print(f"💾 reviews.csv kaydedildi → {REVIEWS_CSV}")


# ================================================= #
# =============== BASİT GRAFİKLER ================= #
print("\n📊 Grafikler oluşturuluyor...")

# yorum sayısı
plt.figure(figsize=(10, 4))
plt.plot(trend["year_month"], trend["yorum_sayisi"])
plt.xticks(rotation=90)
plt.title("Zaman İçinde Yorum Sayısı")
plt.tight_layout()
plt.savefig(FIG_TREND)
plt.close()

# negatif oran
plt.figure(figsize=(10, 4))
plt.plot(trend["year_month"], trend["negatif_oran"], color="red")
plt.xticks(rotation=90)
plt.title("Negatif Yorum Oranı")
plt.tight_layout()
plt.savefig(FIG_SENT)
plt.close()

# topic dağılımı
topic_counts = df["topic"].value_counts().sort_index()
plt.figure(figsize=(8, 4))
plt.bar(topic_counts.index.astype(str), topic_counts.values)
plt.title("Topic Dağılımı")
plt.tight_layout()
plt.savefig(FIG_TOPICS)
plt.close()

print("✔ Grafikler kaydedildi.")


# ================================================= #
# =============== TXT RAPOR (OPSİYONEL) =========== #
print("\n📝 TXT rapor hazırlanıyor...")

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write("CLEAN LDA HOTEL ANALYSIS\n")
    f.write("========================\n\n")

    f.write(f"Toplam yorum: {len(df)}\n\n")

    f.write("Sentiment dağılımı:\n")
    f.write(str(df["sentiment"].value_counts()))
    f.write("\n\n")

    f.write("TOPICS:\n")
    for _, row in topic_summary.iterrows():
        f.write(
            f"- Topic {int(row['topic'])} "
            f"({row['label']} – pos_rate={row['pos_rate']:.2f}, "
            f"neg_rate={row['neg_rate']:.2f}): "
            f"{row['keywords']}\n"
        )

print("✔ TXT raporu oluşturuldu →", OUTPUT_TXT)


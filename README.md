Hotel Review NLP + LDA Topic Modeling Pipeline

This project implements a full end-to-end NLP + Machine Learning pipeline on hotel review datasets.
It includes text cleaning, sentiment analysis, topic modeling, trend extraction, and Power BI–ready outputs.



 Project Overview

This pipeline processes raw hotel reviews and produces:

✔ Cleaned review texts

✔ Sentiment labels (positive / neutral / negative)

✔ LDA topic modeling with keywords

✔ Topic sentiment classification

✔ Monthly review & sentiment trends

✔ Topic–month frequency trends

✔ Export-ready CSVs for Power BI dashboards

All outputs are automatically generated inside the output/ directory.



 1. Text Cleaning
	•	Lowercasing
	•	Removing “read more / read less” artifacts
	•	Removing non-English characters
	•	Removing non-alphabetic characters
	•	Stopword removal
	•	Short word filtering


 2. Sentiment Analysis (VADER)

Each review receives one of three labels:
	•	positive
	•	neutral
	•	negative



 3. Topic Modeling (LDA)

The model extracts 8 latent topics using:
	•	CountVectorizer
	•	LatentDirichletAllocation
	•	Top 15 keywords per topic

Each review is assigned:
	•	topic_id
	•	topic_keywords



 4. Trend Analysis

The pipeline generates:

trend.csv
	•	Monthly review count
	•	Positive/negative review count
	•	Negative ratio

topic_trend.csv
	•	Topic counts per month

5. Visualizations Generated
	•	Monthly review count
	•	Negative review rate trend
	•	Topic distribution bar chart

All images are saved under output/.

6. Run the Pipeline
python src/analysis_pipeline.py

7. Requirements
pandas
numpy
matplotlib
scikit-learn
nltk
vaderSentiment

Install with:
pip install -r requirements.txt


Author

Havva Aygün
NLP, Data Analytics, Machine Learning, Business Intelligence

# -havvaaygun

# Customer Feedback Intelligence — NLP Sentiment Analysis

An NLP and machine learning application that analyzes customer reviews and classifies them as **Positive, Neutral, or Negative**. The application provides single-review predictions, bulk CSV analysis, sentiment distribution charts, and downloadable results through an interactive Streamlit interface.

## Features

- **Single Review Prediction:** Enter a review and predict its sentiment.
- **Bulk Review Analysis:** Upload a CSV file and analyze multiple reviews.
- **Sentiment Distribution:** View sentiment counts and percentages.
- **Visualization:** Explore the distribution of predicted sentiments using a bar chart.
- **Download Results:** Export predictions as a CSV file.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorization
- Linear Support Vector Classification (LinearSVC)
- Streamlit
- Matplotlib
- Joblib
- Jupyter Notebook

## Project Structure

```text
Customer-Feedback-Intelligence-NLP-Sentiment-Analysis/
├── NLP_PROJECT.ipynb
├── nlp_app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

## How It Works

1. Customer review text is provided as input.
2. TF-IDF converts text into numerical features.
3. The trained machine learning model predicts the sentiment.
4. The application displays predictions and visualizes sentiment distribution.
5. For CSV uploads, the predicted results can be downloaded for further analysis.

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/banuprakashreddy1111/Customer-Feedback-Intelligence-NLP-Sentiment-Analysis.git
cd Customer-Feedback-Intelligence-NLP-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run nlp_app.py
```

Open the local URL displayed in your terminal.

## Input Format

For bulk analysis, upload a CSV file containing customer review text. Select the appropriate text column in the application before running predictions.

Example:

| reviewText |
|---|
| This product is amazing and works perfectly |
| The quality is terrible and disappointing |
| The product is okay, nothing special |

## Model Evaluation

The tuned LinearSVC model achieved approximately **80.89% test accuracy** and **0.673 macro F1-score** in the evaluation performed during development. Macro F1 is included because it provides a useful view of performance across sentiment classes, including the less frequent Neutral class.

## Future Improvements

- Improve Neutral sentiment classification.
- Add model evaluation visualizations and error analysis.
- Deploy the Streamlit application online.
- Extend the analysis with aspect-based sentiment analysis.

## Author

**Banu Prakash Reddy**

GitHub: [banuprakashreddy1111](https://github.com/banuprakashreddy1111)

---

This project demonstrates practical experience in natural language processing, text vectorization, machine learning classification, data visualization, and interactive application development.


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Feedback Intelligence",
    page_icon="💬",
    layout="wide"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_artifacts():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

SENTIMENT_NAMES = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

# Application header
st.title("💬 Customer Feedback Intelligence")
st.write(
    "Analyze customer reviews using NLP and Machine Learning."
)

st.divider()

# Single-review prediction
st.subheader("🔍 Analyze a Single Review")

review = st.text_area(
    "Enter a customer review",
    placeholder="Example: This product is amazing and works perfectly"
)

if st.button("Analyze Sentiment", key="single_review"):
    if review.strip():
        features = vectorizer.transform([review])
        prediction = int(model.predict(features)[0])

        st.metric(
            "Predicted Sentiment",
            SENTIMENT_NAMES[prediction]
        )
    else:
        st.warning("Please enter a review.")

st.divider()

# Bulk CSV analysis
st.subheader("📁 Analyze Multiple Reviews")
st.write(
    "Upload a CSV file containing a column of customer review text."
)

uploaded_file = st.file_uploader(
    "Upload customer reviews CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data")
        st.dataframe(df.head())

        review_column = st.selectbox(
            "Select the column containing review text",
            options=df.columns
        )

        if st.button("Analyze All Reviews"):
            reviews = df[review_column].fillna("").astype(str)
            valid_mask = reviews.str.strip().ne("")

            if not valid_mask.any():
                st.warning("No non-empty reviews were found.")
            else:
                result_df = df.loc[valid_mask].copy()
                valid_reviews = reviews.loc[valid_mask]

                features = vectorizer.transform(valid_reviews)
                predictions = model.predict(features)

                result_df["Predicted Sentiment"] = [
                    SENTIMENT_NAMES[int(pred)]
                    for pred in predictions
                ]

                st.success(
                    f"Analyzed {len(result_df):,} reviews successfully."
                )

                # Sentiment counts
                sentiment_order = [
                    "Positive", "Neutral", "Negative"
                ]

                counts = (
                    result_df["Predicted Sentiment"]
                    .value_counts()
                    .reindex(sentiment_order, fill_value=0)
                )

                total = len(result_df)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Positive Reviews",
                        f"{counts['Positive']:,}",
                        f"{counts['Positive'] / total:.1%}"
                    )

                with col2:
                    st.metric(
                        "Neutral Reviews",
                        f"{counts['Neutral']:,}",
                        f"{counts['Neutral'] / total:.1%}"
                    )

                with col3:
                    st.metric(
                        "Negative Reviews",
                        f"{counts['Negative']:,}",
                        f"{counts['Negative'] / total:.1%}"
                    )

                # Sentiment distribution chart
                st.subheader("Sentiment Distribution")

                fig, ax = plt.subplots()
                counts.plot(kind="bar", ax=ax)
                ax.set_xlabel("Sentiment")
                ax.set_ylabel("Number of Reviews")
                ax.set_title("Customer Feedback Sentiment")
                ax.tick_params(axis="x", rotation=0)

                st.pyplot(fig)
                plt.close(fig)

                # Results table
                st.subheader("Predicted Reviews")
                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # Download predictions
                csv_data = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Predictions CSV",
                    data=csv_data,
                    file_name="customer_sentiment_predictions.csv",
                    mime="text/csv"
                )

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV file: {exc}")
    except Exception as exc:
        st.error(f"Something went wrong: {exc}")

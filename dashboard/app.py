import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title = "Fraud Detection Dashboard",
    layout = "wide"
)

st.title("Financial Fraud Detection MLOps Dashboard")

# load artifacts
model = joblib.load("models/fraud_model.pkl")
threshold = joblib.load("models/threshold.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.sidebar.header("Model Info")
st.sidebar.metric("Decision Threshold", round(threshold, 4))
st.sidebar.write("Threshold = top 0.5% risk cutoff")

# load scored data
uploaded_file = st.file_uploader("Upload scored transactions CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)

    total_transactions = len(df)
    flagged = df["flag"].sum()
    avg_score = df["fraud_probability"].mean()

    col1.metric("Total Transactions", total_transactions)
    col2.metric("Flagged Transactions", int(flagged))
    col3.metric("Average Fraud Score", round(avg_score, 4))

    st.subheader("Fraud Score Distribution")
    fig = px.histogram(
        df,
        x = "fraud_probability",
        nbins = 50,
        title = "Distribution of Fraud Risk Scores"
    )
    st.plotly_chart(fig, use_container_width = True)

    st.subheader("Flagged vs Not Flagged")
    flag_counts = df["flag"].value_counts().reset_index()
    flag_counts.columns = ["flag", "count"]

    fig = px.bar(
        flag_counts,
        x = "flag",
        y = "count",
        title = "Flagged Transaction Count"
    )
    st.plotly_chart(fig, use_container_width = True)

    if "amt" in df.columns:
        st.subheader("Amount vs Fraud Score")
        fig = px.scatter(
            df,
            x = "amt",
            y = "fraud_probability",
            color = "flag",
            title = "Transaction Amount vs Fraud Probability"
        )
        st.plotly_chart(fig, use_container_width = True)

    if "distance" in df.columns:
        st.subheader("Distance vs Fraud Score")
        fig = px.scatter(
            df,
            x = "distance",
            y = "fraud_probability",
            color = "flag",
            title = "Distance vs Fraud Probability"
        )
        st.plotly_chart(fig, use_container_width = True)

    if "category" in df.columns:
        st.subheader("Average Fraud Score by Category")
        category_scores = (
            df.groupby("category")["fraud_probability"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig = px.bar(
            category_scores,
            x = "category",
            y = "fraud_probability",
            title = "Average Fraud Risk by Category"
        )
        st.plotly_chart(fig, use_container_width = True)

    st.subheader("Highest Risk Transactions")
    st.dataframe(
        df.sort_values("fraud_probability", ascending = False).head(20)
    )

else:
    st.info("Upload a scored transactions CSV to view dashboard.")
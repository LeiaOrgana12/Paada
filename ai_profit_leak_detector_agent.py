
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Profit Leak Detector", layout="wide")
st.title("🧠 AI-Powered Profit Leak Detector with Explanations")

@st.cache_data
def load_model_and_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['Delta_Amount'] = df['Amount'] - df['Expected_High']
    df['Percent_Variance'] = ((df['Amount'] - df['Expected_High']) / df['Expected_High']).round(2)
    df['Month_Num'] = pd.to_datetime(df['Month']).dt.month
    df['Is_SaaS'] = df['Category'].str.contains("Software", case=False).astype(int)

    X = df[['Amount', 'Expected_Low', 'Expected_High', 'Delta_Amount', 
            'Percent_Variance', 'Month_Num', 'Is_SaaS']]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, df['Leak_Flag'] if 'Leak_Flag' in df.columns else [0]*len(df))
    df['AI_Prediction'] = model.predict(X)

    def generate_explanation(row):
        if row['Percent_Variance'] > 0.5 and row['Delta_Amount'] > 1000:
            return f"Spending on '{row['Subcategory']}' in {row['Month']} was unusually high—{int(row['Percent_Variance']*100)}% above normal. Consider reviewing this cost."
        elif row['Is_SaaS'] == 1 and row['Delta_Amount'] > 500:
            return f"'{row['Subcategory']}' may be part of SaaS redundancy. Check for overlap with other tools."
        elif row['Category'] in ['Contractors', 'Consulting Services'] and row['Delta_Amount'] > 750:
            return f"Large jump in contractor fees detected. Consider moving to fixed-fee or retainer model."
        else:
            return "This cost appears higher than expected. Review context or compare to past months."

    df['AI_Explanation'] = df.apply(lambda row: generate_explanation(row) if row['AI_Prediction'] == 1 else "", axis=1)
    return df

uploaded_file = st.file_uploader("Upload your enriched expense CSV", type=["csv"])

if uploaded_file:
    df = load_model_and_data(uploaded_file)

    st.subheader("📊 Total Spend Over Time")
    monthly_spend = df.groupby("Month")["Amount"].sum().reset_index()
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=monthly_spend, x="Month", y="Amount", marker="o", ax=ax1)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Total Spend (USD)")
    st.pyplot(fig1)

    st.subheader("🚨 Predicted Leaks Summary")
    leak_summary = df[df["AI_Prediction"] == 1]
    total_flagged = leak_summary["Amount"].sum()
    st.metric("Estimated Savings Potential", f"${total_flagged:,.2f}")

    st.dataframe(leak_summary[[
        "Month", "Category", "Subcategory", "Amount", 
        "Expected_High", "Delta_Amount", "Percent_Variance", "AI_Explanation"
    ]].sort_values(by="Delta_Amount", ascending=False), use_container_width=True)

    st.download_button(
        label="Download Flagged Leak Results",
        data=leak_summary.to_csv(index=False),
        file_name="predicted_profit_leaks_with_explanations.csv",
        mime="text/csv"
    )
else:
    st.info("👈 Upload an enriched expense CSV file to begin.")

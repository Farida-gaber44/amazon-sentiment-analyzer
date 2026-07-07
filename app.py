import streamlit as st
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

model = joblib.load("sentiment_model.pkl")

st.title("💬 Amazon Review Sentiment Analyzer")
st.markdown("Paste any product review and get instant sentiment prediction.")
st.markdown("---")

review = st.text_area("✍️ Enter your review here:", height=180,
                       placeholder="e.g. This product is amazing, works perfectly!")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction = model.predict([review])[0]
        proba = model.predict_proba([review])[0]
        classes = model.classes_

        emoji = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}
        st.markdown(f"### Result: {emoji[prediction]} **{prediction}**")

        fig = go.Figure(go.Bar(
            x=list(classes),
            y=[round(p * 100, 1) for p in proba],
            marker_color=["#e84040", "#f7c948", "#00c49a"],
            text=[f"{p*100:.1f}%" for p in proba],
            textposition="outside"
        ))
        fig.update_layout(
            title="Confidence Scores",
            yaxis_title="Confidence (%)",
            yaxis_range=[0, 110],
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

st.caption("Model: TF-IDF + Logistic Regression | Trained on 127,920 Amazon reviews")
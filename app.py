import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from fx_news_sentiment import get_sentiment_for_currencies, summarize_sentiment, prepare_features, train_model, predict_next_sentiment

st.set_page_config(page_title="FX Sentiment Dashboard", layout="wide")
st.title("💱 FX News-Based Sentiment Trading Strategy")

# Sidebar controls
st.sidebar.title("⚙️ Controls")
date_range = st.sidebar.date_input("Select date range", [datetime.date.today() - datetime.timedelta(days=7), datetime.date.today()])
run_analysis = st.sidebar.button("🔄 Refresh Data")

if run_analysis:
    with st.spinner("Fetching latest FX news and analyzing with FinBERT..."):
        df = get_sentiment_for_currencies()
        df.to_csv("fx_news_sentiment.csv", index=False)
        if 'currency' not in df.columns:
            st.error("❌ Missing 'currency' column in data. Check parsing or API structure.")
            st.stop()

        strategies = summarize_sentiment(df)
        st.session_state["df"] = df
        st.session_state["strategies"] = strategies
    st.success("✅ Analysis complete")

# Load from session or file if not refreshed
if "df" not in st.session_state:
    try:
        st.session_state["df"] = pd.read_csv("fx_news_sentiment.csv")
        st.session_state["strategies"] = summarize_sentiment(st.session_state["df"])
    except:
        st.warning("Please click 'Refresh Data' to load news and sentiment analysis.")
        st.stop()

# Variables from session
df = st.session_state["df"]
strategies = st.session_state["strategies"]

# Show strategies
st.subheader("💡 Suggested FX Trading Strategies")
for currency, info in strategies.items():
    st.markdown(
        f"**{currency}** → 📊 `{info['strategy']}` "
        f"(avg sentiment score: `{info['avg_sentiment_score']}`)"
    )
    

# Currency Pair Suggestions
st.subheader("🔗 Currency Pair Strategy Suggestions")
fx_currencies = list(strategies.keys())
col1, col2 = st.columns(2)
with col1:
    base = st.selectbox("Base Currency", fx_currencies, key="base")
with col2:
    quote = st.selectbox("Quote Currency", fx_currencies, key="quote")

def pair_strategy(base, quote, threshold=0.01):
    base_score = strategies.get(base, {}).get("avg_sentiment_score")
    quote_score = strategies.get(quote, {}).get("avg_sentiment_score")

    # Ensure both exist
    if base_score is None or quote_score is None:
        return f"⚠️ No sentiment score for {base}/{quote}"

    diff = base_score - quote_score

    if diff > threshold:
        return f"✅ Suggest going **LONG** on {base}/{quote} (Δ sentiment = +{round(diff, 3)})"
    elif diff < -threshold:
        return f"✅ Suggest going **SHORT** on {base}/{quote} (Δ sentiment = {round(diff, 3)})"
    else:
        return f"⚠️ No strong signal for {base}/{quote} (Δ sentiment = {round(diff, 3)})"


if base and quote and base != quote:
    st.info(pair_strategy(base, quote))

st.divider()

# Currency sentiment breakdown
df["publishedAt"] = pd.to_datetime(df["publishedAt"])
df = df[(df["publishedAt"].dt.date >= date_range[0]) & (df["publishedAt"].dt.date <= date_range[1])]

currencies = sorted(df["currency"].unique())
selected_currency = st.selectbox("📌 Select a currency to view sentiment details", currencies)

st.subheader(f"📈 Sentiment Trend: {selected_currency}")
df_currency = df[df["currency"] == selected_currency].copy()
df_currency['numeric_sentiment'] = df_currency['sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})

if not df_currency.empty:
    trend = df_currency.groupby(df_currency['publishedAt'].dt.date)['numeric_sentiment'].mean().reset_index()
    fig = px.line(trend, x='publishedAt', y='numeric_sentiment',
                  labels={'publishedAt': 'Date', 'numeric_sentiment': 'Avg Sentiment Score'},
                  title=f"Daily Sentiment Score for {selected_currency}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sentiment data found for this currency.")

# News table
st.subheader(f"📰 News Headlines for {selected_currency}")
with st.expander("Click to view articles"):
    for _, row in df_currency.iterrows():
        st.markdown(f"**{row['title']}**  \n*({row['source']}, {row['publishedAt'].date()})*")
        st.markdown(f"Sentiment: `{row['sentiment']}` | Confidence: `{round(row['confidence'], 2)}`")
        st.markdown("---")

df = get_sentiment_for_currencies()
df = prepare_features(df)
data = df.copy()

model = train_model(df)
prediction = predict_next_sentiment(df, model)

# Feature and target
X = df[["sentiment_score"]]      # independent variable
y = df["label"]                 # dependent variable

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if prediction == 1:
    st.success("📈 Sentiment expected to rise tomorrow!")
else:
    st.error("📉 Sentiment likely to fall tomorrow.")

with st.expander("🔍 View Logistic Regression Details"):

    st.subheader("📊 Training Data Preview")
    st.write(data[["sentiment_score", "label"]].tail(10))

    st.subheader("📈 Coefficients")
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    st.markdown(f"**Intercept**: {intercept:.4f}  \n**Coefficient (sentiment score)**: {coef:.4f}")

    st.subheader("📉 Performance Metrics")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.subheader("🔁 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.subheader("🧠 Example Predictions")
    data['predicted_prob'] = model.predict_proba(X)[:, 1]
    data['predicted_class'] = model.predict(X)
    st.dataframe(data[['publishedAt', 'sentiment_score', 'label', 'predicted_class', 'predicted_prob']].tail(10))



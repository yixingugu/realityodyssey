# fx_news_sentiment.py

import requests
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import datetime

# Config
NEWS_API_KEY = '69d9d57d91794ca5bae9c65c923de31e'
FX_KEYWORDS = ['USD', 'EUR', 'JPY', 'GBP', 'CHF']
COUNTRY = 'us'  # or 'sg', 'gb', 'global'
NUM_ARTICLES = 100

# Load FinBERT
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def get_news_from_newsapi(query, limit=NUM_ARTICLES):
    url = ('https://newsapi.org/v2/everything?'
           f'q={query}&language=en&pageSize={limit}&apiKey={NEWS_API_KEY}')
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    return [{
        'currency': query,
        'title': a['title'],
        'description': a['description'],
        'publishedAt': a['publishedAt'],
        'source': a['source']['name']
    } for a in articles]

def analyze_sentiment_finbert(texts):
    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            label_id = torch.argmax(probs).item()
            sentiment = ['negative', 'neutral', 'positive'][label_id]
            confidence = probs[0][label_id].item()
            results.append((sentiment, confidence))
    return results

def get_sentiment_for_currencies():
    all_data = []
    for keyword in FX_KEYWORDS:
        print(f"Fetching news for {keyword}...")
        articles = get_news_from_newsapi(keyword)
        titles = [a['title'] for a in articles]
        sentiments = analyze_sentiment_finbert(titles)

        for article, (label, confidence) in zip(articles, sentiments):
            sentiment_score = {'positive': 1, 'neutral': 0, 'negative': -1}[label]
            article.update({
                'sentiment': label,
                'confidence': confidence,
                'sentiment_score': sentiment_score
            })
        all_data.extend(articles)
    
    df = pd.DataFrame(all_data)

    if 'publishedAt' not in df.columns:
        print("⚠️ Warning: 'publishedAt' column not found in DataFrame.")
        print("Available columns:", df.columns)
        return pd.DataFrame()

    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df.sort_values('publishedAt', inplace=True)
    return df


def summarize_sentiment(df):
    summary = {}
    for currency in FX_KEYWORDS:
        subset = df[df['currency'] == currency]
        if subset.empty:
            continue
        avg_score = subset['sentiment_score'].mean()
        if avg_score > 0.1:
            strategy = "Long"
        elif avg_score < -0.1:
            strategy = "Short"
        else:
            strategy = "Neutral"
        summary[currency] = {
            'avg_sentiment_score': round(avg_score, 3),
            'strategy': strategy
        }
    return summary


def prepare_features(df):
    df['rolling_mean_3'] = df.groupby("currency")['sentiment_score'].transform(lambda x: x.rolling(3).mean())
    df['rolling_std_3'] = df.groupby("currency")['sentiment_score'].transform(lambda x: x.rolling(3).std())
    df['next_sentiment'] = df.groupby("currency")['sentiment_score'].shift(-1)
    df['label'] = (df['next_sentiment'] > df['sentiment_score']).astype(int)
    return df.dropna(subset=['rolling_mean_3', 'rolling_std_3', 'label'])

def train_model(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = df[['rolling_mean_3', 'rolling_std_3']]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict_next_sentiment(df, model):
    latest = df[['rolling_mean_3', 'rolling_std_3']].iloc[-1:]
    return model.predict(latest)[0]

import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample news data
news_data = [
    "Company reports strong quarterly earnings beating expectations",
    "New product launch expected to drive significant revenue growth",
    "Market volatility affecting all technology stocks negatively", 
    "Regulatory concerns causing uncertainty in the sector",
    "Positive analyst upgrades boost investor confidence"
]

def analyze_sentiment(text_list):
    sentiments = []
    for text in text_list:
        blob = TextBlob(text)
        sentiments.append({
            'text': text,
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'sentiment_label': 'Positive' if blob.sentiment.polarity > 0.1 
                                else 'Negative' if blob.sentiment.polarity < -0.1 
                                else 'Neutral'
        })
    return sentiments

# Analyze sentiment
sentiment_results = analyze_sentiment(news_data)
sentiment_df = pd.DataFrame(sentiment_results)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sentiment distribution
sentiment_counts = sentiment_df['sentiment_label'].value_counts()
axes[0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%')
axes[0].set_title('Sentiment Distribution')

# Polarity scores
axes[1].hist(sentiment_df['polarity'], bins=10, alpha=0.7, color='blue')
axes[1].set_title('Sentiment Polarity Distribution')
axes[1].set_xlabel('Polarity Score')

plt.tight_layout()
plt.show()

# Generate insights
avg_sentiment = sentiment_df['polarity'].mean()
positive_pct = (sentiment_df['sentiment_label'] == 'Positive').sum() / len(sentiment_df) * 100

print(sentiment_df)
print(f"Average Sentiment Score: {avg_sentiment:.3f}")
print(f"Positive News: {positive_pct:.1f}%")
print(f"Market Outlook: {'Bullish' if avg_sentiment > 0.2 else 'Bearish' if avg_sentiment < -0.2 else 'Neutral'}")
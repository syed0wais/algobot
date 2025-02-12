import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to fetch news articles using NewsAPI
def get_news_articles(stock_symbol, api_key):
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}&language=en'
    response = requests.get(url)
    data = response.json()

    # If API returns an error or no articles, return an empty list
    if data.get('status') != 'ok' or 'articles' not in data:
        return []

    articles = data['articles']
    return articles

# Function to analyze sentiment of a list of articles
def analyze_sentiment_of_articles(articles):
    sentiment_scores = []
    for article in articles:
        # Get the title and description of the article
        text = article['title'] + " " + article['description'] if article['description'] else article['title']
        sentiment = analyzer.polarity_scores(text)
        sentiment_scores.append(sentiment['compound'])  # Compound score is the overall sentiment

    return sentiment_scores

# Define function to determine overall sentiment (buy/sell/hold)
def generate_signal(sentiment_scores):
    if len(sentiment_scores) == 0:
        return "No data available", 0.0  # In case no articles are fetched

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    if avg_sentiment > 0.1:
        return "BUY", avg_sentiment
    elif avg_sentiment < -0.1:
        return "SELL", avg_sentiment
    else:
        return "HOLD", avg_sentiment

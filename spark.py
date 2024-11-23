import pandas as pd
import re
import emoji
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import sentiwordnet as swn
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# SQLAlchemy Configuration
DATABASE_URI = 'postgresql://postgres:password@localhost/CloudProject2'

# Define Base
Base = declarative_base()

# Create Engine
engine = create_engine(DATABASE_URI)
Session = sessionmaker(bind=engine)
session = Session()

# File details
input_file = "task5gold.csv"

# Define the emoji sentiment map
emoji_sentiment_map = {
    '❤️': 'positive',
    '☺️': 'positive',
    '👍': 'positive',
    '😞': 'negative',
    '😡': 'negative',
    '😢': 'negative',
    '😭': 'negative',
    '😩': 'negative',
}

# Common positive and negative phrases
negative_phrases = [
    "delayed", "delay", "lost luggage", "on hold", "waited too long", "still waiting", "still", "waiting", "help",
    "not happy", "horrible service", "worst experience", "no help", "angry", "frustrated", "reevaluate", "long wait", "fail", 
    "seriously doubt", "no response", "long enough", "long"
]

positive_phrases = [
    "thank you", "good service", "highly recommend", "best flight", "satisfied", "outstanding", "excellent"
]

# Text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http[s]?://\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)          # Remove special characters
    return text.strip().lower()

# Check for common phrases
def contains_phrases(text, phrases):
    text = text.lower()
    return any(phrase in text for phrase in phrases)

# Initialize the TreebankWordTokenizer
tokenizer = TreebankWordTokenizer()

# Calculate sentiment using multiple methods
def get_sentiment(text):
    if not text:
        return 'neutral'

    text_cleaned = clean_text(text)
    words = tokenizer.tokenize(text_cleaned)

    sentiment_score = 0
    count = 0

    # Check for emoji sentiment
    text_emojized = emoji.demojize(text)
    for symbol, sentiment in emoji_sentiment_map.items():
        if symbol in text_emojized:
            sentiment_score += 1 if sentiment == 'positive' else -1
            count += 1

    # Analyze using SentiWordNet
    for word in words:
        synsets = list(swn.senti_synsets(word))
        if synsets:
            synset = synsets[0]
            sentiment_score += synset.pos_score() - synset.neg_score()
            count += 1

    # Check common phrases
    if contains_phrases(text_cleaned, negative_phrases):
        sentiment_score -= 1
        count += 1
    if contains_phrases(text_cleaned, positive_phrases):
        sentiment_score += 1
        count += 1

    # Calculate average sentiment
    if count == 0:
        return 'neutral'

    avg_score = sentiment_score / count
    if avg_score > 0:
        return 'positive'
    elif avg_score < 0:
        return 'negative'
    else:
        return 'neutral'

# Load dataset
df = pd.read_csv(input_file)
print(f"DataFrame Loaded: {df.shape[0]} rows and {df.shape[1]} columns")

# Apply sentiment analysis
df['text_clean'] = df['text'].apply(clean_text)
df['sentiment_analysis'] = df['text'].apply(get_sentiment)

# Calculate accuracy
df['accuracy_score'] = (df['sentiment_analysis'] == df['airline_sentiment_gold']).astype(int)
accuracy = df['accuracy_score'].mean()
print(f"Overall Accuracy: {accuracy:.2%}")

# Save results to PostgreSQL using SQLAlchemy
df[['text', 'sentiment_analysis', 'airline_sentiment_gold', 'accuracy_score']].to_sql(
    'sentiment_results', engine, if_exists='replace', index=False
)

print("Results saved to PostgreSQL database.")

# Restaurant Review Sentiment Analysis (Complete NLP Demo)

import nltk
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#run only first time)
nltk.download('punkt')
nltk.download('stopwords')

# Sample restaurant reviews
reviews = [
    "The food was absolutely delicious and the service was great",
    "I hated the food, it was cold and tasteless",
    "Amazing experience, will visit again!",
    "Very bad service and the food was not good",
    "The ambience was nice but the food was average",
    "Totally loved it, highly recommended",
    "Worst restaurant ever, very disappointed",
    "Food was okay, nothing special"
]

# Function for preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    
    return filtered

# Analyze each review
print("\n--- Restaurant Review Sentiment Analysis ---\n")

for review in reviews:
    print("Original Review:", review)
    
    # Preprocessing
    processed_words = preprocess_text(review)
    print("Processed Words:", processed_words)
    
    # Sentiment Analysis
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    
    print("Sentiment Score:", polarity)
    
    # Classification
    if polarity > 0:
        print("Sentiment: Positive 😊")
    elif polarity < 0:
        print("Sentiment: Negative 😡")
    else:
        print("Sentiment: Neutral 😐")
    
    print("-" * 60)
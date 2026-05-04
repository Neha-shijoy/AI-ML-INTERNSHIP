import nltk
import string
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# First time only
nltk.download('punkt')
nltk.download('stopwords')

# Sample movie reviews
reviews = [
    "This movie was absolutely fantastic and mind-blowing",
    "Worst movie ever, I wasted my time",
    "It was an average film, nothing special",
    "I really loved the storyline and acting was great",
    "The movie was boring and too long",
    "Amazing visuals and outstanding performance",
    "Not good, very disappointing experience",
    "It was okay, some parts were nice"
]

# Stopwords setup (faster + better practice)
stop_words = set(stopwords.words('english'))

# Function for preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stop_words]
    
    return filtered

print("\n--- Movie Review Sentiment Analysis ---\n")

# Analyze each review
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
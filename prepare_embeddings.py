import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv('drugsComTrain_raw.csv')

# Preprocess function
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", '', str(text))
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Preprocess reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)

# Sentiment analysis (optional, but included for future improvement)
sid = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['review'].apply(lambda x: sid.polarity_scores(str(x))['compound'])

# Encode condition (optional, future scope)
le = LabelEncoder()
df['condition_encoded'] = le.fit_transform(df['condition'].astype(str))

# Load SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df['cleaned_review'].tolist(), show_progress_bar=True)

# Save processed data and embeddings
df.to_csv('processed_data.csv', index=False)
np.save('sentence_embeddings.npy', embeddings)

print("✅ Embeddings and processed data saved successfully!")

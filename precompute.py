import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load the dataset
df = pd.read_csv('Cleaned_Books.csv')

# Combine features
df['combined_features'] = df['title'] + " " + df['authors'] + ' ' + df['categories']

# Vectorize
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(df['combined_features'])

# Compute similarity
similarity = cosine_similarity(feature_vectors)

# Save vectorizer and similarity matrix
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

df.to_csv("Processed_Books.csv", index=False)
print("Preprocessing complete! Files saved.")

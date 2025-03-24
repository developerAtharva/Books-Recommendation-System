from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process

# Initialize Flask app
app = Flask(__name__)

# Sample data - Replace with your actual dataset
df = pd.read_csv('Cleaned_Books.csv')

# Combine features (title + description or other features you may have)
combined_features = df['title'] + " " + df['authors'] + ' ' + df['categories']

# Vectorize the combined features
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute similarity
similarity = cosine_similarity(feature_vectors)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_books = []
    
    if request.method == 'POST':
        input_book = request.form['book_name']
        
        # Extract close matches
        list_of_books_in_database = df['title'].tolist()
        close_match = process.extract(input_book, list_of_books_in_database)
        
        closest_match = close_match[0]
        index_of_book = closest_match[2]
        
        # Find similar books
        similar_books = list(enumerate(similarity[index_of_book]))
        sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
        
        # Get top 10 similar books
        # recommended_books = []
        for i, book in enumerate(sorted_similar_books):
            if i < 10:
                book_data = df.iloc[book[0]]
                recommended_books.append(f"{book_data['title']}, {book_data['authors']}, {book_data['categories']}")
        
    return render_template('index.html', recommended_books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)

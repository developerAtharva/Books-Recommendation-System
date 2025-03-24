from flask import Flask, render_template, request
import pandas as pd
import pickle
from rapidfuzz import process
import gdown
import os

file_id = "1NuKijoTUmushyHVae3EweWHoz9G6HrXi"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "similarity.pkl"

# Download the file again
gdown.download(url, output, quiet=False)

# Check if file downloaded correctly
if os.path.exists(output):
    print("Download successful!")
    print(f"File size: {os.path.getsize(output)} bytes")
else:
    print("Download failed again! Check your Google Drive link.")

app = Flask(__name__)

# Load precomputed data
df = pd.read_csv('Processed_Books.csv')

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    recommended_books = []
    
    if request.method == 'POST':
        input_book = request.form['book_name']

        # Find closest match
        list_of_books_in_database = df['title'].tolist()
        close_match = process.extract(input_book, list_of_books_in_database)

        if not close_match:
            return render_template('index.html', recommended_books=["Book not found."])

        closest_match = close_match[0]
        index_of_book = closest_match[2]

        # Get top 10 similar books
        similar_books = list(enumerate(similarity[index_of_book]))
        sorted_similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)

        for i, book in enumerate(sorted_similar_books):
            if i < 10:
                book_data = df.iloc[book[0]]
                recommended_books.append(f"{book_data['title']}, {book_data['authors']}, {book_data['categories']}")

    return render_template('index.html', recommended_books=recommended_books)

if __name__ == '__main__':
    app.run(debug=True)

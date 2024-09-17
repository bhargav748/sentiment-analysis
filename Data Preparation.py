import nltk
from nltk.corpus import movie_reviews
import pandas as pd
from sklearn.model_selection import train_test_split

# Download NLTK data
nltk.download('movie_reviews')

# Load movie reviews dataset
def load_data():
    categories = ['neg', 'pos']
    data = []
    for category in categories:
        for fileid in movie_reviews.fileids(category):
            words = movie_reviews.words(fileid)
            text = ' '.join(words)
            data.append((text, category))
    return pd.DataFrame(data, columns=['text', 'label'])

df = load_data()
df['label'] = df['label'].map({'pos': 1, 'neg': 0})  # Encode labels

# Split the data into training and test sets
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

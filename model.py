import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the dataset
df = pd.read_csv('netflix_titles.csv')

# Handle missing values
df['director'] = df['director'].fillna('')
df['cast'] = df['cast'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')
df['description'] = df['description'].fillna('')

# Feature engineering
df['genres'] = df['listed_in'].apply(lambda x: ' '.join(x.replace(' ', '').replace(',', ' ').split()))
def get_top_cast(x):
    if isinstance(x, str):
        actors = x.split(', ')
        return ' '.join(actors[:3]) if len(actors) >= 3 else ' '.join(actors)
    return ''
df['cast_top'] = df['cast'].apply(get_top_cast)
df['director'] = df['director'].apply(lambda x: x.replace(', ', ' '))
df['soup'] = df['director'] + ' ' + df['cast_top'] + ' ' + df['genres'] + ' ' + df['description']

# Vectorize with fewer features
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)  # Reduced for memory
features = vectorizer.fit_transform(df['soup'])

# Save only df, vectorizer, and features
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump((df, vectorizer, features), f)

print("Model trained and saved! Size reduced.")
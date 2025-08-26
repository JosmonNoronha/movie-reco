import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle  # For saving the model

# Step 3.1: Load the dataset
df = pd.read_csv('netflix_titles.csv')

# Step 3.2: Handle missing values (fill with empty string for text features)
df['director'] = df['director'].fillna('')
df['cast'] = df['cast'].fillna('')
df['listed_in'] = df['listed_in'].fillna('')
df['description'] = df['description'].fillna('')

# Step 3.3: Feature engineering
# - Genres: Convert 'listed_in' to space-separated (e.g., "Dramas, International Movies" -> "Dramas International Movies")
df['genres'] = df['listed_in'].apply(lambda x: ' '.join(x.replace(' ', '').replace(',', ' ').split()))

# - Cast: Take top 3 actors
def get_top_cast(x):
    if isinstance(x, str):
        actors = x.split(', ')
        return ' '.join(actors[:3]) if len(actors) >= 3 else ' '.join(actors)
    return ''
df['cast_top'] = df['cast'].apply(get_top_cast)

# - Director: Keep as is (space-separated if multiple)
df['director'] = df['director'].apply(lambda x: x.replace(', ', ' '))  # Treat multiple directors as words

# - Combine into a 'soup' for vectorization (weight director/cast more by repeating if needed, but keep simple)
df['soup'] = df['director'] + ' ' + df['cast_top'] + ' ' + df['genres'] + ' ' + df['description']

# Step 3.4: Vectorize the soup using TF-IDF (ignores common words)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Limit features to avoid huge matrix
features = vectorizer.fit_transform(df['soup'])

# Step 3.5: Compute cosine similarity matrix (item-to-item similarity)
cosine_sim = cosine_similarity(features)

# Save the model for later use (df, vectorizer, cosine_sim, features)
with open('recommender_model.pkl', 'wb') as f:
    pickle.dump((df, vectorizer, cosine_sim, features), f)

print("Model trained and saved!")

# Function to get recommendations
def get_recommendations(user_titles, top_n=10):
    # Load saved model if needed (for API use)
    # with open('recommender_model.pkl', 'rb') as f:
    #     df, vectorizer, cosine_sim, features = pickle.load(f)
    
    # Find indices of user titles (case-insensitive match)
    user_indices = []
    for title in user_titles:
        match = df[df['title'].str.lower() == title.lower()]
        if not match.empty:
            user_indices.append(match.index[0])
    
    if not user_indices:
        # If no matches, return popular items (fallback)
        popular = df.sort_values('release_year', ascending=False).head(top_n)['title'].tolist()  # Or use ratings if available
        return popular
    
    # Create user profile vector: average of liked items' vectors
    user_vector = np.mean(features[user_indices].toarray(), axis=0).reshape(1, -1)
    
    # Compute similarity to all items
    sim_scores = cosine_similarity(user_vector, features)[0]
    
    # Sort by similarity (descending), exclude user's own titles
    sim_indices = np.argsort(-sim_scores)
    recommendations = []
    for idx in sim_indices:
        title = df.iloc[idx]['title']
        if title.lower() not in [t.lower() for t in user_titles]:
            recommendations.append({
                'title': title,
                'type': df.iloc[idx]['type'],
                'genres': df.iloc[idx]['listed_in'],
                'description': df.iloc[idx]['description'][:100] + '...'  # Truncate for brevity
            })
        if len(recommendations) == top_n:
            break
    
    return recommendations

# Example usage
user_favs = ['Stranger Things', 'The Crown']  # Replace with actual titles from dataset
recs = get_recommendations(user_favs)
print(recs)
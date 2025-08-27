from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Download the model from Google Drive
file_id = '1qUf0S8NpiuQhmyIOt3l8rD7vmSXETomz'  
url = f'https://drive.google.com/uc?id={file_id}'
gdown.download(url, 'recommender_model.pkl', quiet=False)

# Load the model (df, vectorizer, features only)
with open('recommender_model.pkl', 'rb') as f:
    df, vectorizer, features = pickle.load(f)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_titles = data.get('titles', [])
    top_n = data.get('top_n', 10)
    
    user_indices = []
    for title in user_titles:
        match = df[df['title'].str.lower() == title.lower()]
        if not match.empty:
            user_indices.append(match.index[0])
    
    if not user_indices:
        popular = df.sort_values('release_year', ascending=False).head(top_n)['title'].tolist()
        return jsonify({'recommendations': popular})
    
    # Compute user profile vector
    user_vector = np.mean(features[user_indices].toarray(), axis=0).reshape(1, -1)
    
    # Compute similarities on-demand
    sim_scores = cosine_similarity(user_vector, features)[0]
    
    # Sort and filter recommendations
    sim_indices = np.argsort(-sim_scores)
    recommendations = []
    for idx in sim_indices:
        title = df.iloc[idx]['title']
        if title.lower() not in [t.lower() for t in user_titles]:
            recommendations.append({
                'title': title,
                'type': df.iloc[idx]['type'],
                'genres': df.iloc[idx]['listed_in'],
                'description': df.iloc[idx]['description'][:100] + '...'
            })
        if len(recommendations) == top_n:
            break
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port, debug=False)  # Listen on 0.0.0.0 for Render
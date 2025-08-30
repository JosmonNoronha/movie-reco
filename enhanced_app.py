# hybrid_app.py - Netflix + MovieLens API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import gdown

app = Flask(__name__)
CORS(app)

def load_model():
    try:
        file_id = '13yjX735n0Ddwd7r3YselhP71Z-XVo4Pt'  # Update with new file ID
        url = f'https://drive.google.com/uc?id={file_id}'
        
        if not os.path.exists('balanced_recommender_model.pkl'):
            print("Downloading hybrid model...")
            gdown.download(url, 'balanced_recommender_model.pkl', quiet=False)
        
        with open('balanced_recommender_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            return model_data['df'], model_data['vectorizer'], model_data['features']
        else:
            # Old format compatibility
            return model_data
    except:
        with open('balanced_recommender_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            if isinstance(model_data, dict):
                return model_data['df'], model_data['vectorizer'], model_data['features']
            else:
                return model_data

df, vectorizer, features = load_model()

def compute_hybrid_similarity(user_vector, features, df, user_indices):
    """Enhanced similarity using hybrid data"""
    
    content_sim = cosine_similarity(user_vector, features)[0]
    
    # Analyze user preferences
    user_profile = {
        'primary_genres': [],
        'themes': [],
        'avg_rating': 0,
        'content_type': 'mixed',
        'mood_preference': [],
        'prefers_verified': False
    }
    
    verified_count = 0
    for idx in user_indices:
        row = df.iloc[idx]
        
        # Count verified titles user likes
        if row.get('data_source') == 'hybrid':
            verified_count += 1
        
        # Extract preferences
        genres = str(row['listed_in']).lower().split(',')
        user_profile['primary_genres'].extend([g.strip() for g in genres])
        
        themes = str(row.get('content_themes', '')).split()
        user_profile['themes'].extend(themes)
        
        user_profile['avg_rating'] += row.get('avg_rating', 0)
        
        if row['type'] == 'TV Show':
            user_profile['content_type'] = 'series'
        elif user_profile['content_type'] != 'series':
            user_profile['content_type'] = 'movies'
        
        moods = str(row.get('content_mood', '')).split()
        user_profile['mood_preference'].extend(moods)
    
    # User prefers verified data if >50% of their choices have real ratings
    user_profile['prefers_verified'] = verified_count > len(user_indices) * 0.5
    
    # Normalize preferences
    user_profile['avg_rating'] /= len(user_indices)
    user_profile['primary_genres'] = list(set(user_profile['primary_genres']))[:5]
    user_profile['themes'] = list(set(user_profile['themes']))[:8]
    user_profile['mood_preference'] = list(set(user_profile['mood_preference']))[:5]
    
    # Calculate enhanced bonuses
    bonuses = []
    
    for idx, row in df.iterrows():
        total_bonus = 0
        
        # Verified data bonus (for users who like verified content)
        if user_profile['prefers_verified'] and row.get('data_source') == 'hybrid':
            if row.get('ml_rating_count', 0) >= 50:
                total_bonus += 0.15  # Strong bonus for well-rated verified content
            elif row.get('ml_rating_count', 0) >= 20:
                total_bonus += 0.10
            else:
                total_bonus += 0.05
        
        # Genre matching bonus (25% max)
        row_genres = str(row['listed_in']).lower()
        genre_matches = sum(0.03 for user_genre in user_profile['primary_genres'] 
                          if user_genre in row_genres)
        total_bonus += min(0.25, genre_matches)
        
        # Theme matching bonus (20% max)
        row_themes = str(row.get('content_themes', '')).split()
        theme_matches = sum(0.02 for user_theme in user_profile['themes'] 
                          if user_theme in row_themes)
        total_bonus += min(0.20, theme_matches)
        
        # Quality preference bonus (15% max)
        if user_profile['avg_rating'] > 3.5:
            item_rating = row.get('avg_rating', 0)
            # Enhanced for verified data
            if row.get('data_source') == 'hybrid' and pd.notna(row.get('ml_rating_std')):
                if item_rating >= 4.2 and row.get('ml_rating_std', 1.0) < 0.8:
                    total_bonus += 0.18  # Consistent high quality
                elif item_rating >= 3.8:
                    total_bonus += 0.12
            else:
                if item_rating >= 4.2:
                    total_bonus += 0.15
                elif item_rating >= 3.8:
                    total_bonus += 0.10
                elif item_rating >= 3.5:
                    total_bonus += 0.05
        
        # Content type preference (10% max)
        if user_profile['content_type'] == 'series' and row['type'] == 'TV Show':
            total_bonus += 0.10
        elif user_profile['content_type'] == 'movies' and row['type'] == 'Movie':
            total_bonus += 0.10
        
        # Mood matching bonus (10% max)
        row_moods = str(row.get('content_mood', '')).split()
        mood_matches = sum(0.02 for user_mood in user_profile['mood_preference'] 
                         if user_mood in row_moods)
        total_bonus += min(0.10, mood_matches)
        
        bonuses.append(total_bonus)
    
    # Combine content similarity with enhanced bonuses
    final_scores = content_sim + np.array(bonuses)
    
    return final_scores

def filter_universal_recommendations(df, sim_scores, user_indices, user_titles, top_n):
    """Enhanced filtering with hybrid data prioritization"""
    
    sim_indices = np.argsort(-sim_scores)
    recommendations = []
    seen_titles = {title.lower().strip() for title in user_titles}
    
    # Diversity tracking
    genre_count = {}
    mood_count = {}
    
    # Dynamic quality threshold based on data source
    min_rating = 3.2
    
    for idx in sim_indices:
        if len(recommendations) >= top_n:
            break
            
        row = df.iloc[idx]
        title = row['title']
        
        if not title or title.lower().strip() in seen_titles:
            continue
        
        # Enhanced quality filter
        item_rating = row.get('avg_rating', 0)
        if row.get('data_source') == 'hybrid':
            # Higher standards for verified data
            if item_rating < 3.0 or (row.get('ml_rating_count', 0) < 5):
                continue
        else:
            # Standard threshold for Netflix-only
            if item_rating < min_rating:
                continue
        
        # Similarity threshold
        if sim_scores[idx] < 0.05:
            continue
        
        # Diversity controls
        primary_genre = str(row['listed_in']).split(',')[0].strip() if row['listed_in'] else 'Unknown'
        if genre_count.get(primary_genre, 0) >= max(1, int(top_n * 0.4)):
            continue
        
        primary_mood = str(row.get('content_mood', '')).split()[0] if row.get('content_mood') else 'neutral'
        if mood_count.get(primary_mood, 0) >= max(2, int(top_n * 0.5)):
            continue
        
        # Update tracking
        genre_count[primary_genre] = genre_count.get(primary_genre, 0) + 1
        mood_count[primary_mood] = mood_count.get(primary_mood, 0) + 1
        seen_titles.add(title.lower().strip())
        
        rec = {
            'title': title,
            'type': row['type'],
            'genres': row['listed_in'],
            'description': str(row['description'])[:200] + '...' if pd.notna(row['description']) else 'No description available.',
            'release_year': str(int(row['release_year'])) if pd.notna(row['release_year']) else 'Unknown',
            'director': str(row['director']) if pd.notna(row['director']) else 'Unknown',
            'cast': str(row['cast']) if pd.notna(row['cast']) else 'Unknown',
            'similarity_score': float(sim_scores[idx]),
            'avg_rating': float(row.get('avg_rating', 0)),
            'rating_count': int(row.get('rating_count', 0)),
            'source': row.get('source', 'Netflix')
        }
        
        recommendations.append(rec)
    
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_titles = data.get('titles', [])
        top_n = min(data.get('top_n', 10), 15)
        
        if not user_titles:
            return jsonify({
                'recommendations': [],
                'message': 'Please provide at least one movie/show title.'
            })
        
        # Enhanced title matching
        user_indices = []
        found_titles = []
        
        for title in user_titles:
            title_clean = title.lower().strip()
            
            # Exact match
            exact = df[df['title'].str.lower() == title_clean]
            if not exact.empty:
                # Prefer verified data if available
                if any(exact['data_source'] == 'hybrid'):
                    best_match = exact[exact['data_source'] == 'hybrid'].iloc[0]
                else:
                    best_match = exact.iloc[0]
                user_indices.append(best_match.name)
                found_titles.append(best_match['title'])
                continue
            
            # Partial match
            partial = df[df['title'].str.lower().str.contains(title_clean, na=False)]
            if not partial.empty:
                # Prioritize: hybrid data > higher rating
                if any(partial['data_source'] == 'hybrid'):
                    hybrid_matches = partial[partial['data_source'] == 'hybrid']
                    best_idx = hybrid_matches.loc[hybrid_matches['avg_rating'].idxmax()].name
                else:
                    best_idx = partial.loc[partial['avg_rating'].idxmax()].name
                user_indices.append(best_idx)
                found_titles.append(df.loc[best_idx, 'title'])
                continue
        
        if not user_indices:
            # Enhanced fallback with hybrid data preference
            hybrid_popular = df[df['data_source'] == 'hybrid'].nlargest(top_n//2, 'avg_rating')
            netflix_popular = df[df['data_source'] == 'netflix_only'].nlargest(top_n//2, 'avg_rating')
            popular = pd.concat([hybrid_popular, netflix_popular]).head(top_n)
            
            fallback = []
            for _, row in popular.iterrows():
                fallback.append({
                    'title': row['title'],
                    'type': row['type'],
                    'genres': row['listed_in'],
                    'description': str(row['description'])[:200] + '...',
                    'release_year': str,
                    'release_year': str(int(row['release_year'])) if pd.notna(row['release_year']) else 'Unknown',
                    'director': str(row['director']),
                    'cast': str(row['cast']),
                    'avg_rating': float(row['avg_rating']),
                    'source': row['source']
                })
            
            return jsonify({
                'recommendations': fallback,
                'message': 'No exact matches found. Here are top-rated suggestions.',
                'found_titles': []
            })
        
        # Generate hybrid recommendations
        user_vectors = features[user_indices].toarray()
        user_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
        
        sim_scores = compute_hybrid_similarity(user_vector, features, df, user_indices)
        recommendations = filter_universal_recommendations(df, sim_scores, user_indices, user_titles, top_n)
        
        # Extract user's main genre for context
        user_genres = []
        for idx in user_indices:
            genres = str(df.iloc[idx]['listed_in']).split(',')
            user_genres.extend([g.strip() for g in genres])
        
        main_genre = max(set(user_genres), key=user_genres.count) if user_genres else 'Mixed'
        
        # Check how many recommendations have verified ratings
        verified_recs = sum(1 for rec in recommendations if 'MovieLens' in rec['source'])
        
        return jsonify({
            'recommendations': recommendations,
            'found_titles': found_titles,
            'message': f'Found {len(found_titles)} titles. Generated {len(recommendations)} recommendations ({verified_recs} with verified ratings).',
            'user_main_genre': main_genre,
            'recommendation_quality': 'hybrid'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recommendations': [],
            'message': 'Error generating recommendations.'
        }), 500

def compute_universal_similarity(user_vector, features, df, user_indices):
    """Maintain backward compatibility"""
    return compute_hybrid_similarity(user_vector, features, df, user_indices)

@app.route('/health', methods=['GET'])
def health():
    hybrid_count = len(df[df['data_source'] == 'hybrid']) if 'data_source' in df.columns else 0
    
    return jsonify({
        'status': 'healthy',
        'dataset_size': len(df),
        'model_type': 'hybrid_netflix_movielens',
        'avg_rating_range': f"{df['avg_rating'].min():.1f} - {df['avg_rating'].max():.1f}",
        'verified_titles': hybrid_count
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
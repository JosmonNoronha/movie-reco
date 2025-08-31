# hybrid_app.py - Netflix + MovieLens API (FIXED)
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
    
    # Analyze user preferences from their selected titles
    user_profile = {
        'primary_genres': [],
        'themes': [],
        'avg_rating': 0,
        'content_type': 'mixed',
        'mood_preference': [],
        'prefers_verified': False,
        'prefers_movielens': False,
        'source_preference': 'mixed'
    }
    
    verified_count = 0
    movielens_count = 0
    
    for idx in user_indices:
        row = df.iloc[idx]
        
        # Count data sources user likes
        if row.get('data_source') == 'hybrid':
            verified_count += 1
        elif row.get('data_source') == 'movielens_only':
            movielens_count += 1
        
        # Extract preferences
        genres = str(row['listed_in']).lower().split(',')
        ml_genres = str(row.get('ml_genres', '')).lower().split('|')
        all_genres = genres + ml_genres
        user_profile['primary_genres'].extend([g.strip() for g in all_genres if g.strip()])
        
        themes = str(row.get('content_themes', '')).split()
        user_profile['themes'].extend(themes)
        
        user_profile['avg_rating'] += row.get('avg_rating', 0)
        
        if row['type'] == 'TV Show':
            user_profile['content_type'] = 'series'
        elif user_profile['content_type'] != 'series':
            user_profile['content_type'] = 'movies'
        
        moods = str(row.get('content_mood', '')).split()
        user_profile['mood_preference'].extend(moods)
    
    # Determine user's data source preferences
    total_selections = len(user_indices)
    user_profile['prefers_verified'] = verified_count > total_selections * 0.3
    user_profile['prefers_movielens'] = movielens_count > total_selections * 0.3
    
    if verified_count + movielens_count > total_selections * 0.6:
        user_profile['source_preference'] = 'quality_data'
    elif movielens_count > verified_count:
        user_profile['source_preference'] = 'movielens'
    elif verified_count > 0:
        user_profile['source_preference'] = 'hybrid'
    else:
        user_profile['source_preference'] = 'netflix'
    
    # Normalize preferences
    user_profile['avg_rating'] /= len(user_indices)
    user_profile['primary_genres'] = list(set(user_profile['primary_genres']))[:8]
    user_profile['themes'] = list(set(user_profile['themes']))[:10]
    user_profile['mood_preference'] = list(set(user_profile['mood_preference']))[:5]
    
    # Calculate enhanced bonuses
    bonuses = []
    
    for idx, row in df.iterrows():
        total_bonus = 0
        
        # DATA SOURCE BONUS (25% max)
        if user_profile['source_preference'] == 'quality_data':
            if row.get('data_source') == 'movielens_only' and row.get('ml_rating_count', 0) >= 50:
                total_bonus += 0.25  # Strong preference for well-rated MovieLens titles
            elif row.get('data_source') == 'hybrid' and row.get('ml_rating_count', 0) >= 30:
                total_bonus += 0.20  # Good hybrid data
            elif row.get('data_source') == 'hybrid':
                total_bonus += 0.15
        elif user_profile['source_preference'] == 'movielens':
            if row.get('data_source') == 'movielens_only':
                total_bonus += 0.20
            elif row.get('data_source') == 'hybrid':
                total_bonus += 0.10
        elif user_profile['source_preference'] == 'hybrid':
            if row.get('data_source') == 'hybrid':
                total_bonus += 0.15
            elif row.get('data_source') == 'movielens_only':
                total_bonus += 0.10
        
        # GENRE MATCHING BONUS (20% max)
        row_genres = (str(row['listed_in']) + '|' + str(row.get('ml_genres', ''))).lower()
        genre_matches = sum(0.025 for user_genre in user_profile['primary_genres'] 
                          if user_genre in row_genres)
        total_bonus += min(0.20, genre_matches)
        
        # THEME MATCHING BONUS (15% max)
        row_themes = str(row.get('content_themes', '')).split()
        theme_matches = sum(0.015 for user_theme in user_profile['themes'] 
                          if user_theme in row_themes)
        total_bonus += min(0.15, theme_matches)
        
        # QUALITY PREFERENCE BONUS (15% max)
        if user_profile['avg_rating'] > 3.5:
            item_rating = row.get('avg_rating', 0)
            # Enhanced for verified data
            if row.get('data_source') in ['hybrid', 'movielens_only'] and pd.notna(row.get('ml_rating_std')):
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
        
        # CONTENT TYPE PREFERENCE (10% max)
        if user_profile['content_type'] == 'series' and row['type'] == 'TV Show':
            total_bonus += 0.10
        elif user_profile['content_type'] == 'movies' and row['type'] == 'Movie':
            total_bonus += 0.10
        
        # MOOD MATCHING BONUS (10% max)
        row_moods = str(row.get('content_mood', '')).split()
        mood_matches = sum(0.02 for user_mood in user_profile['mood_preference'] 
                         if user_mood in row_moods)
        total_bonus += min(0.10, mood_matches)
        
        # RATING COUNT BONUS (5% max) - Prefer well-rated content
        rating_count = row.get('ml_rating_count', row.get('rating_count', 0))
        if rating_count >= 100:
            total_bonus += 0.05
        elif rating_count >= 50:
            total_bonus += 0.03
        elif rating_count >= 20:
            total_bonus += 0.01
        
        bonuses.append(total_bonus)
    
    # Combine content similarity with enhanced bonuses
    final_scores = content_sim + np.array(bonuses)
    
    return final_scores

def filter_universal_recommendations(df, sim_scores, user_indices, user_titles, top_n):
    """Enhanced filtering with true hybrid data support"""
    
    sim_indices = np.argsort(-sim_scores)
    recommendations = []
    seen_titles = {title.lower().strip() for title in user_titles}
    
    # Diversity tracking
    genre_count = {}
    mood_count = {}
    source_count = {'Netflix': 0, 'Netflix + MovieLens': 0, 'MovieLens': 0}
    
    # Dynamic quality thresholds based on data source
    quality_thresholds = {
        'netflix_only': 3.2,
        'hybrid': 3.0,
        'movielens_only': 3.0
    }
    
    for idx in sim_indices:
        if len(recommendations) >= top_n:
            break
            
        row = df.iloc[idx]
        title = row['title']
        
        if not title or title.lower().strip() in seen_titles:
            continue
        
        # Enhanced quality filter based on data source
        item_rating = row.get('avg_rating', 0)
        data_source = row.get('data_source', 'netflix_only')
        min_rating = quality_thresholds.get(data_source, 3.2)
        
        if data_source in ['hybrid', 'movielens_only']:
            # Higher standards for verified data but more lenient entry
            if item_rating < min_rating or (row.get('ml_rating_count', 0) < 5):
                continue
        else:
            # Standard threshold for Netflix-only
            if item_rating < min_rating:
                continue
        
        # Similarity threshold
        if sim_scores[idx] < 0.03:  # Slightly lowered for more variety
            continue
        
        # Enhanced diversity controls
        primary_genre = str(row['listed_in']).split(',')[0].strip() if row['listed_in'] else \
                       str(row.get('ml_genres', '')).split('|')[0].strip() if row.get('ml_genres') else 'Unknown'
        
        max_genre_count = max(2, int(top_n * 0.4))
        if genre_count.get(primary_genre, 0) >= max_genre_count:
            continue
        
        # Source diversity - ensure mix of sources
        source = row.get('source', 'Netflix')
        max_source_count = max(3, int(top_n * 0.7))  # Allow up to 70% from one source
        if source_count.get(source, 0) >= max_source_count:
            continue
        
        primary_mood = str(row.get('content_mood', '')).split()[0] if row.get('content_mood') else 'neutral'
        if mood_count.get(primary_mood, 0) >= max(2, int(top_n * 0.5)):
            continue
        
        # Update tracking
        genre_count[primary_genre] = genre_count.get(primary_genre, 0) + 1
        mood_count[primary_mood] = mood_count.get(primary_mood, 0) + 1
        source_count[source] = source_count.get(source, 0) + 1
        seen_titles.add(title.lower().strip())
        
        # Enhanced recommendation object with safe NaN handling
        ml_count = row.get('ml_rating_count', np.nan)
        regular_count = row.get('rating_count', np.nan)
        
        if pd.notna(ml_count):
            rating_count = int(ml_count)
        elif pd.notna(regular_count):
            rating_count = int(regular_count)
        else:
            rating_count = 0
        
        rec = {
            'title': title,
            'type': row['type'],
            'genres': row['listed_in'] if pd.notna(row['listed_in']) else row.get('ml_genres', '').replace('|', ', '),
            'description': str(row['description'])[:200] + '...' if pd.notna(row['description']) and len(str(row['description'])) > 200 else str(row['description']) if pd.notna(row['description']) else 'No description available.',
            'release_year': str(int(row['release_year'])) if pd.notna(row['release_year']) else 'Unknown',
            'director': str(row['director']) if pd.notna(row['director']) and row['director'] else 'Unknown',
            'cast': str(row['cast']) if pd.notna(row['cast']) and row['cast'] else 'Unknown',
            'similarity_score': float(sim_scores[idx]),
            'avg_rating': float(row.get('avg_rating', 0)),
            'rating_count': rating_count,
            'source': row.get('source', 'Netflix'),
            'data_quality': 'verified' if row.get('data_source') in ['hybrid', 'movielens_only'] else 'estimated'
        }
        
        # Add MovieLens-specific data if available
        if pd.notna(row.get('ml_rating_std')):
            rec['rating_consistency'] = f"±{float(row['ml_rating_std']):.1f}"
        
        if row.get('ml_tags') and pd.notna(row.get('ml_tags')):
            rec['tags'] = str(row['ml_tags'])[:100]  # Limit tag length
        
        recommendations.append(rec)
    
    return recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.json
        user_titles = data.get('titles', [])
        top_n = min(data.get('top_n', 10), 20)  # Increased max to 20
        
        if not user_titles:
            return jsonify({
                'recommendations': [],
                'message': 'Please provide at least one movie/show title.'
            })
        
        # Enhanced title matching across all sources
        user_indices = []
        found_titles = []
        
        for title in user_titles:
            title_clean = title.lower().strip()
            
            # Exact match across all sources
            exact = df[df['title'].str.lower() == title_clean]
            if not exact.empty:
                # Prioritize: MovieLens-only > Hybrid > Netflix-only (for data quality)
                if any(exact['data_source'] == 'movielens_only'):
                    best_match = exact[exact['data_source'] == 'movielens_only'].iloc[0]
                elif any(exact['data_source'] == 'hybrid'):
                    best_match = exact[exact['data_source'] == 'hybrid'].iloc[0]
                else:
                    best_match = exact.iloc[0]
                user_indices.append(best_match.name)
                found_titles.append(best_match['title'])
                continue
            
            # Partial match across all sources
            partial = df[df['title'].str.lower().str.contains(title_clean, na=False)]
            if not partial.empty:
                # Prioritize by data quality and rating
                if any(partial['data_source'] == 'movielens_only'):
                    ml_matches = partial[partial['data_source'] == 'movielens_only']
                    best_idx = ml_matches.loc[ml_matches['avg_rating'].idxmax()].name
                elif any(partial['data_source'] == 'hybrid'):
                    hybrid_matches = partial[partial['data_source'] == 'hybrid']
                    best_idx = hybrid_matches.loc[hybrid_matches['avg_rating'].idxmax()].name
                else:
                    best_idx = partial.loc[partial['avg_rating'].idxmax()].name
                user_indices.append(best_idx)
                found_titles.append(df.loc[best_idx, 'title'])
                continue
        
        if not user_indices:
            # Enhanced fallback with true source diversity
            print("No matches found, using diverse fallback...")
            
            # Get top titles from each source
            netflix_top = df[df['data_source'] == 'netflix_only'].nlargest(top_n//3, 'avg_rating')
            hybrid_top = df[df['data_source'] == 'hybrid'].nlargest(top_n//3, 'avg_rating') 
            ml_top = df[df['data_source'] == 'movielens_only'].nlargest(top_n//3, 'avg_rating')
            
            popular = pd.concat([ml_top, hybrid_top, netflix_top]).head(top_n)
            
            fallback = []
            for _, row in popular.iterrows():
                # Safe integer conversion for rating_count
                ml_count = row.get('ml_rating_count', np.nan)
                regular_count = row.get('rating_count', np.nan)
                
                if pd.notna(ml_count):
                    rating_count = int(ml_count)
                elif pd.notna(regular_count):
                    rating_count = int(regular_count)
                else:
                    rating_count = 0
                
                fallback.append({
                    'title': row['title'],
                    'type': row['type'],
                    'genres': row['listed_in'] if pd.notna(row['listed_in']) else row.get('ml_genres', '').replace('|', ', '),
                    'description': str(row['description'])[:200] + '...' if pd.notna(row['description']) and len(str(row['description'])) > 200 else str(row['description']) if pd.notna(row['description']) else 'No description available.',
                    'release_year': str(int(row['release_year'])) if pd.notna(row['release_year']) else 'Unknown',
                    'director': str(row['director']) if pd.notna(row['director']) and row['director'] else 'Unknown',
                    'cast': str(row['cast']) if pd.notna(row['cast']) and row['cast'] else 'Unknown',
                    'avg_rating': float(row.get('avg_rating', 0)),
                    'rating_count': rating_count,
                    'source': row.get('source', 'Netflix'),
                    'data_quality': 'verified' if row.get('data_source') in ['hybrid', 'movielens_only'] else 'estimated'
                })
            
            return jsonify({
                'recommendations': fallback,
                'message': 'No exact matches found. Here are top-rated suggestions from all sources.',
                'found_titles': [],
                'sources_used': ['Netflix', 'MovieLens', 'Hybrid'],
                'recommendation_quality': 'diverse_fallback'
            })
        
        # Generate hybrid recommendations
        user_vectors = features[user_indices].toarray()
        user_vector = np.mean(user_vectors, axis=0).reshape(1, -1)
        
        sim_scores = compute_hybrid_similarity(user_vector, features, df, user_indices)
        recommendations = filter_universal_recommendations(df, sim_scores, user_indices, user_titles, top_n)
        
        # Extract user's main preferences for context
        user_genres = []
        user_sources = []
        for idx in user_indices:
            row = df.iloc[idx]
            genres = str(row['listed_in']).split(',') + str(row.get('ml_genres', '')).split('|')
            user_genres.extend([g.strip() for g in genres if g.strip()])
            user_sources.append(row.get('source', 'Netflix'))
        
        main_genre = max(set(user_genres), key=user_genres.count) if user_genres else 'Mixed'
        main_source = max(set(user_sources), key=user_sources.count) if user_sources else 'Netflix'
        
        # Analyze recommendation sources
        rec_sources = {}
        for rec in recommendations:
            source = rec['source']
            rec_sources[source] = rec_sources.get(source, 0) + 1
        
        verified_recs = sum(1 for rec in recommendations if rec['data_quality'] == 'verified')
        
        return jsonify({
            'recommendations': recommendations,
            'found_titles': found_titles,
            'message': f'Found {len(found_titles)} titles. Generated {len(recommendations)} recommendations ({verified_recs} with verified ratings).',
            'user_main_genre': main_genre,
            'user_main_source': main_source,
            'recommendation_sources': rec_sources,
            'recommendation_quality': 'hybrid_enhanced'
        })
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
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
    # Enhanced health check with source breakdown
    source_stats = {}
    if 'data_source' in df.columns:
        for source in df['data_source'].unique():
            count = len(df[df['data_source'] == source])
            avg_rating = df[df['data_source'] == source]['avg_rating'].mean()
            source_stats[source] = {
                'count': count,
                'avg_rating': round(avg_rating, 2)
            }
    
    return jsonify({
        'status': 'healthy',
        'total_titles': len(df),
        'model_type': 'true_hybrid_netflix_movielens',
        'avg_rating_range': f"{df['avg_rating'].min():.1f} - {df['avg_rating'].max():.1f}",
        'source_breakdown': source_stats,
        'sources_available': list(df.get('source', pd.Series(['Netflix'])).unique()),
        'features_count': features.shape[1] if 'features' in globals() else 'unknown'
    })

@app.route('/search', methods=['POST'])
def search_titles():
    """New endpoint to search across all sources"""
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        limit = min(data.get('limit', 20), 50)
        
        if not query:
            return jsonify({
                'results': [],
                'message': 'Please provide a search query.'
            })
        
        # Search across titles, descriptions, genres, and tags
        search_mask = (
            df['title'].str.lower().str.contains(query, na=False) |
            df['description'].str.lower().str.contains(query, na=False) |
            df['listed_in'].str.lower().str.contains(query, na=False) |
            df.get('ml_genres', pd.Series([''] * len(df))).str.lower().str.contains(query, na=False) |
            df.get('ml_tags', pd.Series([''] * len(df))).str.lower().str.contains(query, na=False)
        )
        
        search_results = df[search_mask].nlargest(limit, 'avg_rating')
        
        results = []
        for _, row in search_results.iterrows():
            results.append({
                'title': row['title'],
                'type': row['type'],
                'genres': row['listed_in'] if pd.notna(row['listed_in']) else row.get('ml_genres', '').replace('|', ', '),
                'description': str(row['description'])[:150] + '...' if pd.notna(row['description']) and len(str(row['description'])) > 150 else str(row['description']) if pd.notna(row['description']) else 'No description available.',
                'release_year': str(int(row['release_year'])) if pd.notna(row['release_year']) else 'Unknown',
                'avg_rating': float(row.get('avg_rating', 0)),
                'rating_count': int(row.get('ml_rating_count', row.get('rating_count', 0))),
                'source': row.get('source', 'Netflix'),
                'data_quality': 'verified' if row.get('data_source') in ['hybrid', 'movielens_only'] else 'estimated'
            })
        
        # Source breakdown for search results
        source_breakdown = {}
        for result in results:
            source = result['source']
            source_breakdown[source] = source_breakdown.get(source, 0) + 1
        
        return jsonify({
            'results': results,
            'total_found': len(search_results),
            'query': query,
            'source_breakdown': source_breakdown,
            'message': f'Found {len(results)} titles matching "{query}"'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'results': [],
            'message': 'Error searching titles.'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
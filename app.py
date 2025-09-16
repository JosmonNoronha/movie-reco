# database_hybrid_app.py - SQLite Database Version (SUPER FAST)
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
from functools import lru_cache
import logging
from contextlib import closing
import threading
import time

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
vectorizer = None
db_path = 'recommendations.db'
connection_pool = threading.local()

def get_db_connection():
    """Get thread-local database connection"""
    if not hasattr(connection_pool, 'connection'):
        connection_pool.connection = sqlite3.connect(db_path, check_same_thread=False)
        connection_pool.connection.row_factory = sqlite3.Row  # Enable dict-like access
    return connection_pool.connection

def load_vectorizer():
    """Load vectorizer for text processing"""
    global vectorizer
    if vectorizer is None:
        if os.path.exists('vectorizer_only.pkl'):
            with open('vectorizer_only.pkl', 'rb') as f:
                vectorizer = pickle.load(f)
            logger.info("Vectorizer loaded")
        else:
            logger.warning("Vectorizer not found - text-based similarity disabled")
    return vectorizer

def check_database():
    """Check if database exists and has data"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database {db_path} not found. Run convert_to_database.py first.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM movies")
    movie_count = cursor.fetchone()[0]
    
    if movie_count == 0:
        raise ValueError("Database is empty. Run convert_to_database.py first.")
    
    logger.info(f"Database ready with {movie_count} movies")

@lru_cache(maxsize=500)
def search_movies_by_title(title_query):
    """Fast cached title search"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    title_clean = title_query.lower().strip()
    
    # Exact match first
    cursor.execute("""
        SELECT id, title, avg_rating FROM movies 
        WHERE clean_title = ? 
        ORDER BY avg_rating DESC LIMIT 1
    """, (title_clean,))
    
    result = cursor.fetchone()
    if result:
        return [(result['id'], result['title'], result['avg_rating'])]
    
    # Partial match
    cursor.execute("""
        SELECT id, title, avg_rating FROM movies 
        WHERE clean_title LIKE ? 
        ORDER BY avg_rating DESC LIMIT 5
    """, (f'%{title_clean}%',))
    
    results = cursor.fetchall()
    return [(row['id'], row['title'], row['avg_rating']) for row in results]

def get_movie_details(movie_ids):
    """Get detailed movie information"""
    if not movie_ids:
        return []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(movie_ids))
    query = f"""
        SELECT * FROM movies 
        WHERE id IN ({placeholders})
        ORDER BY avg_rating DESC
    """
    
    cursor.execute(query, movie_ids)
    results = cursor.fetchall()
    
    movies = []
    for row in results:
        movie = {
            'id': row['id'],
            'title': row['title'],
            'type': row['type'],
            'genres': row['genres'] if row['genres'] else (row['ml_genres'] or '').replace('|', ', '),
            'description': row['description'][:200] + '...' if row['description'] and len(row['description']) > 200 else row['description'] or 'No description available.',
            'release_year': str(row['release_year']) if row['release_year'] else 'Unknown',
            'director': row['director'] or 'Unknown',
            'cast': row['cast'] or 'Unknown',
            'avg_rating': float(row['avg_rating']),
            'rating_count': row['rating_count'] or 0,
            'source': row['source'] or 'Netflix',
            'data_quality': row['data_quality'] or 'estimated'
        }
        
        # Add MovieLens specific data if available
        if row['ml_avg_rating']:
            movie['ml_rating'] = float(row['ml_avg_rating'])
        if row['ml_rating_count']:
            movie['ml_rating_count'] = int(row['ml_rating_count'])
        if row['ml_rating_std']:
            movie['rating_consistency'] = f"±{float(row['ml_rating_std']):.1f}"
        
        movies.append(movie)
    
    return movies

def get_similar_movies_from_db(movie_ids, limit=50):
    """Get similar movies using precomputed similarities"""
    if not movie_ids:
        return []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get similar movies from precomputed similarities
    placeholders = ','.join(['?'] * len(movie_ids))
    query = f"""
        SELECT movie_id_2 as similar_id, AVG(similarity_score) as avg_similarity
        FROM similarities 
        WHERE movie_id_1 IN ({placeholders})
        AND movie_id_2 NOT IN ({placeholders})
        GROUP BY movie_id_2
        ORDER BY avg_similarity DESC
        LIMIT ?
    """
    
    params = movie_ids + movie_ids + [limit]
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    if results:
        similar_ids = [(row['similar_id'], row['avg_similarity']) for row in results]
        return similar_ids
    
    return []

def get_popular_recommendations(category='overall', limit=20):
    """Get popular movies from cache"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT movie_ids FROM popular_cache WHERE category = ?", (category,))
    result = cursor.fetchone()
    
    if result:
        movie_ids = json.loads(result['movie_ids'])[:limit]
        return movie_ids
    
    # Fallback to database query
    cursor.execute("""
        SELECT id FROM movies 
        ORDER BY avg_rating DESC 
        LIMIT ?
    """, (limit,))
    
    return [row['id'] for row in cursor.fetchall()]

def get_content_based_recommendations(movie_ids, limit=30):
    """Get recommendations based on genres, themes, etc."""
    if not movie_ids:
        return []
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get user's preferred genres and attributes
    placeholders = ','.join(['?'] * len(movie_ids))
    query = f"""
        SELECT genres, ml_genres, content_themes, type, data_source
        FROM movies WHERE id IN ({placeholders})
    """
    cursor.execute(query, movie_ids)
    user_movies = cursor.fetchall()
    
    # Extract preferences
    all_genres = set()
    preferred_type = None
    preferred_source = None
    type_counts = {}
    source_counts = {}
    
    for movie in user_movies:
        # Extract genres
        genres = (movie['genres'] or '').split(', ') + (movie['ml_genres'] or '').split('|')
        all_genres.update(g.strip().lower() for g in genres if g.strip())
        
        # Count types and sources
        movie_type = movie['type']
        if movie_type:
            type_counts[movie_type] = type_counts.get(movie_type, 0) + 1
        
        source = movie['data_source']
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
    
    preferred_type = max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
    preferred_source = max(source_counts.items(), key=lambda x: x[1])[0] if source_counts else None
    
    # Build recommendation query
    where_conditions = []
    params = []
    
    if all_genres:
        # Match at least one genre
        genre_conditions = []
        for genre in list(all_genres)[:5]:  # Limit to top 5 genres
            genre_conditions.append("(genres LIKE ? OR ml_genres LIKE ?)")
            params.extend([f'%{genre}%', f'%{genre}%'])
        
        if genre_conditions:
            where_conditions.append(f"({' OR '.join(genre_conditions)})")
    
    if preferred_type:
        where_conditions.append("type = ?")
        params.append(preferred_type)
    
    # Exclude user's movies
    where_conditions.append(f"id NOT IN ({placeholders})")
    params.extend(movie_ids)
    
    # Build final query
    where_clause = ' AND '.join(where_conditions) if where_conditions else '1=1'
    
    query = f"""
        SELECT id, avg_rating FROM movies 
        WHERE {where_clause}
        ORDER BY avg_rating DESC
        LIMIT ?
    """
    params.append(limit)
    
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    return [(row['id'], row['avg_rating']) for row in results]

@app.route('/recommend', methods=['POST'])
def recommend():
    """Super fast recommendation endpoint using database"""
    try:
        data = request.json
        user_titles = data.get('titles', [])
        top_n = min(data.get('top_n', 10), 20)
        
        if not user_titles:
            return jsonify({
                'recommendations': [],
                'message': 'Please provide at least one movie/show title.'
            })
        
        start_time = time.time()
        
        # Find user movies
        user_movie_ids = []
        found_titles = []
        
        for title in user_titles:
            matches = search_movies_by_title(title)
            if matches:
                best_match = matches[0]  # Highest rated match
                user_movie_ids.append(best_match[0])
                found_titles.append(best_match[1])
        
        if not user_movie_ids:
            logger.info("No matches found, using popular fallback")
            popular_ids = get_popular_recommendations('overall', top_n)
            recommendations = get_movie_details(popular_ids)
            
            return jsonify({
                'recommendations': recommendations,
                'message': 'No exact matches found. Here are popular suggestions.',
                'found_titles': [],
                'processing_time': round(time.time() - start_time, 3)
            })
        
        # Get recommendations from multiple sources
        similar_movies = []
        
        # 1. Precomputed similarities (60% of results)
        similarity_results = get_similar_movies_from_db(user_movie_ids, top_n)
        similar_ids_with_scores = similarity_results[:int(top_n * 0.6)]
        
        # 2. Content-based recommendations (40% of results)
        content_results = get_content_based_recommendations(user_movie_ids, top_n)
        content_ids_with_scores = content_results[:int(top_n * 0.4)]
        
        # Combine and deduplicate
        all_recommendations = {}
        
        # Add similarity-based recommendations
        for movie_id, score in similar_ids_with_scores:
            all_recommendations[movie_id] = score
        
        # Add content-based recommendations (with lower weight)
        for movie_id, rating in content_ids_with_scores:
            if movie_id not in all_recommendations:
                all_recommendations[movie_id] = rating * 0.2  # Lower weight for content-based
        
        # Sort by score and get top N
        sorted_recommendations = sorted(all_recommendations.items(), 
                                      key=lambda x: x[1], reverse=True)[:top_n]
        
        final_movie_ids = [movie_id for movie_id, _ in sorted_recommendations]
        
        # If we don't have enough recommendations, fill with popular ones
        if len(final_movie_ids) < top_n:
            popular_ids = get_popular_recommendations('overall', top_n - len(final_movie_ids))
            # Add popular movies that aren't already in recommendations
            for pop_id in popular_ids:
                if pop_id not in final_movie_ids and pop_id not in user_movie_ids:
                    final_movie_ids.append(pop_id)
                if len(final_movie_ids) >= top_n:
                    break
        
        # Get detailed movie information
        recommendations = get_movie_details(final_movie_ids)
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'recommendations': recommendations[:top_n],
            'found_titles': found_titles,
            'message': f'Found {len(found_titles)} titles. Generated {len(recommendations)} recommendations.',
            'processing_time': round(processing_time, 3),
            'recommendation_sources': {
                'similarity_based': len(similar_ids_with_scores),
                'content_based': len(content_ids_with_scores),
                'popular_fallback': max(0, top_n - len(similar_ids_with_scores) - len(content_ids_with_scores))
            }
        })
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        return jsonify({
            'error': str(e),
            'recommendations': [],
            'message': 'Error generating recommendations.'
        }), 500

@app.route('/search', methods=['POST'])
def search_titles():
    """Fast search endpoint"""
    try:
        data = request.json
        query = data.get('query', '').lower().strip()
        limit = min(data.get('limit', 20), 50)
        
        if not query:
            return jsonify({
                'results': [],
                'message': 'Please provide a search query.'
            })
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Search in title, genres, and description
        search_query = f'%{query}%'
        cursor.execute("""
            SELECT id FROM movies 
            WHERE clean_title LIKE ? 
            OR genres LIKE ? 
            OR ml_genres LIKE ?
            OR LOWER(description) LIKE ?
            ORDER BY avg_rating DESC
            LIMIT ?
        """, (search_query, search_query, search_query, search_query, limit))
        
        movie_ids = [row['id'] for row in cursor.fetchall()]
        results = get_movie_details(movie_ids)
        
        return jsonify({
            'results': results,
            'total_found': len(results),
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error in search: {str(e)}")
        return jsonify({
            'error': str(e),
            'results': []
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check with database stats"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM movies")
        total_movies = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM similarities")
        total_similarities = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT source) FROM movies")
        total_sources = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(avg_rating) FROM movies")
        avg_rating = cursor.fetchone()[0]
        
        return jsonify({
            'status': 'healthy',
            'database_type': 'sqlite',
            'total_movies': total_movies,
            'precomputed_similarities': total_similarities,
            'data_sources': total_sources,
            'avg_rating': round(avg_rating, 2) if avg_rating else 0,
            'vectorizer_loaded': vectorizer is not None
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/popular/<category>', methods=['GET'])
def get_popular(category):
    """Get popular movies by category"""
    try:
        limit = min(int(request.args.get('limit', 20)), 50)
        
        popular_ids = get_popular_recommendations(category, limit)
        recommendations = get_movie_details(popular_ids)
        
        return jsonify({
            'category': category,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'recommendations': []
        }), 500

# Initialize on startup
try:
    check_database()
    load_vectorizer()
    logger.info("Database-powered recommendation system ready!")
except Exception as e:
    logger.error(f"Startup error: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
# convert_to_database.py - Convert pickle model to SQLite database
import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_pickle_model():
    """Load the existing pickle model"""
    logger.info("Loading pickle model...")
    
    with open('balanced_recommender_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    if isinstance(model_data, dict):
        df = model_data['df']
        vectorizer = model_data['vectorizer'] 
        features = model_data['features']
    else:
        df, vectorizer, features = model_data
    
    logger.info(f"Loaded {len(df)} titles")
    return df, vectorizer, features

def create_database_schema(conn):
    """Create optimized database schema"""
    logger.info("Creating database schema...")
    
    # Main movies table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS movies (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            clean_title TEXT NOT NULL,
            type TEXT,
            genres TEXT,
            ml_genres TEXT,
            description TEXT,
            director TEXT,
            cast TEXT,
            country TEXT,
            release_year INTEGER,
            avg_rating REAL,
            rating_count INTEGER,
            ml_rating_count INTEGER,
            ml_avg_rating REAL,
            ml_rating_std REAL,
            data_source TEXT,
            source TEXT,
            data_quality TEXT,
            ml_tags TEXT,
            content_themes TEXT,
            content_mood TEXT
        )
    ''')
    
    # Precomputed similarities table (for top movies)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS similarities (
            movie_id_1 INTEGER,
            movie_id_2 INTEGER,
            similarity_score REAL,
            PRIMARY KEY (movie_id_1, movie_id_2),
            FOREIGN KEY (movie_id_1) REFERENCES movies (id),
            FOREIGN KEY (movie_id_2) REFERENCES movies (id)
        )
    ''')
    
    # Popular movies cache table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS popular_cache (
            category TEXT PRIMARY KEY,
            movie_ids TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create indexes for fast queries
    indexes = [
        'CREATE INDEX IF NOT EXISTS idx_title ON movies(title)',
        'CREATE INDEX IF NOT EXISTS idx_clean_title ON movies(clean_title)',
        'CREATE INDEX IF NOT EXISTS idx_genres ON movies(genres)',
        'CREATE INDEX IF NOT EXISTS idx_ml_genres ON movies(ml_genres)',
        'CREATE INDEX IF NOT EXISTS idx_rating ON movies(avg_rating)',
        'CREATE INDEX IF NOT EXISTS idx_type ON movies(type)',
        'CREATE INDEX IF NOT EXISTS idx_source ON movies(source)',
        'CREATE INDEX IF NOT EXISTS idx_data_source ON movies(data_source)',
        'CREATE INDEX IF NOT EXISTS idx_release_year ON movies(release_year)',
        'CREATE INDEX IF NOT EXISTS idx_similarity_1 ON similarities(movie_id_1)',
        'CREATE INDEX IF NOT EXISTS idx_similarity_2 ON similarities(movie_id_2)'
    ]
    
    for index_sql in indexes:
        conn.execute(index_sql)
    
    conn.commit()
    logger.info("Database schema created")

def insert_movies_data(conn, df):
    """Insert movies data into database"""
    logger.info("Inserting movie data...")
    
    # Prepare data for insertion
    movies_data = []
    for idx, row in df.iterrows():
        movie_record = (
            int(idx),  # id
            str(row['title']),
            str(row.get('clean_title', row['title'])).lower().strip(),
            str(row['type']),
            str(row['listed_in']) if pd.notna(row['listed_in']) else '',
            str(row.get('ml_genres', '')) if pd.notna(row.get('ml_genres')) else '',
            str(row['description']) if pd.notna(row['description']) else '',
            str(row['director']) if pd.notna(row['director']) else '',
            str(row['cast']) if pd.notna(row['cast']) else '',
            str(row['country']) if pd.notna(row['country']) else '',
            int(row['release_year']) if pd.notna(row['release_year']) else None,
            float(row.get('avg_rating', 0)),
            int(row.get('rating_count', 0)),
            int(row.get('ml_rating_count', 0)) if pd.notna(row.get('ml_rating_count')) else None,
            float(row.get('ml_avg_rating', 0)) if pd.notna(row.get('ml_avg_rating')) else None,
            float(row.get('ml_rating_std', 0)) if pd.notna(row.get('ml_rating_std')) else None,
            str(row.get('data_source', 'netflix_only')),
            str(row.get('source', 'Netflix')),
            'verified' if row.get('data_source') in ['hybrid', 'movielens_only'] else 'estimated',
            str(row.get('ml_tags', '')) if pd.notna(row.get('ml_tags')) else '',
            str(row.get('content_themes', '')) if pd.notna(row.get('content_themes')) else '',
            str(row.get('content_mood', '')) if pd.notna(row.get('content_mood')) else ''
        )
        movies_data.append(movie_record)
    
    conn.executemany('''
        INSERT OR REPLACE INTO movies VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', movies_data)
    
    conn.commit()
    logger.info(f"Inserted {len(movies_data)} movies")

def compute_and_store_similarities(conn, df, features, limit=1000):
    """Compute and store similarities for top movies"""
    logger.info(f"Computing similarities for top {limit} movies...")
    
    # Get top movies by rating
    top_movies = df.nlargest(limit, 'avg_rating')
    top_indices = top_movies.index.tolist()
    
    # Convert features to array
    if hasattr(features, 'toarray'):
        features_array = features.toarray()
    else:
        features_array = np.asarray(features)
    
    # Compute similarities for top movies
    top_features = features_array[top_indices]
    similarities = cosine_similarity(top_features)
    
    # Store similarities (only for significant similarities > 0.1)
    similarity_data = []
    for i, idx1 in enumerate(top_indices):
        for j, idx2 in enumerate(top_indices):
            if i != j and similarities[i][j] > 0.1:  # Only store significant similarities
                similarity_data.append((int(idx1), int(idx2), float(similarities[i][j])))
    
    conn.executemany('''
        INSERT OR REPLACE INTO similarities (movie_id_1, movie_id_2, similarity_score)
        VALUES (?, ?, ?)
    ''', similarity_data)
    
    conn.commit()
    logger.info(f"Stored {len(similarity_data)} similarity relationships")

def create_popular_cache(conn, df):
    """Create popular movies cache"""
    logger.info("Creating popular movies cache...")
    
    cache_data = []
    
    # Overall popular
    overall_popular = df.nlargest(50, 'avg_rating').index.tolist()
    cache_data.append(('overall', json.dumps(overall_popular)))
    
    # Popular by genre
    for genre in ['Crime', 'Comedy', 'Romance', 'Horror', 'Sci-Fi', 'Action', 'Drama', 'Thriller']:
        genre_mask = (df['listed_in'].str.contains(genre, case=False, na=False) | 
                     df.get('ml_genres', pd.Series(dtype=str)).str.contains(genre, case=False, na=False))
        if genre_mask.any():
            popular = df[genre_mask].nlargest(20, 'avg_rating').index.tolist()
            cache_data.append((f'genre_{genre.lower()}', json.dumps(popular)))
    
    # Popular by source
    for source in df['source'].unique():
        popular = df[df['source'] == source].nlargest(20, 'avg_rating').index.tolist()
        cache_data.append((f'source_{source.lower().replace(" ", "_")}', json.dumps(popular)))
    
    # Popular by type
    for content_type in ['Movie', 'TV Show']:
        popular = df[df['type'] == content_type].nlargest(20, 'avg_rating').index.tolist()
        cache_data.append((f'type_{content_type.lower().replace(" ", "_")}', json.dumps(popular)))
    
    conn.executemany('''
        INSERT OR REPLACE INTO popular_cache (category, movie_ids)
        VALUES (?, ?)
    ''', cache_data)
    
    conn.commit()
    logger.info(f"Created {len(cache_data)} popular movie caches")

def save_vectorizer_separately(vectorizer):
    """Save vectorizer for text processing"""
    logger.info("Saving vectorizer...")
    with open('vectorizer_only.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    logger.info("Vectorizer saved")

def create_database_from_pickle():
    """Main function to convert pickle to database"""
    try:
        # Load pickle model
        df, vectorizer, features = load_pickle_model()
        
        # Create database
        db_name = 'recommendations.db'
        conn = sqlite3.connect(db_name)
        
        # Create schema
        create_database_schema(conn)
        
        # Insert data
        insert_movies_data(conn, df)
        
        # Compute and store similarities
        compute_and_store_similarities(conn, df, features, limit=1000)
        
        # Create popular caches
        create_popular_cache(conn, df)
        
        # Save vectorizer separately
        save_vectorizer_separately(vectorizer)
        
        conn.close()
        
        logger.info(f"Database conversion complete! Database saved as '{db_name}'")
        
        # Print database stats
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM movies")
        movie_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM similarities")
        similarity_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM popular_cache")
        cache_count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\nDatabase Statistics:")
        print(f"Movies: {movie_count}")
        print(f"Similarity relationships: {similarity_count}")
        print(f"Popular caches: {cache_count}")
        
        return db_name
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        raise

if __name__ == "__main__":
    create_database_from_pickle()
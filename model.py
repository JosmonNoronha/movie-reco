# hybrid_model.py - Netflix + MovieLens integration (FIXED)
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import re
import os
from fuzzywuzzy import fuzz

def load_and_merge_datasets():
    """Load both datasets and create a truly hybrid dataset"""
    
    print("Loading Netflix dataset...")
    netflix_df = pd.read_csv('netflix_titles.csv')
    
    print("Loading MovieLens datasets...")
    ratings_df = pd.read_csv('ml-latest-small/ratings.csv')
    movies_df = pd.read_csv('ml-latest-small/movies.csv')
    tags_df = pd.read_csv('ml-latest-small/tags.csv') if os.path.exists('ml-latest-small/tags.csv') else None
    
    # Process MovieLens data
    print("Processing MovieLens ratings...")
    movie_stats = ratings_df.groupby('movieId').agg({
        'rating': ['mean', 'count', 'std']
    }).round(2)
    movie_stats.columns = ['ml_avg_rating', 'ml_rating_count', 'ml_rating_std']
    
    # Merge with movies metadata
    ml_movies = movies_df.merge(movie_stats, on='movieId', how='left')
    
    # Add tags if available
    if tags_df is not None:
        print("Processing MovieLens tags...")
        movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ', '.join(x.unique())).reset_index()
        ml_movies = ml_movies.merge(movie_tags, on='movieId', how='left')
    else:
        ml_movies['tag'] = ''
    
    # Clean titles for matching
    netflix_df['clean_title'] = netflix_df['title'].str.lower().str.strip()
    ml_movies['year'] = ml_movies['title'].str.extract(r'\((\d{4})\)$').astype(float)
    ml_movies['clean_title'] = ml_movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True).str.lower().str.strip()
    
    # STEP 1: Match existing Netflix titles with MovieLens data
    print("Matching Netflix titles with MovieLens data...")
    netflix_df['ml_avg_rating'] = np.nan
    netflix_df['ml_rating_count'] = np.nan
    netflix_df['ml_rating_std'] = np.nan
    netflix_df['ml_genres'] = ''
    netflix_df['ml_tags'] = ''
    netflix_df['data_source'] = 'netflix_only'
    netflix_df['movieId'] = np.nan
    
    matched_movie_ids = set()
    matched_count = 0
    
    for idx, netflix_row in netflix_df.iterrows():
        netflix_title = netflix_row['clean_title']
        netflix_year = netflix_row['release_year']
        
        # Find best match in MovieLens
        best_match = None
        best_score = 0
        
        for _, ml_row in ml_movies.iterrows():
            ml_title = ml_row['clean_title']
            ml_year = ml_row['year']
            
            # Calculate similarity score
            title_score = fuzz.ratio(netflix_title, ml_title)
            
            # Year bonus if available
            if pd.notna(netflix_year) and pd.notna(ml_year):
                year_diff = abs(netflix_year - ml_year)
                if year_diff == 0:
                    title_score += 20
                elif year_diff <= 1:
                    title_score += 10
                elif year_diff <= 2:
                    title_score += 5
            
            if title_score > best_score and title_score >= 85:  # High threshold
                best_score = title_score
                best_match = ml_row
        
        # Apply best match if found
        if best_match is not None:
            netflix_df.at[idx, 'ml_avg_rating'] = best_match['ml_avg_rating']
            netflix_df.at[idx, 'ml_rating_count'] = best_match['ml_rating_count'] 
            netflix_df.at[idx, 'ml_rating_std'] = best_match['ml_rating_std']
            netflix_df.at[idx, 'ml_genres'] = best_match['genres']
            netflix_df.at[idx, 'ml_tags'] = best_match.get('tag', '')
            netflix_df.at[idx, 'data_source'] = 'hybrid'
            netflix_df.at[idx, 'movieId'] = best_match['movieId']
            matched_movie_ids.add(best_match['movieId'])
            matched_count += 1
    
    print(f"Successfully matched {matched_count} Netflix titles with MovieLens data")
    
    # STEP 2: Add MovieLens-only titles to create truly hybrid dataset
    print("Adding MovieLens-only titles...")
    
    # Filter out already matched movies and low-quality entries
    unmatched_ml = ml_movies[
        (~ml_movies['movieId'].isin(matched_movie_ids)) &
        (ml_movies['ml_rating_count'] >= 10) &  # At least 10 ratings
        (ml_movies['ml_avg_rating'] >= 3.0)     # Decent rating
    ].copy()
    
    print(f"Found {len(unmatched_ml)} high-quality MovieLens-only titles")
    
    # Convert MovieLens titles to Netflix format
    movielens_as_netflix = []
    
    for _, ml_row in unmatched_ml.iterrows():
        # Generate synthetic Netflix-style data for MovieLens titles
        title = ml_row['title'].replace(f" ({int(ml_row['year'])})", "") if pd.notna(ml_row['year']) else ml_row['title']
        
        # Determine content type from genres
        genres_lower = str(ml_row['genres']).lower()
        content_type = 'TV Show' if any(tv_indicator in genres_lower for tv_indicator in ['tv', 'series']) else 'Movie'
        
        # Create synthetic description from genres and tags
        description_parts = []
        if pd.notna(ml_row['genres']) and ml_row['genres']:
            description_parts.append(f"A {ml_row['genres'].lower().replace('|', ', ')} ")
        
        if pd.notna(ml_row.get('tag', '')) and ml_row.get('tag', ''):
            tags = str(ml_row['tag']).split(', ')[:3]  # Top 3 tags
            if tags:
                description_parts.append(f"featuring themes of {', '.join(tags).lower()}")
        
        if not description_parts:
            description_parts.append("A compelling story")
        
        description = ''.join(description_parts).strip()
        if not description.endswith('.'):
            description += '.'
        
        # Map MovieLens genres to Netflix format
        netflix_genres = ml_row['genres'].replace('|', ', ') if pd.notna(ml_row['genres']) else ''
        
        netflix_row = {
            'show_id': f"ml_{ml_row['movieId']}",
            'type': content_type,
            'title': title,
            'director': '',  # MovieLens doesn't have director info
            'cast': '',      # MovieLens doesn't have cast info
            'country': '',   # MovieLens doesn't have country info
            'date_added': '', # MovieLens doesn't have date added
            'release_year': int(ml_row['year']) if pd.notna(ml_row['year']) else np.nan,
            'rating': '',    # MovieLens doesn't have content rating
            'duration': '',  # MovieLens doesn't have duration
            'listed_in': netflix_genres,
            'description': description,
            'clean_title': ml_row['clean_title'],
            'ml_avg_rating': ml_row['ml_avg_rating'],
            'ml_rating_count': ml_row['ml_rating_count'],
            'ml_rating_std': ml_row['ml_rating_std'],
            'ml_genres': ml_row['genres'],
            'ml_tags': ml_row.get('tag', ''),
            'data_source': 'movielens_only',
            'movieId': ml_row['movieId']
        }
        
        movielens_as_netflix.append(netflix_row)
    
    # Convert to DataFrame and combine
    if movielens_as_netflix:
        ml_netflix_df = pd.DataFrame(movielens_as_netflix)
        
        # Ensure all columns exist in both DataFrames
        all_columns = set(netflix_df.columns) | set(ml_netflix_df.columns)
        
        for col in all_columns:
            if col not in netflix_df.columns:
                netflix_df[col] = np.nan if col in ['ml_avg_rating', 'ml_rating_count', 'ml_rating_std', 'movieId'] else ''
            if col not in ml_netflix_df.columns:
                ml_netflix_df[col] = np.nan if col in ['ml_avg_rating', 'ml_rating_count', 'ml_rating_std', 'movieId'] else ''
        
        # Combine datasets
        hybrid_df = pd.concat([netflix_df, ml_netflix_df], ignore_index=True)
        print(f"Created hybrid dataset with {len(hybrid_df)} total titles:")
        print(f"  - Netflix-only: {len(hybrid_df[hybrid_df['data_source'] == 'netflix_only'])}")
        print(f"  - Hybrid (Netflix + MovieLens): {len(hybrid_df[hybrid_df['data_source'] == 'hybrid'])}")
        print(f"  - MovieLens-only: {len(hybrid_df[hybrid_df['data_source'] == 'movielens_only'])}")
        
    else:
        hybrid_df = netflix_df
        print("No additional MovieLens titles added")
    
    return hybrid_df, ratings_df, ml_movies

def create_hybrid_features(df):
    """Enhanced feature creation using both datasets"""
    
    # Handle missing values
    df['director'] = df['director'].fillna('')
    df['cast'] = df['cast'].fillna('')
    df['listed_in'] = df['listed_in'].fillna('')
    df['description'] = df['description'].fillna('')
    
    # Use MovieLens ratings when available, synthetic as fallback
    def get_final_rating(row):
        if pd.notna(row['ml_avg_rating']) and row['ml_rating_count'] >= 5:
            return row['ml_avg_rating']
        else:
            # Calculate synthetic rating
            base_score = 3.2
            if row['type'] == 'TV Show':
                base_score += 0.15
            
            genres = str(row['listed_in']).lower()
            genre_bonuses = {
                'crime': 0.4, 'thriller': 0.3, 'drama': 0.2,
                'comedy': 0.25, 'action': 0.2, 'romance': 0.15,
                'horror': 0.2, 'sci-fi': 0.3, 'fantasy': 0.25,
                'documentary': 0.35, 'animation': 0.2
            }
            
            for genre, bonus in genre_bonuses.items():
                if genre in genres:
                    base_score += bonus
                    break
            
            if pd.notna(row['release_year']):
                if row['release_year'] >= 2018:
                    base_score += 0.2
                elif row['release_year'] >= 2010:
                    base_score += 0.1
            
            desc = str(row['description']).lower()
            quality_indicators = ['gripping', 'compelling', 'masterful', 'brilliant', 'outstanding']
            base_score += sum(0.1 for word in quality_indicators if word in desc)
            
            import random
            random.seed(hash(str(row['title'])) % 1000)
            base_score += random.uniform(-0.4, 0.6)
            
            return min(5.0, max(2.2, round(base_score, 1)))
    
    def get_final_rating_count(row):
        if pd.notna(row['ml_rating_count']) and row['ml_rating_count'] >= 5:
            return int(row['ml_rating_count'])
        else:
            return np.random.randint(30, 1200)
    
    df['avg_rating'] = df.apply(get_final_rating, axis=1)
    df['rating_count'] = df.apply(get_final_rating_count, axis=1)
    
    # Enhanced quality assessment
    def get_hybrid_quality(row):
        score = 0
        rating = row['avg_rating']
        count = row['rating_count']
        
        # Real data gets priority
        if pd.notna(row['ml_avg_rating']) and row['ml_rating_count'] >= 20:
            if rating >= 4.2 and count >= 100:
                score += 5  # Verified high quality
            elif rating >= 3.8 and count >= 50:
                score += 4
            elif rating >= 3.5:
                score += 3
        else:
            # Standard quality assessment
            if rating >= 4.5 and count >= 100:
                score += 4
            elif rating >= 4.0 and count >= 50:
                score += 3
            elif rating >= 3.5:
                score += 2
        
        # Low variance bonus (consistent quality)
        if pd.notna(row['ml_rating_std']) and row['ml_rating_std'] < 0.8:
            score += 1
        
        # Description quality indicators
        desc = str(row['description']).lower()
        quality_words = ['award', 'acclaimed', 'masterpiece', 'brilliant', 'outstanding', 
                        'exceptional', 'phenomenal', 'captivating', 'compelling']
        score += sum(1 for word in quality_words if word in desc)
        
        # Recent content gets slight boost
        if pd.notna(row['release_year']) and row['release_year'] >= 2015:
            score += 1
        
        if score >= 7:
            return 'verified_premium '
        elif score >= 5:
            return 'verified_quality ' if pd.notna(row['ml_avg_rating']) else 'premium '
        elif score >= 3:
            return 'high_quality '
        elif score >= 1:
            return 'good_quality '
        return ''
    
    df['quality_indicator'] = df.apply(get_hybrid_quality, axis=1)
    
    # UNIVERSAL THEME EXTRACTION
    theme_keywords = {
        'crime_underworld': ['cartel', 'drug lord', 'mafia', 'gangster', 'heist', 'murder'],
        'investigation': ['detective', 'investigation', 'mystery', 'solve', 'clues'],
        'romance': ['love', 'romantic', 'relationship', 'wedding', 'dating', 'couple'],
        'family_drama': ['family', 'parent', 'child', 'sibling', 'marriage', 'divorce'],
        'action_adventure': ['action', 'fight', 'battle', 'war', 'soldier', 'adventure', 'hero'],
        'superhero': ['superhero', 'powers', 'marvel', 'dc', 'mutant', 'villain'],
        'comedy': ['funny', 'comedy', 'humor', 'laugh', 'hilarious', 'sitcom'],
        'romantic_comedy': ['rom-com', 'romantic comedy', 'meet cute', 'friends to lovers'],
        'sci_fi': ['alien', 'space', 'future', 'robot', 'technology', 'cyberpunk'],
        'fantasy': ['magic', 'wizard', 'dragon', 'medieval', 'fantasy', 'supernatural'],
        'horror': ['horror', 'scary', 'ghost', 'monster', 'demon', 'haunted'],
        'historical': ['historical', 'period', 'based on true', 'biography', 'war'],
        'royalty': ['queen', 'king', 'royal', 'palace', 'crown', 'monarchy'],
        'psychological': ['psychological', 'mental', 'therapy', 'trauma', 'identity'],
        'character_study': ['character study', 'personal journey', 'transformation', 'coming of age'],
        'teen_content': ['teen', 'high school', 'college', 'young adult', 'teenager'],
        'coming_of_age': ['growing up', 'adolescent', 'first love', 'graduation']
    }
    
    def extract_themes(text, genre_text="", ml_genres="", ml_tags=""):
        combined_text = ' '.join([str(text), str(genre_text), str(ml_genres), str(ml_tags)]).lower()
        features = []
        
        for category, keywords in theme_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in combined_text)
            if matches > 0:
                features.extend([category] * min(matches, 2))
        
        return ' '.join(features)
    
    # Extract themes using all available data
    df['content_themes'] = df.apply(lambda x: extract_themes(
        x['description'], x['listed_in'], x['ml_genres'], x.get('ml_tags', '')
    ), axis=1)
    
    # MOOD & TONE ANALYSIS
    def extract_mood(description, tags=""):
        mood_keywords = {
            'dark_serious': ['dark', 'gritty', 'intense', 'brutal', 'harsh', 'violent'],
            'light_hearted': ['light', 'fun', 'cheerful', 'upbeat', 'feel-good', 'heartwarming'],
            'emotional': ['emotional', 'touching', 'heartbreaking', 'inspiring', 'moving'],
            'suspenseful': ['suspense', 'tension', 'thriller', 'edge of seat', 'gripping'],
            'humorous': ['funny', 'hilarious', 'comedy', 'witty', 'amusing', 'laugh']
        }
        
        combined_text = str(description).lower() + ' ' + str(tags).lower()
        moods = []
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                moods.append(mood)
        
        return ' '.join(moods)
    
    df['content_mood'] = df.apply(lambda x: extract_mood(x['description'], x.get('ml_tags', '')), axis=1)
    
    # SMART CAST & DIRECTOR PROCESSING
    def extract_talent_profile(director_str, cast_str):
        people = []
        
        a_list = ['leonardo dicaprio', 'brad pitt', 'angelina jolie', 'will smith', 
                 'tom hanks', 'meryl streep', 'robert downey jr', 'scarlett johansson']
        
        top_directors = ['christopher nolan', 'martin scorsese', 'quentin tarantino', 
                        'steven spielberg', 'david fincher', 'vince gilligan']
        
        combined = str(director_str).lower() + ' ' + str(cast_str).lower()
        
        for director in top_directors:
            if director in combined:
                people.extend([director.replace(' ', '_')] * 3)
        
        for actor in a_list:
            if actor in combined:
                people.extend([actor.replace(' ', '_')] * 2)
        
        if cast_str and isinstance(cast_str, str):
            cast_members = [member.strip().replace(' ', '_') for member in cast_str.split(',')[:3]]
            people.extend(cast_members)
        
        return ' '.join(people)
    
    df['talent_profile'] = df.apply(lambda x: extract_talent_profile(x['director'], x['cast']), axis=1)
    
    # ENHANCED GENRE PROCESSING (Netflix + MovieLens)
    def process_hybrid_genres(netflix_genres, ml_genres):
        all_genres = str(netflix_genres).lower() + ' ' + str(ml_genres).lower().replace('|', ' ')
        processed = []
        
        major_genres = {
            'crime': 'crime_content',
            'thriller': 'thriller_content', 
            'drama': 'drama_content',
            'comedy': 'comedy_content',
            'action': 'action_content',
            'romance': 'romance_content',
            'horror': 'horror_content',
            'sci-fi': 'scifi_content',
            'fantasy': 'fantasy_content',
            'documentary': 'documentary_content',
            'animation': 'animation_content',
            'adventure': 'adventure_content',
            'musical': 'musical_content',
            'western': 'western_content',
            'war': 'war_content'
        }
        
        for genre_key, genre_value in major_genres.items():
            if genre_key in all_genres:
                processed.extend([genre_value] * 2)
        
        return ' '.join(processed)
    
    df['balanced_genres'] = df.apply(lambda x: process_hybrid_genres(x['listed_in'], x['ml_genres']), axis=1)
    
    # CONTENT ERA & TYPE
    def get_content_era_type(row):
        profile = []
        
        if row['type'] == 'TV Show':
            profile.extend(['series_content'] * 2)
        else:
            profile.extend(['movie_content'] * 2)
        
        year = row['release_year']
        if pd.notna(year):
            if year >= 2020:
                profile.append('current_era')
            elif year >= 2010:
                profile.append('modern_era') 
            elif year >= 2000:
                profile.append('millennium')
            else:
                profile.append('classic')
        
        return ' '.join(profile)
    
    df['era_type'] = df.apply(get_content_era_type, axis=1)
    
    # Add data reliability and source information
    def get_reliability_and_source(row):
        if row['data_source'] == 'hybrid':
            return 'verified_hybrid_data ', 'Netflix + MovieLens'
        elif row['data_source'] == 'movielens_only':
            return 'verified_ml_data ', 'MovieLens'
        else:
            return 'netflix_data ', 'Netflix'
    
    df[['reliability', 'source']] = df.apply(
        lambda x: pd.Series(get_reliability_and_source(x)), axis=1
    )
    
    # ENHANCED FEATURE SOUP
    df['soup'] = (
        # Content themes - 4x
        df['content_themes'].apply(lambda x: (x + ' ') * 4) +
        
        # Balanced genres - 3x  
        df['balanced_genres'].apply(lambda x: (x + ' ') * 3) +
        
        # Talent profile - 3x
        df['talent_profile'].apply(lambda x: (x + ' ') * 3) +
        
        # Content mood - 2x
        df['content_mood'].apply(lambda x: (x + ' ') * 2) +
        
        # Quality indicator - 2x
        df['quality_indicator'].apply(lambda x: (x + ' ') * 2) +
        
        # Era and type - 2x
        df['era_type'].apply(lambda x: (x + ' ') * 2) +
        
        # Data reliability - 1x
        df['reliability'] +
        
        # MovieLens tags (when available) - 1x
        df.get('ml_tags', pd.Series([''] * len(df))).apply(lambda x: str(x).lower().replace(',', ' ') + ' ') +
        
        # Clean description - 1x
        df['description'].apply(lambda x: re.sub(r'[^\w\s]', ' ', str(x).lower()))
    )
    
    return df

def create_universal_features(df):
    """Wrapper to maintain compatibility"""
    return create_hybrid_features(df)

# Main execution
import os

# Check if files exist
required_files = ['netflix_titles.csv', 'ml-latest-small/ratings.csv', 'ml-latest-small/movies.csv']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print(f"Error: Missing required files: {missing_files}")
    print("Please ensure you have:")
    print("1. netflix_titles.csv in the current directory")
    print("2. ml-latest-small/ folder with ratings.csv and movies.csv")
    exit(1)

hybrid_df, ratings_df, ml_movies = load_and_merge_datasets()

print("Creating enhanced hybrid features...")
df = create_hybrid_features(hybrid_df)

print("Hybrid vectorization...")
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=15000,  # Increased for more features
    ngram_range=(1, 2),
    min_df=2,  # Lowered to capture MovieLens-specific terms
    max_df=0.85,
    sublinear_tf=True
)

features = vectorizer.fit_transform(df['soup'])

print("Saving enhanced hybrid model...")
model_data = {
    'df': df,
    'vectorizer': vectorizer, 
    'features': features,
    'movielens_data': {
        'ratings': ratings_df,
        'movies': ml_movies
    },
    'model_version': 'hybrid_v2.0'
}

with open('balanced_recommender_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Enhanced hybrid model created!")
print(f"Total dataset size: {len(df)} titles")
print(f"Feature dimensions: {features.shape}")
print("\nDataset breakdown:")
print(f"  Netflix-only titles: {len(df[df['data_source'] == 'netflix_only'])}")
print(f"  Hybrid titles (Netflix + MovieLens): {len(df[df['data_source'] == 'hybrid'])}")
print(f"  MovieLens-only titles: {len(df[df['data_source'] == 'movielens_only'])}")

# Test across different sources
print("\nSource distribution:")
for source in df['source'].unique():
    count = len(df[df['source'] == source])
    avg_rating = df[df['source'] == source]['avg_rating'].mean()
    print(f"  {source}: {count} titles (avg rating: {avg_rating:.2f})")

# Test across different genres with source info
print("\nGenre analysis with sources:")
test_genres = ['Crime', 'Comedy', 'Romance', 'Horror', 'Sci-Fi', 'Action']
for genre in test_genres:
    genre_mask = df['listed_in'].str.contains(genre, case=False, na=False) | df['ml_genres'].str.contains(genre, case=False, na=False)
    total_count = len(df[genre_mask])
    netflix_count = len(df[genre_mask & (df['source'] == 'Netflix')])
    hybrid_count = len(df[genre_mask & (df['source'] == 'Netflix + MovieLens')])
    ml_count = len(df[genre_mask & (df['source'] == 'MovieLens')])
    
    print(f"  {genre}: {total_count} total (Netflix: {netflix_count}, Hybrid: {hybrid_count}, MovieLens: {ml_count})")

print(f"\nModel saved as 'balanced_recommender_model.pkl'")
print("The model now includes both Netflix and MovieLens titles!")
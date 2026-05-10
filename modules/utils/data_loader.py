"""Loader đọc u.data, u.item, u.user của MovieLens 100K."""

import os
import pandas as pd
import numpy as np

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]


def load_ratings(data_dir='ml-100k'):
    """Load u.data → DataFrame (user_id, movie_id, rating, timestamp)."""
    path = os.path.join(data_dir, 'u.data')
    df = pd.read_csv(
        path, sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    df = df.astype({'user_id': int, 'movie_id': int, 'rating': float, 'timestamp': int})
    return df


def load_movies(data_dir='ml-100k'):
    """Load u.item → DataFrame (movie_id, title, year, genre_flags...)."""
    path = os.path.join(data_dir, 'u.item')
    cols = ['movie_id', 'title', 'release_date', 'video_release', 'imdb_url'] + GENRE_COLS
    df = pd.read_csv(
        path, sep='|', names=cols,
        encoding='latin-1',
        usecols=[0, 1, 2] + list(range(5, 24))
    )
    # Trích năm từ title, ví dụ "Toy Story (1995)" → 1995
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    df['year'] = df['year'].fillna(df['year'].median())
    df = df.drop(columns=['release_date'])
    df = df.astype({'movie_id': int})
    return df


def load_users(data_dir='ml-100k'):
    """Load u.user → DataFrame (user_id, age, gender, occupation, zip)."""
    path = os.path.join(data_dir, 'u.user')
    df = pd.read_csv(
        path, sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip']
    )
    df = df.astype({'user_id': int, 'age': int})
    return df


def load_all(data_dir='ml-100k'):
    """Load cả 3 file, trả về (ratings, movies, users)."""
    ratings = load_ratings(data_dir)
    movies  = load_movies(data_dir)
    users   = load_users(data_dir)

    print(f"Ratings : {len(ratings):>7,} rows  |  users={ratings['user_id'].nunique()}, movies={ratings['movie_id'].nunique()}")
    print(f"Movies  : {len(movies):>7,} rows  |  genres={len(GENRE_COLS)}")
    print(f"Users   : {len(users):>7,} rows  |  occupations={users['occupation'].nunique()}")
    return ratings, movies, users

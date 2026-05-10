"""Feature engineering: build user_features, movie_features, interaction_matrix, CF latent factors."""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

N_USERS  = 943
N_MOVIES = 1682
N_LATENT = 50   # số chiều SVD

def build_interaction_matrix(ratings):
    """Ma trận rating 943×1682. matrix[u,m]=rating hoặc 0."""
    matrix = np.zeros((N_USERS, N_MOVIES), dtype=np.float32)
    u_idx  = ratings['user_id'].values - 1
    m_idx  = ratings['movie_id'].values - 1
    r_vals = ratings['rating'].values.astype(np.float32)
    matrix[u_idx, m_idx] = r_vals
    sparsity = (matrix == 0).sum() / matrix.size
    print(f"Interaction matrix: {matrix.shape}  |  sparsity = {sparsity:.2%}")
    return matrix

def build_user_features(train_ratings, users, movies):
    """User feature matrix 943×~46: demographics, rating stats, genre preferences."""
    global_mean = train_ratings['rating'].mean()

    stats = (train_ratings.groupby('user_id')['rating']
             .agg(mean_rating='mean', rating_std='std', num_ratings='count')
             .reset_index())
    stats['rating_std']  = stats['rating_std'].fillna(0.0)
    stats['rating_bias'] = stats['mean_rating'] - global_mean

    time_stats = (train_ratings.groupby('user_id')['timestamp']
                  .agg(first_ts='min', last_ts='max').reset_index())
    time_stats['active_days'] = ((time_stats['last_ts'] - time_stats['first_ts'])
                                  / 86400).clip(lower=0).round(1)

    rg = train_ratings.merge(movies[['movie_id'] + GENRE_COLS], on='movie_id', how='left')
    # Long format: 1 row per (user, genre) pair where movie has that genre
    rg_long = rg.melt(id_vars=['user_id', 'rating'],
                      value_vars=GENRE_COLS,
                      var_name='genre', value_name='has_genre')
    rg_long = rg_long[rg_long['has_genre'] == 1]
    genre_pref = (rg_long.pivot_table(index='user_id', columns='genre',
                                       values='rating', aggfunc='mean')
                  .reindex(columns=GENRE_COLS)
                  .fillna(global_mean)
                  .reindex(range(1, N_USERS + 1), fill_value=global_mean))
    genre_pref.columns = [f'pref_{g}' for g in GENRE_COLS]
    pref_cols = genre_pref.columns.tolist()

    pv = genre_pref.values.copy()
    pv_min = pv.min(axis=1, keepdims=True)
    pv_norm = (pv - pv_min + 1e-9)
    pv_norm = pv_norm / pv_norm.sum(axis=1, keepdims=True)
    genre_entropy = -np.sum(pv_norm * np.log(pv_norm + 1e-9), axis=1)
    genre_pref_reset = genre_pref.reset_index().rename(columns={'index': 'user_id'})
    genre_pref_reset['genre_entropy'] = genre_entropy

    genre_count = (rg.groupby('user_id')[GENRE_COLS]
                   .any().astype(int).sum(axis=1)
                   .rename('genre_count_rated').reset_index())

    u = users.copy()
    u['gender_binary'] = (u['gender'] == 'M').astype(int)
    OCCUPATIONS = sorted(u['occupation'].unique().tolist())
    occ_dummies = pd.get_dummies(u['occupation'], prefix='occ')
    occ_cols = occ_dummies.columns.tolist()
    u = pd.concat([u[['user_id', 'age', 'gender_binary']], occ_dummies], axis=1)

    feat = (u.merge(stats, on='user_id', how='left')
             .merge(time_stats[['user_id', 'active_days']], on='user_id', how='left')
             .merge(genre_pref_reset, on='user_id', how='left')
             .merge(genre_count, on='user_id', how='left')
             .sort_values('user_id')
             .fillna(0.0))

    feature_names = [c for c in feat.columns if c != 'user_id']
    matrix = feat.drop('user_id', axis=1).values.astype(np.float32)
    print(f"user_features: {matrix.shape}  |  {len(feature_names)} features")
    return matrix, feature_names

def build_movie_features(train_ratings, movies):
    """Movie feature matrix 1682×~28: genre flags, popularity stats, bayesian avg."""
    global_mean = train_ratings['rating'].mean()
    C = 50  # prior count cho Bayesian average

    stats = (train_ratings.groupby('movie_id')['rating']
             .agg(avg_rating='mean', rating_std='std', num_ratings='count')
             .reset_index())
    stats['rating_std'] = stats['rating_std'].fillna(0.0)

    # Bayesian average: kéo về global_mean khi số lượng rating ít
    stats['bayesian_avg'] = ((stats['num_ratings'] * stats['avg_rating'] + C * global_mean)
                              / (stats['num_ratings'] + C))

    # Popularity rank (percentile: 0 = ít phổ biến nhất, 1 = phổ biến nhất)
    stats['popularity_rank']    = stats['num_ratings'].rank(pct=True).astype(np.float32)
    stats['rating_percentile']  = stats['avg_rating'].rank(pct=True).astype(np.float32)

    m = movies[['movie_id', 'year'] + GENRE_COLS].copy()
    m['year'] = m['year'].fillna(m['year'].median())
    m['movie_age']   = (2000 - m['year']).clip(lower=0)   # tuổi phim tính đến 2000
    m['genre_count'] = m[GENRE_COLS].sum(axis=1)

    feat = (m.merge(stats, on='movie_id', how='left')
             .sort_values('movie_id')
             .fillna({'avg_rating': global_mean, 'rating_std': 0.0,
                      'num_ratings': 0, 'bayesian_avg': global_mean,
                      'popularity_rank': 0.0, 'rating_percentile': 0.5}))

    feature_names = [c for c in feat.columns if c not in ('movie_id', 'year')]
    matrix = feat.drop(['movie_id', 'year'], axis=1).values.astype(np.float32)
    print(f"movie_features: {matrix.shape}  |  {len(feature_names)} features")
    return matrix, feature_names

def build_cf_features(interaction_matrix, k=N_LATENT):
    """Truncated SVD (k latent factors) trên ma trận rating mean-centered per user."""
    mat = interaction_matrix.copy().astype(np.float64)

    # Mean-center theo user (chỉ trên rated items)
    rated_mask  = mat != 0
    user_counts = rated_mask.sum(axis=1).clip(min=1)
    user_means  = mat.sum(axis=1) / user_counts
    mat = mat - (user_means[:, np.newaxis] * rated_mask)

    # SVD
    sparse = csr_matrix(mat)
    U, sigma, Vt = svds(sparse, k=k)

    # Scale: user_factor[u] · movie_factor[m] ≈ normalized predicted rating
    sqrt_sigma    = np.sqrt(np.abs(sigma))
    user_factors  = (U  * sqrt_sigma[np.newaxis, :]).astype(np.float32)
    movie_factors = (Vt * sqrt_sigma[:, np.newaxis]).T.astype(np.float32)

    print(f"user_factors : {user_factors.shape}")
    print(f"movie_factors: {movie_factors.shape}")
    return user_factors, movie_factors

def save_all_features(out_dir, interaction_matrix, user_features, movie_features,
                      user_factors, movie_factors, user_feat_names, movie_feat_names,
                      train_df, test_df):
    """Lưu tất cả feature matrices và metadata vào out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, 'interaction_matrix.npy'), interaction_matrix)
    np.save(os.path.join(out_dir, 'user_features.npy'),      user_features)
    np.save(os.path.join(out_dir, 'movie_features.npy'),     movie_features)
    np.save(os.path.join(out_dir, 'user_latent.npy'),        user_factors)
    np.save(os.path.join(out_dir, 'movie_latent.npy'),       movie_factors)
    np.save(os.path.join(out_dir, 'user_feat_names.npy'),
            np.array(user_feat_names))
    np.save(os.path.join(out_dir, 'movie_feat_names.npy'),
            np.array(movie_feat_names))

    train_df.to_csv(os.path.join(out_dir, 'train_ratings.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir,  'test_ratings.csv'),  index=False)

    print(f"\n✅ Saved to '{out_dir}/':")
    files = {
        'interaction_matrix.npy': interaction_matrix.shape,
        'user_features.npy':      user_features.shape,
        'movie_features.npy':     movie_features.shape,
        'user_latent.npy':        user_factors.shape,
        'movie_latent.npy':       movie_factors.shape,
    }
    for fname, shape in files.items():
        print(f"  {fname:<30} {str(shape)}")

def load_all_features(feat_dir='features'):
    """Load tất cả feature matrices đã lưu từ feat_dir."""
    return {
        'interaction_matrix': np.load(os.path.join(feat_dir, 'interaction_matrix.npy')),
        'user_features':      np.load(os.path.join(feat_dir, 'user_features.npy')),
        'movie_features':     np.load(os.path.join(feat_dir, 'movie_features.npy')),
        'user_latent':        np.load(os.path.join(feat_dir, 'user_latent.npy')),
        'movie_latent':       np.load(os.path.join(feat_dir, 'movie_latent.npy')),
        'user_feat_names':    np.load(os.path.join(feat_dir, 'user_feat_names.npy'), allow_pickle=True).tolist(),
        'movie_feat_names':   np.load(os.path.join(feat_dir, 'movie_feat_names.npy'), allow_pickle=True).tolist(),
        'train_ratings':      pd.read_csv(os.path.join(feat_dir, 'train_ratings.csv')),
        'test_ratings':       pd.read_csv(os.path.join(feat_dir, 'test_ratings.csv')),
    }

def build_ml_features(ratings_df, users_df, movies_df, global_mean=3.53):
    """32 float features cho DT regression: user stats, movie stats, 19 genre flags, biases."""
    user_stats = (
        ratings_df.groupby('user_id')['rating']
        .agg(mean_rating='mean', num_ratings='count')
        .reset_index()
    )
    user_stats = users_df.merge(user_stats, on='user_id', how='left').fillna(
        {'mean_rating': global_mean, 'num_ratings': 0}
    )

    rg = ratings_df.merge(movies_df[['movie_id'] + GENRE_COLS], on='movie_id', how='left')
    rg_long = rg.melt(id_vars=['user_id', 'rating'],
                      value_vars=GENRE_COLS, var_name='genre', value_name='has_genre')
    rg_long = rg_long[rg_long['has_genre'] == 1]
    genre_pref = (
        rg_long.pivot_table(index='user_id', columns='genre',
                            values='rating', aggfunc='mean')
        .reindex(columns=GENRE_COLS).fillna(global_mean)
    )

    def _top3(uid):
        if uid not in genre_pref.index:
            return set()
        return set(genre_pref.loc[uid].nlargest(3).index)

    user_top3 = {uid: _top3(uid) for uid in user_stats['user_id']}

    movie_stats = (
        ratings_df.groupby('movie_id')['rating']
        .agg(avg_rating='mean', num_ratings_m='count')
        .reset_index()
    )
    C = 50
    movie_stats['bayesian_avg'] = (
        (movie_stats['num_ratings_m'] * movie_stats['avg_rating'] + C * global_mean)
        / (movie_stats['num_ratings_m'] + C)
    )
    movies_ext = movies_df.merge(movie_stats, on='movie_id', how='left').fillna(
        {'avg_rating': global_mean, 'num_ratings_m': 0, 'bayesian_avg': global_mean}
    )
    movies_dict = movies_ext.set_index('movie_id').to_dict('index')
    user_dict   = user_stats.set_index('user_id').to_dict('index')

    X_rows, y_rows = [], []
    for _, row in ratings_df.iterrows():
        uid = int(row['user_id'])
        mid = int(row['movie_id'])

        us = user_dict.get(uid, {})
        ms = movies_dict.get(mid, {})

        age      = float(us.get('age', 30))
        gender   = 1 if us.get('gender', 'M') == 'M' else 0
        activity = float(np.log1p(us.get('num_ratings', 0)))
        mean_r   = float(us.get('mean_rating', global_mean))
        tendency = (0 if mean_r < global_mean - 0.3
                    else 2 if mean_r > global_mean + 0.3 else 1)

        yr      = float(ms.get('year', 1985))
        bq      = float(ms.get('bayesian_avg', global_mean))
        pop_log = float(np.log1p(ms.get('num_ratings_m', 0)))
        n_g     = int(sum(int(ms.get(g, 0)) for g in GENRE_COLS))

        genre_flags = [int(ms.get(g, 0)) for g in GENRE_COLS]
        top3        = user_top3.get(uid, set())
        top3_match  = 1 if any(ms.get(g, 0) for g in top3) else 0
        user_bias   = mean_r - global_mean
        movie_bias  = bq - global_mean
        q_x_gen     = 1 if bq > global_mean + 0.5 and tendency == 2 else 0

        X_rows.append([age, gender, activity, mean_r, tendency,
                        yr, bq, pop_log, n_g] + genre_flags +
                       [top3_match, user_bias, movie_bias, q_x_gen])
        y_rows.append(float(row['rating']))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)

def build_features_for_inference(user_id, movie_ids, train_df, users_df, movies_df,
                                  global_mean=3.53, precomputed_movie_stats=None):
    """32 features cho (user, movie) pairs lúc inference, không cần ratings thật."""
    u_row      = users_df[users_df['user_id'] == user_id]
    age        = float(u_row['age'].iloc[0]) if len(u_row) > 0 else 30.0
    gender     = 1 if (len(u_row) > 0 and u_row['gender'].iloc[0] == 'M') else 0
    u_ratings  = train_df[train_df['user_id'] == user_id]
    num_r      = len(u_ratings)
    mean_r     = float(u_ratings['rating'].mean()) if num_r > 0 else global_mean
    activity   = float(np.log1p(num_r))
    tendency   = 0 if mean_r < global_mean - 0.3 else (2 if mean_r > global_mean + 0.3 else 1)
    user_bias  = mean_r - global_mean

    # Top-3 genres từ lịch sử user
    top3 = set()
    if num_r > 0:
        rg = u_ratings.merge(movies_df[['movie_id'] + GENRE_COLS], on='movie_id', how='left')
        rg_long = rg.melt(id_vars=['rating'], value_vars=GENRE_COLS,
                          var_name='genre', value_name='has_genre')
        rg_long = rg_long[rg_long['has_genre'] == 1]
        if len(rg_long) > 0:
            top3 = set(rg_long.groupby('genre')['rating'].mean().nlargest(3).index)

    if precomputed_movie_stats is not None:
        movies_dict = precomputed_movie_stats
    else:
        movie_stats = (
            train_df.groupby('movie_id')['rating']
            .agg(avg_rating='mean', num_ratings_m='count').reset_index()
        )
        C = 50
        movie_stats['bayesian_avg'] = (
            (movie_stats['num_ratings_m'] * movie_stats['avg_rating'] + C * global_mean)
            / (movie_stats['num_ratings_m'] + C)
        )
        movies_ext  = movies_df.merge(movie_stats, on='movie_id', how='left').fillna(
            {'avg_rating': global_mean, 'num_ratings_m': 0, 'bayesian_avg': global_mean}
        )
        movies_dict = movies_ext.set_index('movie_id').to_dict('index')

    X_rows = []
    for mid in movie_ids:
        ms      = movies_dict.get(int(mid), {})
        yr      = float(ms.get('year', 1985))
        bq      = float(ms.get('bayesian_avg', global_mean))
        pop_log = float(np.log1p(ms.get('num_ratings_m', 0)))
        n_g     = int(sum(int(ms.get(g, 0)) for g in GENRE_COLS))
        g_flags = [int(ms.get(g, 0)) for g in GENRE_COLS]
        top3_match = 1 if any(ms.get(g, 0) for g in top3) else 0
        movie_bias = bq - global_mean
        q_x_gen    = 1 if bq > global_mean + 0.5 and tendency == 2 else 0

        X_rows.append([age, gender, activity, mean_r, tendency,
                       yr, bq, pop_log, n_g] + g_flags +
                      [top3_match, user_bias, movie_bias, q_x_gen])

    return np.array(X_rows, dtype=np.float32)

def build_and_save_all(train_ratings, test_ratings, ratings, movies, users,
                       out_dir='features'):
    """Build toàn bộ feature matrices và lưu file."""
    print("=" * 55)
    print("1/4  Interaction matrix...")
    interaction_matrix = build_interaction_matrix(train_ratings)

    print("\n2/4  User features...")
    user_features, user_feat_names = build_user_features(train_ratings, users, movies)

    print("\n3/4  Movie features...")
    movie_features, movie_feat_names = build_movie_features(train_ratings, movies)

    print("\n4/4  CF latent features (SVD k=50)...")
    user_factors, movie_factors = build_cf_features(interaction_matrix)

    save_all_features(out_dir, interaction_matrix, user_features, movie_features,
                      user_factors, movie_factors, user_feat_names, movie_feat_names,
                      train_ratings, test_ratings)
    return {
        'interaction_matrix': interaction_matrix,
        'user_features':      user_features,
        'movie_features':     movie_features,
        'user_factors':       user_factors,
        'movie_factors':      movie_factors,
        'user_feat_names':    user_feat_names,
        'movie_feat_names':   movie_feat_names,
    }

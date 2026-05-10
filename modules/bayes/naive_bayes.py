"""[D] Naive Bayes binary classifier (like/dislike) với Laplace smoothing."""

import numpy as np
import pickle

GENRE_COLS = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
    'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def build_nb_features(ratings_df, users_df, movies_df, global_mean=3.53):
    """
    Xây dựng feature matrix X và label vector y cho Naive Bayes.

    Với mỗi (user, movie) interaction:
        Label y  = 1 nếu rating >= 4 (like), 0 nếu ngược lại

    Features (tất cả integer-encoded):
        [0]  user_age_group    : 0=<18, 1=18-30, 2=31-50, 3=51+
        [1]  user_gender       : 0=F, 1=M
        [2]  user_activity     : 0=low(<20), 1=mid(20-100), 2=high(>100)
        [3]  user_tendency     : 0=harsh(<gm-0.3), 1=normal, 2=generous(>gm+0.3)
        [4]  movie_decade      : 0=pre-1970, 1=70s, 2=80s, 3=90s, 4=2000s
        [5]  movie_quality     : 0=low(<gm-0.2), 1=mid, 2=high(>gm+0.3)
        [6]  movie_popularity  : 0=rare(<10), 1=mid(10-100), 2=popular(>100)
        [7-25] genre_pref_g    : 1 nếu movie thuộc genre g VÀ user thích genre g
                                  (user pref > global_mean), else 0   ← 19 features
        [26] top_genre_match   : 1 nếu movie có ít nhất 1 trong top-3 genre của user
        [27] quality_x_generous: 1 nếu movie_quality=high VÀ user_tendency=generous

    Tổng: 28 features
    """
    # ── User stats ─────────────────────────────────────────────────────────
    user_stats = (
        ratings_df.groupby('user_id')['rating']
        .agg(mean_rating='mean', num_ratings='count')
        .reset_index()
    )
    user_stats = users_df.merge(user_stats, on='user_id', how='left').fillna(
        {'mean_rating': global_mean, 'num_ratings': 0}
    )

    # ── Genre preference per user ───────────────────────────────────────────
    rg = ratings_df.merge(movies_df[['movie_id'] + GENRE_COLS], on='movie_id', how='left')
    rg_long = rg.melt(
        id_vars=['user_id', 'rating'],
        value_vars=GENRE_COLS,
        var_name='genre', value_name='has_genre'
    )
    rg_long = rg_long[rg_long['has_genre'] == 1]
    genre_pref = (
        rg_long.pivot_table(index='user_id', columns='genre',
                            values='rating', aggfunc='mean')
        .reindex(columns=GENRE_COLS)
        .fillna(global_mean)
    )

    def _top3_genres(uid):
        if uid not in genre_pref.index:
            return set()
        row = genre_pref.loc[uid]
        return set(row.nlargest(3).index)

    user_top3 = {uid: _top3_genres(uid) for uid in user_stats['user_id']}

    # ── Movie stats ─────────────────────────────────────────────────────────
    movie_stats = (
        ratings_df.groupby('movie_id')['rating']
        .agg(avg_rating='mean', num_ratings_m='count')
        .reset_index()
    )
    import pandas as pd
    C = 50
    movie_stats['bayesian_avg'] = (
        (movie_stats['num_ratings_m'] * movie_stats['avg_rating'] + C * global_mean)
        / (movie_stats['num_ratings_m'] + C)
    )
    movies_ext = movies_df.merge(movie_stats, on='movie_id', how='left').fillna(
        {'avg_rating': global_mean, 'num_ratings_m': 0, 'bayesian_avg': global_mean}
    )
    movies_dict = movies_ext.set_index('movie_id').to_dict('index')

    # ── Build feature matrix ────────────────────────────────────────────────
    user_dict = user_stats.set_index('user_id').to_dict('index')

    X_rows, y_rows = [], []
    for _, row in ratings_df.iterrows():
        uid  = int(row['user_id'])
        mid  = int(row['movie_id'])
        rat  = float(row['rating'])
        y    = 1 if rat >= 4.0 else 0

        us   = user_dict.get(uid, {})
        ms   = movies_dict.get(mid, {})

        age  = us.get('age', 30)
        age_group = (0 if age < 18 else 1 if age < 31 else 2 if age <= 50 else 3)

        gender = 1 if us.get('gender', 'M') == 'M' else 0

        nr = us.get('num_ratings', 0)
        activity = 0 if nr < 20 else (1 if nr <= 100 else 2)

        mr = us.get('mean_rating', global_mean)
        tendency = (0 if mr < global_mean - 0.3
                    else 2 if mr > global_mean + 0.3
                    else 1)

        yr = ms.get('year', 1985)
        decade = (0 if yr < 1970 else 1 if yr < 1980 else
                  2 if yr < 1990 else 3 if yr < 2000 else 4)

        bq = ms.get('bayesian_avg', global_mean)
        quality = (0 if bq < global_mean - 0.2
                   else 2 if bq > global_mean + 0.3
                   else 1)

        pop_m = ms.get('num_ratings_m', 0)
        popularity = 0 if pop_m < 10 else (1 if pop_m <= 100 else 2)

        # Genre preference interaction features (19)
        genre_feat = []
        upref = genre_pref.loc[uid] if uid in genre_pref.index else None
        for g in GENRE_COLS:
            movie_has_g = int(ms.get(g, 0))
            if upref is not None:
                user_likes_g = 1 if upref.get(g, global_mean) > global_mean else 0
            else:
                user_likes_g = 0
            genre_feat.append(movie_has_g * user_likes_g)

        # Top-3 genre match
        top3 = user_top3.get(uid, set())
        top3_match = 1 if any(ms.get(g, 0) for g in top3) else 0

        # Interaction: quality × generous
        q_x_gen = 1 if quality == 2 and tendency == 2 else 0

        feat = [age_group, gender, activity, tendency, decade,
                quality, popularity] + genre_feat + [top3_match, q_x_gen]
        X_rows.append(feat)
        y_rows.append(y)

    return np.array(X_rows, dtype=np.int32), np.array(y_rows, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Naive Bayes Classifier
# ─────────────────────────────────────────────────────────────────────────────

class NaiveBayes:
    """
    Categorical Naive Bayes với Laplace smoothing.

    Mô hình:
        P(y=c | x) ∝ P(y=c) · Π_i P(x_i | y=c)

    Với mỗi feature i và class c:
        P(x_i = v | y=c) = (count(x_i=v, y=c) + α) / (count(y=c) + |V_i| · α)

    Args:
        alpha: Laplace smoothing (default=1.0)
    """

    def __init__(self, alpha=1.0):
        self.alpha       = alpha
        self.classes_    = None
        self.log_priors_ = None        # {class: log P(y=c)}
        self.log_cond_   = None        # list of {class: {val: log P(x_i=v|y=c)}}
        self.n_features_ = None

    def fit(self, X, y):
        """
        X: (n, p) int array
        y: (n,) int array {0, 1}
        """
        X = np.asarray(X, dtype=int)
        y = np.asarray(y, dtype=int)
        n, p = X.shape
        self.n_features_ = p
        self.classes_    = np.unique(y)

        # ── Class log-priors ─────────────────────────────────────────────────
        self.log_priors_ = {}
        for c in self.classes_:
            nc = (y == c).sum()
            self.log_priors_[c] = np.log(
                (nc + self.alpha) / (n + len(self.classes_) * self.alpha)
            )

        # ── Feature conditional log-probs ────────────────────────────────────
        self.log_cond_ = []
        for i in range(p):
            vals  = np.unique(X[:, i])
            n_v   = len(vals)
            feat  = {}
            for c in self.classes_:
                mask = (y == c)
                nc   = mask.sum()
                val_lp = {}
                for v in vals:
                    cnt = np.sum(X[mask, i] == v)
                    val_lp[int(v)] = np.log(
                        (cnt + self.alpha) / (nc + n_v * self.alpha)
                    )
                # fallback cho giá trị không thấy khi train
                feat[c] = {
                    'vals':      vals,
                    'val_lp':    val_lp,
                    'fallback':  np.log(self.alpha / (nc + n_v * self.alpha + 1e-12)),
                }
            self.log_cond_.append(feat)
        return self

    def predict_log_proba(self, X):
        X  = np.asarray(X, dtype=int)
        n  = len(X)
        nc = len(self.classes_)
        log_p = np.zeros((n, nc))

        for j, c in enumerate(self.classes_):
            log_p[:, j] = self.log_priors_[c]
            for i in range(self.n_features_):
                fd  = self.log_cond_[i][c]
                lp  = fd['val_lp']
                fb  = fd['fallback']
                log_p[:, j] += np.array([lp.get(int(v), fb) for v in X[:, i]])

        return log_p

    def predict_proba(self, X):
        """Trả về P(y=c | x) cho mỗi class."""
        log_p  = self.predict_log_proba(X)
        log_p -= log_p.max(axis=1, keepdims=True)   # numerical stability
        proba  = np.exp(log_p)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        if len(self.classes_) == 2:
            pos_idx = int(np.where(self.classes_ == 1)[0][0])
            return (proba[:, pos_idx] >= threshold).astype(int)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=int)))

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
